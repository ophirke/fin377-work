"""
Backtesting framework for core-periphery stock strategy.

Runs the Rossa core-periphery algorithm at regular intervals over a historical
period, allocating long positions to peripheral stocks and short positions to
core stocks. Tracks daily portfolio values and outputs results to Excel with visualizations.
"""

import argparse
import logging
import os
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rossa
from factor import compute_factor_loadings, load_factor_data

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for verbose output

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

RF_RATE = 0.03


def generate_rebalance_dates(
    start_date: str, end_date: str, interval_days: int
) -> List[pd.Timestamp]:
    """
    Generate rebalance dates at regular intervals within a date range.

    Uses BUSINESS DAYS (not calendar days) to align with trading days.
    This ensures rebalances happen on tradeable dates.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval_days: Business days between rebalances (e.g., 21 for ~monthly)

    Returns:
        List of rebalance dates (all on weekdays)
    """
    # Use 'B' for business days instead of 'D' for calendar days
    dates = pd.date_range(start=start_date, end=end_date, freq=f"{interval_days}B")
    return list(dates)


def allocate_by_coreness(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps coreness scores to portfolio allocations.

    Strategy:
    - Divide total short exposure (-30%) equally among core stocks
    - Divide total long exposure (+130%) equally among peripheral stocks
    - Net exposure: 100% long
    - Gross exposure: 160%

    CRITICAL FIX: Allocations are divided by stock count to ensure total exposure
    equals target (not 30% per core stock times number of core stocks).

    Ranking: Stocks are ranked by coreness (1 = most peripheral, N = most core).

    Args:
        results_df: DataFrame with columns ['Stock', 'Coreness']

    Returns:
        DataFrame with columns ['Stock', 'Coreness', 'Allocation', 'Coreness_Rank']
    """
    # Determine if stock is core or peripheral based on coreness score
    results_df_without_IWM = results_df[results_df["Stock"] != "IWM"]

    # Use 20th percentile as threshold: stocks in lowest 20% are PERIPHERAL
    quantile_prop = 0.2
    coreness_threshold = results_df_without_IWM["Coreness"].quantile(quantile_prop)

    # Count how many core and peripheral stocks
    is_core = results_df_without_IWM["Coreness"] >= coreness_threshold
    n_core = is_core.sum()
    n_peripheral = (~is_core).sum()
    logger.info(
        f"  → {n_peripheral} peripheral stocks (bottom {quantile_prop * 100:.0f}%), {n_core} core stocks (top {(1 - quantile_prop) * 100:.0f}%) at coreness threshold {coreness_threshold:.4f}"
    )

    allocations = []
    for _, row in results_df_without_IWM.iterrows():
        if row["Coreness"] >= coreness_threshold:
            # Core stock: divide -30% exposure equally among all core stocks
            allocation = -0.30 / n_core if n_core > 0 else 0.0
        else:
            # Peripheral stock: divide 130% exposure equally among all peripheral stocks
            allocation = 1.30 / n_peripheral if n_peripheral > 0 else 0.0

        allocations.append(
            {
                "Stock": row["Stock"],
                "Coreness": row["Coreness"],
                "Allocation": allocation,
            }
        )

    # Sort by coreness ascending (rank 1 = lowest coreness = most peripheral)
    allocations_df = pd.DataFrame(allocations)
    allocations_df = allocations_df.sort_values("Coreness", ascending=True).reset_index(
        drop=True
    )
    allocations_df["Coreness_Rank"] = range(1, len(allocations_df) + 1)

    return allocations_df


def get_rebalance_allocations(
    ticker_list: List[str],
    rebalance_dates: List[pd.Timestamp],
    lookback_days: int,
    cache_file: str = "india_stocks_history.csv",
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Compute allocations for each rebalance date using Rossa algorithm.

    Args:
        ticker_list: List of stock tickers
        rebalance_dates: List of rebalance dates
        lookback_days: Number of days of history to use for Rossa
        cache_file: Cache file for price data

    Returns:
        Dictionary mapping rebalance_date -> allocation DataFrame
    """
    rebalance_allocations = {}

    for rebalance_date in rebalance_dates:
        # Calculate lookback window
        lookback_start = rebalance_date - timedelta(days=lookback_days)
        # CRITICAL: Shift end date back 1 day to avoid lookahead bias
        # (don't use today's data to make today's trades)
        analysis_end_date = rebalance_date - timedelta(days=1)

        logger.info(
            f"\nRebalance on {rebalance_date.date()}: lookback [{lookback_start.date()} to {analysis_end_date.date()}]"
        )

        # Run Rossa analysis for this period
        results = rossa.analyze_core_periphery(
            ticker_list=ticker_list,
            price_history_start_date=lookback_start.strftime("%Y-%m-%d"),
            price_history_end_date=analysis_end_date.strftime("%Y-%m-%d"),
            visualize_filename=None,  # Skip visualization for speed
            cache_file=cache_file,
        )

        # Get allocations based on coreness
        allocations = allocate_by_coreness(results)

        # Add rebalance metadata
        allocations["RebalanceDate"] = rebalance_date
        allocations["Coreness_Rank"] = range(
            1, len(allocations) + 1
        )  # 1 = most peripheral

        rebalance_allocations[rebalance_date] = allocations

        logger.info(f"  → {len(allocations)} stocks allocated")

    return rebalance_allocations


def get_active_allocation(
    date: pd.Timestamp, rebalance_schedule: Dict[pd.Timestamp, pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """
    Get the active allocation for a given date.

    Returns allocation from most recent rebalance on or before the date.

    Args:
        date: Target date
        rebalance_schedule: Dictionary mapping rebalance_date -> allocation_df

    Returns:
        Allocation DataFrame or None if no prior rebalance
    """
    valid_dates = [d for d in rebalance_schedule.keys() if d <= date]
    if not valid_dates:
        return None

    most_recent = max(valid_dates)
    allocation_df = rebalance_schedule[most_recent].copy()
    # Remove the RebalanceDate and Coreness_Rank columns for calculation
    return allocation_df[["Stock", "Allocation"]]


def calculate_portfolio_daily_values(
    prices: pd.DataFrame,
    rebalance_schedule: Dict[pd.Timestamp, pd.DataFrame],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
) -> pd.DataFrame:
    """
    Calculate portfolio holdings and values with proper position tracking.

    CRITICAL FIXES:
    1. Track actual number of shares (not just target weights) to avoid daily rebalancing illusion
    2. Let position weights DRIFT between rebalance dates (only rebalance on scheduled dates)
    3. Use only ffill() for prices to avoid lookahead bias from bfill()
    4. Track cash balance to handle uninvested capital from failed rebalances

    Args:
        prices: DataFrame with stock prices (index: date, columns: tickers)
        rebalance_schedule: Dictionary mapping rebalance_date -> allocation_df
        start_date: Backtest start (YYYY-MM-DD)
        end_date: Backtest end (YYYY-MM-DD)
        initial_capital: Starting portfolio value (default $100k)

    Returns:
        DataFrame with columns:
        [Date, Ticker, Price, Position_Value, Portfolio_Total_Value]
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Fix lookahead bias: Only use ffill(), never bfill()
    # This prevents projecting prices backward in time for stocks that join later
    prices_filtered = (
        prices[(prices.index >= start) & (prices.index <= end)].ffill().copy()
    )

    daily_data = []
    current_portfolio_value = initial_capital
    current_shares = {}  # Ticker -> number of shares held
    current_cash = initial_capital  # Cash balance (uninvested capital)
    last_rebalance_date = None

    for date in prices_filtered.index:
        # Find the most recent rebalance date on or before today
        rebalance_dates_on_or_before = [
            d for d in rebalance_schedule.keys() if d <= date
        ]
        current_rebalance_date = (
            max(rebalance_dates_on_or_before) if rebalance_dates_on_or_before else None
        )

        # If we've hit a NEW rebalance date, execute the rebalance
        if (
            current_rebalance_date is not None
            and current_rebalance_date != last_rebalance_date
        ):
            allocation_df = rebalance_schedule[current_rebalance_date].copy()
            current_shares = {}
            invested_capital = (
                0.0  # Track exactly how much we successfully invested/shorted
            )

            # Convert target allocations to shares at today's prices
            for _, row in allocation_df.iterrows():
                ticker = row["Stock"]
                allocation_weight = row["Allocation"]

                if ticker not in prices_filtered.columns:
                    continue

                target_price = prices_filtered.loc[date, ticker]
                if pd.isna(target_price) or target_price == 0:
                    # Failed to get price: cannot invest in this position
                    current_shares[ticker] = 0
                else:
                    # Calculate shares needed to achieve target allocation
                    target_dollars = allocation_weight * current_portfolio_value
                    current_shares[ticker] = target_dollars / target_price
                    invested_capital += (
                        target_dollars  # Track total successfully deployed
                    )

            # Any capital not successfully deployed goes to cash (prevents "cash leakage")
            # Ideally invested_capital == current_portfolio_value, so cash_balance = 0.
            # But if some stocks have bad data, cash_balance absorbs the undeployed capital.
            current_cash = current_portfolio_value - invested_capital
            last_rebalance_date = current_rebalance_date

        # Skip if no active positions yet
        if not current_shares:
            continue

        # Calculate portfolio value: start with cash, then add all position values
        portfolio_value = current_cash
        day_records = []

        for ticker, num_shares in current_shares.items():
            if num_shares == 0:
                continue

            if ticker not in prices_filtered.columns:
                continue

            current_price = prices_filtered.loc[date, ticker]

            if pd.isna(current_price):
                continue

            position_value = num_shares * current_price
            portfolio_value += position_value

            day_records.append(
                {
                    "Date": date,
                    "Ticker": ticker,
                    "Price": current_price,
                    "Position_Value": position_value,
                    "Portfolio_Total_Value": portfolio_value,  # Will be updated below
                }
            )

        # Update all records with final portfolio value (cash + all positions)
        for record in day_records:
            record["Portfolio_Total_Value"] = portfolio_value
            daily_data.append(record)

        # Update portfolio value for next day
        current_portfolio_value = portfolio_value

    return pd.DataFrame(daily_data)


def calculate_portfolio_metrics(
    daily_values_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate daily portfolio metrics and summary statistics.

    Args:
        daily_values_df: Output from calculate_portfolio_daily_values

    Returns:
        Tuple of (daily_metrics_df, summary_stats_dict)
        - daily_metrics_df: DataFrame with columns:
          [Date, Portfolio_Value, Daily_Return, Cumulative_Return]
        - summary_stats_dict: Dict with keys:
          [Total_Return, Volatility, Sharpe_Ratio, Sortino_Ratio, Max_Drawdown]
    """
    # Get unique dates and portfolio values
    portfolio_summary = daily_values_df[
        ["Date", "Portfolio_Total_Value"]
    ].drop_duplicates()
    portfolio_summary = portfolio_summary.sort_values("Date").reset_index(drop=True)
    portfolio_summary.columns = ["Date", "Portfolio_Value"]

    # Calculate returns
    portfolio_summary["Daily_Return"] = portfolio_summary[
        "Portfolio_Value"
    ].pct_change()
    portfolio_summary["Cumulative_Return"] = (
        1 + portfolio_summary["Daily_Return"]
    ).cumprod() - 1

    # Calculate performance metrics
    daily_returns = portfolio_summary["Daily_Return"].dropna()

    # Guard against insufficient data
    if len(daily_returns) == 0:
        logger.warning("No valid daily returns. Returning zeros for all metrics.")
        summary_stats = {
            "Total_Return": 0.0,
            "Annualized_Return": 0.0,
            "Arithmetic_Annual_Return": 0.0,
            "Volatility": 0.0,
            "Volatility_Drag": 0.0,
            "Sharpe_Ratio": 0.0,
            "Sortino_Ratio": 0.0,
            "Max_Drawdown": 0.0,
        }
        return portfolio_summary, summary_stats

    # Total return
    total_return = portfolio_summary["Cumulative_Return"].iloc[-1]

    # Volatility (annualized daily standard deviation)
    # Assuming 252 trading days per year
    daily_volatility = daily_returns.std()
    annualized_volatility = daily_volatility * np.sqrt(252)

    # Sharpe Ratio (assuming 0% risk-free rate)
    # Sharpe = (Return - Risk_Free_Rate) / Volatility
    # Annualized return (geometric)
    annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1

    # Arithmetic average return (annualized)
    arithmetic_mean_daily = daily_returns.mean()
    arithmetic_annual_return = arithmetic_mean_daily * 252

    # Volatility drag
    volatility_drag = (annualized_volatility**2) / 2

    # For Sharpe, use excess return over risk-free rate
    annual_return_excess = annual_return - RF_RATE
    sharpe_ratio = (
        annual_return_excess / annualized_volatility
        if annualized_volatility > 0
        else 0.0
    )

    # Sortino Ratio (only penalize downside volatility)
    # Sortino = (Return - Risk_Free_Rate) / Downside_Volatility
    # Downside volatility uses semi-deviation: sqrt(mean(min(r - target, 0)²)) over full N
    target = 0.0
    downside_diff = np.minimum(daily_returns - target, 0.0)
    downside_volatility = np.sqrt((downside_diff**2).mean())
    downside_volatility_annual = downside_volatility * np.sqrt(252)
    sortino_ratio = (
        annual_return_excess / downside_volatility_annual
        if downside_volatility_annual > 0
        else 0.0
    )

    # Max drawdown
    running_max = portfolio_summary["Portfolio_Value"].expanding().max()
    drawdown_series = (running_max - portfolio_summary["Portfolio_Value"]) / running_max
    max_drawdown = drawdown_series.max()

    summary_stats = {
        "Total_Return": total_return,
        "Annualized_Return": annual_return,
        "Arithmetic_Annual_Return": arithmetic_annual_return,
        "Volatility": annualized_volatility,
        "Volatility_Drag": volatility_drag,
        "Sharpe_Ratio": sharpe_ratio,
        "Sortino_Ratio": sortino_ratio,
        "Max_Drawdown": max_drawdown,
    }

    return portfolio_summary, summary_stats


def export_to_excel(
    daily_holdings: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    rebalance_schedule: Dict[pd.Timestamp, pd.DataFrame],
    summary_stats: Dict,
    output_filename: str = "backtest_results.xlsx",
) -> None:
    """
    Export backtest results to Excel workbook.

    Creates four sheets:
    - "Daily Holdings": Daily position values (price-based tracking)
    - "Portfolio Summary": Portfolio performance metrics
    - "Performance Metrics": Key risk/return statistics
    - "Rebalance Events": Rebalance decisions

    Args:
        daily_holdings: DataFrame from calculate_portfolio_daily_values
        portfolio_summary: DataFrame from calculate_portfolio_metrics
        rebalance_schedule: Dictionary of rebalance allocations
        summary_stats: Dict with performance metrics from calculate_portfolio_metrics
        output_filename: Output Excel filename
    """
    # DEBUG: Check data integrity before export
    last_date = daily_holdings["Date"].max()
    last_day_holdings = daily_holdings[daily_holdings["Date"] == last_date]
    num_holdings = len(last_day_holdings)
    last_portfolio_value = (
        last_day_holdings["Portfolio_Total_Value"].iloc[0]
        if len(last_day_holdings) > 0
        else 0
    )

    logger.debug("Export integrity check:")
    logger.debug(f"  Last date: {last_date.date()}")
    logger.debug(f"  Holdings on last date: {num_holdings} stocks")
    logger.debug(f"  Final portfolio value: ${last_portfolio_value:,.2f}")
    logger.debug(f"  Total daily records: {len(daily_holdings)}")

    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        # Sheet 1: Daily Holdings
        daily_export = daily_holdings.copy()
        daily_export = daily_export[
            ["Date", "Ticker", "Price", "Position_Value", "Portfolio_Total_Value"]
        ]
        daily_export["Date"] = daily_export["Date"].dt.strftime("%Y-%m-%d")
        daily_export.to_excel(writer, sheet_name="Daily Holdings", index=False)

        # Sheet 2: Portfolio Summary
        portfolio_export = portfolio_summary.copy()
        portfolio_export["Date"] = portfolio_export["Date"].dt.strftime("%Y-%m-%d")
        portfolio_export["Daily_Return"] = portfolio_export["Daily_Return"].fillna(0)
        portfolio_export["Cumulative_Return"] = portfolio_export[
            "Cumulative_Return"
        ].fillna(0)
        portfolio_export.to_excel(writer, sheet_name="Portfolio Summary", index=False)

        # Sheet 3: Performance Metrics
        metrics_data = [
            ["Metric", "Value"],
            ["Total Return", f"{summary_stats['Total_Return'] * 100:.2f}%"],
            [
                "Annualized Return (geometric)",
                f"{summary_stats['Annualized_Return'] * 100:.2f}%",
            ],
            [
                "Arithmetic Average Return (annualized)",
                f"{summary_stats['Arithmetic_Annual_Return'] * 100:.2f}%",
            ],
            ["Volatility (annualized)", f"{summary_stats['Volatility'] * 100:.2f}%"],
            ["Volatility Drag", f"{summary_stats['Volatility_Drag'] * 100:.2f}%"],
            ["Sharpe Ratio", f"{summary_stats['Sharpe_Ratio']:.4f}"],
            ["Sortino Ratio", f"{summary_stats['Sortino_Ratio']:.4f}"],
            ["Max Drawdown", f"{summary_stats['Max_Drawdown'] * 100:.2f}%"],
        ]
        metrics_df = pd.DataFrame(metrics_data[1:], columns=metrics_data[0])
        metrics_df.to_excel(writer, sheet_name="Performance Metrics", index=False)

        # Sheet 4: Rebalance Events
        rebalance_events = []
        for rebalance_date, allocation_df in rebalance_schedule.items():
            for _, row in allocation_df.iterrows():
                rebalance_events.append(
                    {
                        "Rebalance_Date": rebalance_date.strftime("%Y-%m-%d"),
                        "Ticker": row["Stock"],
                        "Coreness": row["Coreness"],
                        "Allocation": row["Allocation"],
                        "Action": "SHORT" if row["Allocation"] < 0 else "LONG",
                    }
                )

        rebalance_df = pd.DataFrame(rebalance_events)
        rebalance_df.to_excel(writer, sheet_name="Rebalance Events", index=False)

    logger.info(f"Excel report exported: {output_filename}")


def plot_backtest_results(
    portfolio_summary: pd.DataFrame, output_dir: str = "."
) -> None:
    """
    Create visualization plots for backtest results.

    Generates:
    - Portfolio value over time
    - Cumulative returns
    - Drawdown from peak (inverted so big drawdowns go DOWN on graph)

    Args:
        portfolio_summary: DataFrame from calculate_portfolio_metrics
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Portfolio Value
    axes[0].plot(
        portfolio_summary["Date"],
        portfolio_summary["Portfolio_Value"],
        linewidth=2,
        color="blue",
    )
    axes[0].set_title("Portfolio Value Over Time", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Cumulative Returns
    axes[1].plot(
        portfolio_summary["Date"],
        portfolio_summary["Cumulative_Return"] * 100,
        linewidth=2,
        color="green",
    )
    axes[1].set_title("Cumulative Returns (%)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Cumulative Return (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.5)

    # Plot 3: Drawdown from Peak (INVERTED - big drawdowns go DOWN)
    running_max = portfolio_summary["Portfolio_Value"].expanding().max()
    # Drawdown = (Current - Peak) / Peak (negative when price drops from peak)
    drawdown = (portfolio_summary["Portfolio_Value"] - running_max) / running_max * 100
    axes[2].fill_between(portfolio_summary["Date"], drawdown, 0, alpha=0.3, color="red")
    axes[2].plot(portfolio_summary["Date"], drawdown, linewidth=2, color="darkred")
    axes[2].set_title(
        "Drawdown from Peak (% - inverted, so down = bad)",
        fontsize=14,
        fontweight="bold",
    )
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Drawdown (%)")
    axes[2].set_ylim([min(drawdown.min(), -1), 1])  # Ensure 0 line is visible
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "backtest_plots.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Plots saved: {output_file}")
    plt.close()


def export_summary_txt(
    summary_stats: Dict,
    portfolio_summary: pd.DataFrame,
    output_filename: str = "summary.txt",
) -> None:
    """
    Export backtest summary statistics to a simple text file.

    Args:
        summary_stats: Dict with performance metrics from calculate_portfolio_metrics
        portfolio_summary: DataFrame from calculate_portfolio_metrics
        output_filename: Output text filename
    """
    with open(output_filename, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("BACKTEST SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Total Return:                    {summary_stats['Total_Return'] * 100:>12.2f}%\n"
        )
        f.write(
            f"Annualized Return (geometric):   {summary_stats['Annualized_Return'] * 100:>12.2f}%\n"
        )
        f.write(
            f"Arithmetic Average Return:       {summary_stats['Arithmetic_Annual_Return'] * 100:>12.2f}%\n"
        )
        f.write(
            f"Volatility (annualized):         {summary_stats['Volatility'] * 100:>12.2f}%\n"
        )
        f.write(
            f"Volatility Drag:                 {summary_stats['Volatility_Drag'] * 100:>12.2f}%\n"
        )
        f.write("\n")

        f.write("RISK-ADJUSTED METRICS\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Sharpe Ratio:                    {summary_stats['Sharpe_Ratio']:>12.4f}\n"
        )
        f.write(
            f"Sortino Ratio:                   {summary_stats['Sortino_Ratio']:>12.4f}\n"
        )
        f.write(
            f"Max Drawdown:                    {summary_stats['Max_Drawdown'] * 100:>12.2f}%\n"
        )
        f.write("\n")

        f.write("PORTFOLIO VALUES\n")
        f.write("-" * 70 + "\n")
        f.write(f"Initial Capital:                 ${100000.0:>12,.2f}\n")
        f.write(
            f"Final Portfolio Value:           ${portfolio_summary['Portfolio_Value'].iloc[-1]:>12,.2f}\n"
        )
        f.write("\n")

        f.write("=" * 70 + "\n")

    logger.info(f"Summary exported: {output_filename}")


def compute_factor_loadings_over_time(
    portfolio_summary: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    factor_lookback_days: int,
    factor_list: Optional[List[str]] = None,
    factor_data_file: str = "factor_returns.xlsx",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Compute factor loadings at each rebalance date using a rolling lookback window.

    For each rebalance date, calculates portfolio returns for the lookback period
    and computes factor exposures via OLS regression.

    Args:
        portfolio_summary: DataFrame from calculate_portfolio_metrics with daily returns
        rebalance_dates: List of rebalance dates
        factor_lookback_days: Number of days to look back for factor analysis
        factor_list: Optional list of factor names; if None, uses all available factors
        factor_data_file: Path to Excel file with factor returns

    Returns:
        Tuple of (factor_loadings_df, rsquared_series)
        - factor_loadings_df: Each row is a rebalance date, columns are factors
        - rsquared_series: R² value for each rebalance date
    """
    logger.info("=" * 70)
    logger.info("FACTOR ANALYSIS: COMPUTING LOADINGS AT EACH REBALANCE")
    logger.info("=" * 70)
    logger.info(f"Lookback period: {factor_lookback_days} days")
    logger.info(f"Rebalance dates: {len(rebalance_dates)}")
    logger.info(f"Factor data file: {factor_data_file}")

    # Load factor data
    try:
        factor_returns = load_factor_data(factor_data_file, sheet_name=0)
    except FileNotFoundError:
        logger.warning(f"Factor data file not found: {factor_data_file}")
        logger.warning("Skipping factor analysis.")
        return None, None

    # Store loadings and R² for each rebalance
    loadings_records = []
    rsquared_values = []
    rebalance_dates_list = []

    for rebalance_date in rebalance_dates:
        # Calculate lookback window
        lookback_start = rebalance_date - timedelta(days=factor_lookback_days)

        # Get portfolio returns for the lookback period
        port_ret_window = portfolio_summary[
            (portfolio_summary["Date"] >= lookback_start)
            & (portfolio_summary["Date"] <= rebalance_date)
        ].copy()

        if len(port_ret_window) < 10:
            logger.debug(
                f"  {rebalance_date.date()}: Skipped (insufficient data: {len(port_ret_window)} days)"
            )
            continue

        port_ret_window = port_ret_window.sort_values("Date").set_index("Date")
        portfolio_returns_series = port_ret_window["Daily_Return"].dropna()

        if len(portfolio_returns_series) < 10:
            logger.debug(
                f"  {rebalance_date.date()}: Skipped (insufficient returns: {len(portfolio_returns_series)} days)"
            )
            continue

        # Compute factor loadings for this window
        try:
            loadings_df, diagnostics = compute_factor_loadings(
                portfolio_returns=portfolio_returns_series,
                factor_returns=factor_returns,
                factors=factor_list,
                market_factor="Market",
                confidence_threshold=0.95,
                min_factor_coverage=0.7,  # Allow factors with 70%+ coverage
            )

            # Extract loadings as a dict
            loading_dict = dict(zip(loadings_df["Factor"], loadings_df["Loading"]))
            loading_dict["Rebalance_Date"] = rebalance_date
            loadings_records.append(loading_dict)
            rsquared_values.append(diagnostics["r_squared"])
            rebalance_dates_list.append(rebalance_date)

            logger.info(
                f"  {rebalance_date.date()}: R²={diagnostics['r_squared']:.4f}, {len(loadings_df)} factors"
            )

        except Exception as e:
            logger.error(f"  {rebalance_date.date()}: Error - {str(e)}")
            continue

    if not loadings_records:
        logger.warning("No factor loadings computed. Skipping factor analysis.")
        return None, None

    # Convert to DataFrames (keep NaN for non-significant factors in each window)
    # Do NOT fillna(0) — NaN represents "not significant in this window", not zero loading
    factor_loadings_df = pd.DataFrame(loadings_records)

    # Filter to factors that appeared in at least 10% of windows (meaningful signal)
    min_appearances = int(0.1 * len(factor_loadings_df))
    factor_loadings_df = factor_loadings_df.dropna(axis=1, thresh=min_appearances)

    rsquared_series = pd.Series(rsquared_values, index=rebalance_dates_list)

    logger.info(
        f"Factor loadings computed for {len(factor_loadings_df)} rebalance dates"
    )
    logger.info(
        f"Factors with ≥10% appearance frequency: {len(factor_loadings_df.columns) - 1}"
    )  # -1 for Rebalance_Date

    return factor_loadings_df, rsquared_series


def plot_factor_loadings_multiline(
    factor_loadings_df: pd.DataFrame, output_file: str = "factor_loadings_multiline.png"
) -> None:
    """
    Plot factor loadings over time as a multiline chart (all factors on one chart).

    NaN values represent periods where the factor was not statistically significant
    and will appear as discontinuous lines.

    Args:
        factor_loadings_df: DataFrame from compute_factor_loadings_over_time
        output_file: Output PNG filename
    """
    plt.figure(figsize=(14, 7))

    # Drop Rebalance_Date column and plot
    df_plot = factor_loadings_df.drop(columns=["Rebalance_Date"], errors="ignore")
    dates = factor_loadings_df["Rebalance_Date"]

    for col in df_plot.columns:
        # Market factor gets emphasis; others get muted
        if col == "Market":
            alpha = 1.0
            lw = 2.5
        else:
            alpha = 0.6
            lw = 1.5

        plt.plot(
            dates,
            df_plot[col],
            marker="o",
            label=col,
            linewidth=lw,
            markersize=3,
            alpha=alpha,
        )

    plt.title("Factor Loadings Over Time (All Factors)", fontsize=14, fontweight="bold")
    plt.xlabel("Rebalance Date")
    plt.ylabel("Loading")
    plt.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Multiline factor plot saved: {output_file}")
    plt.close()


def plot_factor_loadings_subplots(
    factor_loadings_df: pd.DataFrame, output_file: str = "factor_loadings_subplots.png"
) -> None:
    """
    Plot factor loadings over time as subplots (one chart per factor).

    NaN values represent periods where the factor was not statistically significant.
    These appear as gaps in the line (honest representation of data availability).

    Args:
        factor_loadings_df: DataFrame from compute_factor_loadings_over_time
        output_file: Output PNG filename
    """
    df_plot = factor_loadings_df.drop(columns=["Rebalance_Date"], errors="ignore")
    dates = factor_loadings_df["Rebalance_Date"]

    n_factors = len(df_plot.columns)
    n_cols = 3
    n_rows = (n_factors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for idx, col in enumerate(df_plot.columns):
        ax = axes[idx]
        ax.plot(
            dates,
            df_plot[col],
            marker="o",
            linewidth=2,
            markersize=4,
            color="steelblue",
        )

        # Fill between only where data exists (skip NaN gaps)
        valid_mask = df_plot[col].notna()
        ax.fill_between(
            dates[valid_mask], df_plot[col][valid_mask], 0, alpha=0.3, color="steelblue"
        )

        ax.set_title(f"{col}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Rebalance Date")
        ax.set_ylabel("Loading")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=45)

    # Hide unused subplots
    for idx in range(len(df_plot.columns), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Subplot factor plot saved: {output_file}")
    plt.close()


def plot_factor_rsquared(
    rsquared_series: pd.Series, output_file: str = "factor_rsquared.png"
) -> None:
    """
    Plot R² values for factor models over time.

    Args:
        rsquared_series: Series from compute_factor_loadings_over_time
        output_file: Output PNG filename
    """
    plt.figure(figsize=(14, 6))

    plt.plot(
        rsquared_series.index,
        rsquared_series.values,
        marker="o",
        linewidth=2,
        markersize=6,
        color="darkgreen",
    )
    plt.fill_between(
        rsquared_series.index, rsquared_series.values, 0, alpha=0.3, color="green"
    )

    plt.title("Factor Model R² Over Time", fontsize=14, fontweight="bold")
    plt.xlabel("Rebalance Date")
    plt.ylabel("R²")
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"R² plot saved: {output_file}")
    plt.close()


def export_factor_analysis_to_excel(
    factor_loadings_df: pd.DataFrame,
    rsquared_series: pd.Series,
    output_filename: str = "factor_analysis.xlsx",
) -> None:
    """
    Export factor analysis results to Excel.

    Args:
        factor_loadings_df: DataFrame from compute_factor_loadings_over_time
        rsquared_series: Series from compute_factor_loadings_over_time
        output_filename: Output Excel filename
    """
    if factor_loadings_df is None:
        return

    with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
        # Sheet 1: Factor Loadings
        loadings_export = factor_loadings_df.copy()
        loadings_export["Rebalance_Date"] = loadings_export[
            "Rebalance_Date"
        ].dt.strftime("%Y-%m-%d")
        loadings_export.to_excel(writer, sheet_name="Factor Loadings", index=False)

        # Sheet 2: R² Values
        rsq_export = pd.DataFrame(
            {
                "Rebalance_Date": rsquared_series.index.strftime("%Y-%m-%d"),
                "R_Squared": rsquared_series.values,
            }
        )
        rsq_export.to_excel(writer, sheet_name="R_Squared", index=False)

    logger.info(f"Factor analysis exported: {output_filename}")


def run_backtest(
    ticker_list: List[str],
    start_backtest_date: str,
    end_backtest_date: str,
    lookback_days: int,
    rebalance_interval_days: int,
    cache_file: str = "india_stocks_history.csv",
    output_excel: str = "backtest_results.xlsx",
    output_plots: bool = True,
    factor_list: Optional[List[str]] = None,
    factor_lookback_days: Optional[int] = None,
    factor_data_file: str = "factor_returns.xlsx",
    summary_file: str = "backtest_summary.txt",
) -> pd.DataFrame:
    """
    Run complete backtest of core-periphery strategy.

    Args:
        ticker_list: List of stock tickers
        start_backtest_date: Backtest start (YYYY-MM-DD)
        end_backtest_date: Backtest end (YYYY-MM-DD)
        lookback_days: Days of history for Rossa analysis
        rebalance_interval_days: Days between rebalances
        cache_file: Price data cache file
        output_excel: Output Excel filename
        output_plots: Whether to generate plots
        factor_list: Optional list of factor names for factor analysis (None = use all)
        factor_lookback_days: Optional number of days for factor lookback window
        factor_data_file: Path to Excel file with factor returns
        summary_file: Path to text file for summary output
    Returns:
        Portfolio summary DataFrame
    """
    logger.info("=" * 70)
    logger.info("BACKTEST: CORE-PERIPHERY STRATEGY")
    logger.info("=" * 70)
    logger.info(f"Backtest period: {start_backtest_date} to {end_backtest_date}")
    logger.info(f"Lookback period: {lookback_days} days")
    logger.info(f"Rebalance interval: {rebalance_interval_days} days")
    logger.info(f"Tickers: {len(ticker_list)}")

    # Step 1: Fetch price data
    logger.info("[1/6] Fetching price data...")
    # Convert list to tuple for caching
    price_data = rossa.fetch_and_cache_stock_data(tuple(ticker_list), cache_file)

    # Verify data covers the required backtest period
    actual_start = price_data.index[0]
    actual_end = price_data.index[-1]
    logger.info(f"Data covers period: {actual_start.date()} to {actual_end.date()}")
    logger.info(f"Backtest will use: {start_backtest_date} to {end_backtest_date}")

    # Step 2: Generate rebalance dates
    logger.info("[2/6] Generating rebalance schedule...")
    rebalance_dates = generate_rebalance_dates(
        start_backtest_date, end_backtest_date, rebalance_interval_days
    )
    logger.info(f"  → {len(rebalance_dates)} rebalance dates generated")

    # Step 3: Compute allocations for each rebalance
    logger.info("[3/6] Computing allocations for each rebalance...")
    rebalance_schedule = get_rebalance_allocations(
        ticker_list, rebalance_dates, lookback_days, cache_file
    )

    # Step 4: Calculate daily portfolio values
    logger.info("[4/6] Calculating daily portfolio values...")
    daily_holdings = calculate_portfolio_daily_values(
        price_data, rebalance_schedule, start_backtest_date, end_backtest_date
    )

    # Step 5: Calculate metrics and export
    logger.info("[5/6] Generating reports...")
    portfolio_summary, summary_stats = calculate_portfolio_metrics(daily_holdings)

    # Export Excel
    if output_excel is not None:
        export_to_excel(
            daily_holdings,
            portfolio_summary,
            rebalance_schedule,
            summary_stats,
            output_excel,
        )

    # Export summary text file
    if summary_file is not None:
        export_summary_txt(summary_stats, portfolio_summary, summary_file)

    # Generate plots
    if output_plots:
        plot_backtest_results(portfolio_summary)

    # Print summary statistics
    print_backtest_summary(summary_stats, portfolio_summary)

    # Step 6: Factor analysis (if requested)
    factor_loadings_df = None
    rsquared_series = None

    if factor_lookback_days is not None and factor_lookback_days > 0:
        logger.info("[6/6] Computing factor analysis...")
        factor_loadings_df, rsquared_series = compute_factor_loadings_over_time(
            portfolio_summary=portfolio_summary,
            rebalance_dates=rebalance_dates,
            factor_lookback_days=factor_lookback_days,
            factor_list=factor_list,
            factor_data_file=factor_data_file,
        )

        if factor_loadings_df is not None:
            # Plot factor loadings
            plot_factor_loadings_multiline(
                factor_loadings_df, "factor_loadings_multiline.png"
            )
            plot_factor_loadings_subplots(
                factor_loadings_df, "factor_loadings_subplots.png"
            )
            plot_factor_rsquared(rsquared_series, "factor_rsquared.png")

            # Export to Excel
            export_factor_analysis_to_excel(
                factor_loadings_df, rsquared_series, "factor_analysis.xlsx"
            )

            # Print summary of most recent loadings
            logger.info("-" * 70)
            logger.info("MOST RECENT FACTOR LOADINGS (Latest Rebalance)")
            logger.info("-" * 70)
            most_recent_idx = len(factor_loadings_df) - 1
            most_recent_row = factor_loadings_df.iloc[most_recent_idx]
            most_recent_date = most_recent_row["Rebalance_Date"]
            most_recent_rsq = rsquared_series.iloc[most_recent_idx]

            logger.info(f"Date: {most_recent_date.date()}")
            logger.info(f"R²: {most_recent_rsq:.4f}")
            logger.info("Factor Loadings:")
            for col in factor_loadings_df.columns:
                if col != "Rebalance_Date":
                    loading = most_recent_row[col]
                    logger.info(f"  {col:20s}: {loading:>10.6f}")
            logger.info("=" * 70)

    return portfolio_summary, summary_stats


def print_backtest_summary(
    summary_stats: Dict, portfolio_summary: pd.DataFrame
) -> None:
    logger.info("-" * 70)
    logger.info("BACKTEST SUMMARY")
    logger.info("-" * 70)
    logger.info(f"Total Return: {summary_stats['Total_Return'] * 100:.2f}%")
    logger.info(
        f"Annualized Return (geometric): {summary_stats['Annualized_Return'] * 100:.2f}%"
    )
    logger.info(
        f"Arithmetic Average Return (annualized): {summary_stats['Arithmetic_Annual_Return'] * 100:.2f}%"
    )
    logger.info(f"Volatility (annualized): {summary_stats['Volatility'] * 100:.2f}%")
    logger.info(f"Volatility Drag: {summary_stats['Volatility_Drag'] * 100:.2f}%")
    logger.info(f"Sharpe Ratio: {summary_stats['Sharpe_Ratio']:.4f}")
    logger.info(f"Sortino Ratio: {summary_stats['Sortino_Ratio']:.4f}")
    logger.info(f"Max Drawdown: {summary_stats['Max_Drawdown'] * 100:.2f}%")
    logger.info(
        f"Final Portfolio Value: ${portfolio_summary['Portfolio_Value'].iloc[-1]:.2f}"
    )
    logger.info("=" * 70)


def plot_backtest_summary_over_time(
    summary_stats_over_time: List[Dict], output_file: str = "summary_over_time.png"
) -> None:
    """
    Plot summary statistics over time (e.g. annualized return, volatility, Sharpe).

    Args:
        summary_stats_over_time: List of summary stats dicts for each period
        output_file: Output PNG filename
    """
    if not summary_stats_over_time:
        logger.warning("No summary stats to plot.")
        return

    # Convert list of dicts to DataFrame
    summary_df = pd.DataFrame(summary_stats_over_time)

    # Ensure Date column is datetime
    if "Date" in summary_df.columns:
        summary_df["Date"] = pd.to_datetime(summary_df["Date"])
        summary_df = summary_df.sort_values("Date")
    else:
        logger.warning("No 'Date' column in summary stats. Cannot plot over time.")
        return

    if len(summary_df) < 2:
        logger.warning("Not enough data points to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Annualized Return
    axes[0].plot(
        summary_df["Date"],
        summary_df["Annualized_Return"] * 100,
        marker="o",
        linewidth=2,
        markersize=6,
        color="blue",
    )
    axes[0].set_title("Annualized Return Over Time", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Return (%)")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Volatility
    axes[1].plot(
        summary_df["Date"],
        summary_df["Volatility"] * 100,
        marker="o",
        linewidth=2,
        markersize=6,
        color="orange",
    )
    axes[1].set_title("Volatility Over Time", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Volatility (%)")
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Sharpe Ratio
    axes[2].plot(
        summary_df["Date"],
        summary_df["Sharpe_Ratio"],
        marker="o",
        linewidth=2,
        markersize=6,
        color="green",
    )
    axes[2].set_title("Sharpe Ratio Over Time", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Sharpe Ratio")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Summary over time plot saved: {output_file}")
    plt.close()


def main() -> None:
    # Load tickers from file
    if not os.path.exists("tickers.txt"):
        logger.error("tickers.txt not found")
        return

    with open("tickers.txt", "r") as f:
        ticker_list = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(ticker_list)} tickers from tickers.txt")

    # =================== PARAMS ==================
    
    lookback_days = 365
    rebalance_interval_days = 3
    
    # =================== PARAMS ==================

    # Run backtest with default parameters
    overall_start_date = "2008-04-13"
    overall_end_date = "2026-04-13"
    eval_lookback = pd.DateOffset(years=1)
    eval_interval = pd.DateOffset(months=1)
    last_eval_end_date = pd.to_datetime(overall_end_date)
    last_eval_start_date = last_eval_end_date - eval_lookback + pd.DateOffset(days=1)
    eval_date_pairs = [(last_eval_start_date, last_eval_end_date)]
    while eval_date_pairs[-1][0] > pd.to_datetime(overall_start_date):
        prev_start = eval_date_pairs[-1][0] - eval_interval
        prev_end = eval_date_pairs[-1][1] - eval_interval
        eval_date_pairs.append((prev_start, prev_end))

    portfolio_summary_over_time = []
    summary_stats_over_time = []
    for sub_start, sub_end in reversed(eval_date_pairs):
        portfolio_summary, summary_stats = run_backtest(
            ticker_list=ticker_list,
            start_backtest_date=sub_start,
            end_backtest_date=sub_end,
            lookback_days=lookback_days,
            rebalance_interval_days=rebalance_interval_days,
            cache_file="india_stocks_history.csv",
            output_excel=None,
            output_plots=False,
            factor_list=None,
            factor_lookback_days=None,
            factor_data_file="factor_returns.xlsx",
            summary_file=None,
        )
        # Add date to summary stats for tracking
        summary_stats["Date"] = pd.to_datetime(sub_start)
        portfolio_summary_over_time.append(portfolio_summary)
        summary_stats_over_time.append(summary_stats)

    plot_backtest_summary_over_time(summary_stats_over_time, "summary_over_time.png")

    full_portfolio_summary, full_summary_stats = run_backtest(
        ticker_list=ticker_list,
        start_backtest_date="2008-04-13",
        end_backtest_date="2026-04-13",
        lookback_days=lookback_days,
        rebalance_interval_days=rebalance_interval_days,
        cache_file="india_stocks_history.csv",
        # output_excel="backtest_results.xlsx",
        output_excel=None,
        output_plots=True,
        factor_list=None,  # Use all available factors
        factor_lookback_days=365 * 3,
        factor_data_file="factor_returns.xlsx",
    )
    
    print_backtest_summary(full_summary_stats, full_portfolio_summary)


if __name__ == "__main__":
    main()
