"""
Backtesting framework for core-periphery stock strategy.

Runs the Rossa core-periphery algorithm at regular intervals over a historical
period, allocating long positions to peripheral stocks and short positions to
core stocks. Tracks daily portfolio values and outputs results to Excel with visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict
import os

import rossa

RF_RATE = 0.03

def generate_rebalance_dates(
    start_date: str,
    end_date: str,
    interval_days: int
) -> List[pd.Timestamp]:
    """
    Generate rebalance dates at regular intervals within a date range.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        interval_days: Days between rebalances (e.g., 21 for monthly)
    
    Returns:
        List of rebalance dates
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=f'{interval_days}D')
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
    
    Args:
        results_df: DataFrame with columns ['Stock', 'Coreness']
    
    Returns:
        DataFrame with columns ['Stock', 'Coreness', 'Allocation']
    """
    # Determine if stock is core or peripheral based on median coreness
    results_df_without_IWM = results_df[results_df['Stock'] != 'IWM']
    
    # median_coreness = results_df_without_IWM['Coreness'].median()
    median_coreness = results_df_without_IWM['Coreness'].quantile(0.1)
    
    # Count how many core and peripheral stocks
    is_core = (results_df_without_IWM['Coreness'] >= median_coreness)
    n_core = is_core.sum()
    n_peripheral = (~is_core).sum()
    print(f"  → {n_core} core stocks, {n_peripheral} peripheral stocks (median coreness: {median_coreness:.4f})")
    
    allocations = []
    for _, row in results_df_without_IWM.iterrows():
        
        if row['Coreness'] >= median_coreness:
            # Core stock: divide -30% exposure equally among all core stocks
            allocation = -0.30 / n_core if n_core > 0 else 0.0
            pass
        else:
            # Peripheral stock: divide 130% exposure equally among all peripheral stocks
            allocation = 1.30 / n_peripheral if n_peripheral > 0 else 0.0
        
        allocations.append({
            'Stock': row['Stock'],
            'Coreness': row['Coreness'],
            'Allocation': allocation
        })
        
    # allocations.append({
    #     'Stock': 'IWM',
    #     'Coreness': -1,
    #     'Allocation': -0.3,
    # })
    
    return pd.DataFrame(allocations)


def get_rebalance_allocations(
    ticker_list: List[str],
    rebalance_dates: List[pd.Timestamp],
    lookback_days: int,
    cache_file: str = 'india_stocks_history.csv'
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
        
        print(f"\nRebalance on {rebalance_date.date()}: lookback [{lookback_start.date()} to {analysis_end_date.date()}]")
        
        # Run Rossa analysis for this period
        results = rossa.analyze_core_periphery(
            ticker_list=ticker_list,
            price_history_start_date=lookback_start.strftime('%Y-%m-%d'),
            price_history_end_date=analysis_end_date.strftime('%Y-%m-%d'),
            visualize_filename=None,  # Skip visualization for speed
            cache_file=cache_file
        )
        
        # Get allocations based on coreness
        allocations = allocate_by_coreness(results)
        
        # Add rebalance metadata
        allocations['RebalanceDate'] = rebalance_date
        allocations['Coreness_Rank'] = range(1, len(allocations) + 1)  # 1 = most peripheral
        
        rebalance_allocations[rebalance_date] = allocations
        
        print(f"  → {len(allocations)} stocks allocated")
    
    return rebalance_allocations


def get_active_allocation(
    date: pd.Timestamp,
    rebalance_schedule: Dict[pd.Timestamp, pd.DataFrame]
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
    return allocation_df[['Stock', 'Allocation']]


def calculate_portfolio_daily_values(
    prices: pd.DataFrame,
    rebalance_schedule: Dict[pd.Timestamp, pd.DataFrame],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0
) -> pd.DataFrame:
    """
    Calculate portfolio holdings and values using daily returns and weights.
    Fixed: Price gap handling and End-of-Day (EOD) position valuation.
    
    Two critical fixes:
    1. Forward-fill prices to prevent NaN gaps from dropping active positions
    2. Calculate End-of-Day position values so accounting balances perfectly
    
    Args:
        prices: DataFrame with stock prices (index: date, columns: tickers)
        rebalance_schedule: Dictionary mapping rebalance_date -> allocation_df
        start_date: Backtest start (YYYY-MM-DD)
        end_date: Backtest end (YYYY-MM-DD)
        initial_capital: Starting portfolio value (default $100k)
    
    Returns:
        DataFrame with columns:
        [Date, Ticker, Price, Allocation, Position_Value, Portfolio_Total_Value]
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Fix 1: Forward-fill prices to prevent NaN gaps from dropping active positions
    prices_filtered = prices[(prices.index >= start) & (prices.index <= end)].ffill().bfill().copy()
    
    # Calculate daily percentage returns
    daily_returns = prices_filtered.pct_change().fillna(0.0)
    
    daily_data = []
    current_portfolio_value = initial_capital
    
    for date in prices_filtered.index:
        allocation_df = get_active_allocation(date, rebalance_schedule)
        if allocation_df is None:
            continue
        
        portfolio_daily_return = 0.0
        day_records = []
        
        for _, allocation_row in allocation_df.iterrows():
            ticker = allocation_row['Stock']
            allocation_weight = allocation_row['Allocation']
            
            if ticker not in daily_returns.columns:
                continue
            
            stock_return = daily_returns.loc[date, ticker]
            
            # Contribution of this stock to portfolio's daily return
            portfolio_daily_return += allocation_weight * stock_return
            
            # Store interim data for End-of-Day calculation
            day_records.append({
                'Date': date,
                'Ticker': ticker,
                'Price': prices_filtered.loc[date, ticker],
                'Allocation': allocation_weight,
                'Stock_Return': stock_return
            })
        
        # Fix 2: Calculate End-of-Day portfolio value
        eod_portfolio_value = current_portfolio_value * (1.0 + portfolio_daily_return)
        
        # Calculate End-of-Day position values to match the EOD portfolio value
        for record in day_records:
            # EOD Position = (Start-of-day allocated capital) * (1 + stock's daily return)
            eod_position_value = (record['Allocation'] * current_portfolio_value) * (1.0 + record['Stock_Return'])
            
            daily_data.append({
                'Date': record['Date'],
                'Ticker': record['Ticker'],
                'Price': record['Price'],
                'Allocation': record['Allocation'],
                'Position_Value': eod_position_value,
                'Portfolio_Total_Value': eod_portfolio_value
            })
        
        # Roll forward capital for the next day
        current_portfolio_value = eod_portfolio_value
    
    return pd.DataFrame(daily_data)


def calculate_portfolio_metrics(daily_values_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
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
    portfolio_summary = daily_values_df[['Date', 'Portfolio_Total_Value']].drop_duplicates()
    portfolio_summary = portfolio_summary.sort_values('Date').reset_index(drop=True)
    portfolio_summary.columns = ['Date', 'Portfolio_Value']
    
    # Calculate returns
    portfolio_summary['Daily_Return'] = portfolio_summary['Portfolio_Value'].pct_change()
    portfolio_summary['Cumulative_Return'] = (1 + portfolio_summary['Daily_Return']).cumprod() - 1
    
    # Calculate performance metrics
    daily_returns = portfolio_summary['Daily_Return'].dropna()
    
    # Total return
    total_return = portfolio_summary['Cumulative_Return'].iloc[-1]
    
    # Volatility (annualized daily standard deviation)
    # Assuming 252 trading days per year
    daily_volatility = daily_returns.std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    
    # Sharpe Ratio (assuming 0% risk-free rate)
    # Sharpe = (Return - Risk_Free_Rate) / Volatility
    # Annualized return
    annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1
    annual_return = annual_return - RF_RATE
    sharpe_ratio = annual_return / annualized_volatility if annualized_volatility > 0 else 0.0
    
    # Sortino Ratio (only penalize downside volatility)
    # Sortino = (Return - Risk_Free_Rate) / Downside_Volatility
    downside_returns = daily_returns[daily_returns < 0]
    downside_volatility = np.sqrt((downside_returns ** 2).mean())
    downside_volatility_annual = downside_volatility * np.sqrt(252)
    sortino_ratio = annual_return / downside_volatility_annual if downside_volatility_annual > 0 else 0.0
    
    # Max drawdown
    running_max = portfolio_summary['Portfolio_Value'].expanding().max()
    drawdown_series = (running_max - portfolio_summary['Portfolio_Value']) / running_max
    max_drawdown = drawdown_series.max()
    
    summary_stats = {
        'Total_Return': total_return,
        'Volatility': annualized_volatility,
        'Sharpe_Ratio': sharpe_ratio,
        'Sortino_Ratio': sortino_ratio,
        'Max_Drawdown': max_drawdown
    }
    
    return portfolio_summary, summary_stats


def export_to_excel(
    daily_holdings: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    rebalance_schedule: Dict[pd.Timestamp, pd.DataFrame],
    summary_stats: Dict,
    output_filename: str = 'backtest_results.xlsx'
) -> None:
    """
    Export backtest results to Excel workbook.
    
    Creates four sheets:
    - "Daily Holdings": Daily position values
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
    last_date = daily_holdings['Date'].max()
    last_day_holdings = daily_holdings[daily_holdings['Date'] == last_date]
    total_allocations = last_day_holdings['Allocation'].sum()
    num_holdings = len(last_day_holdings)
    
    print(f"\n[DEBUG] Export integrity check:")
    print(f"  Last date: {last_date.date()}")
    print(f"  Holdings on last date: {num_holdings} stocks")
    print(f"  Sum of allocations: {total_allocations:.4f} (expected ~1.0)")
    print(f"  Total daily records: {len(daily_holdings)}")
    
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # Sheet 1: Daily Holdings
        daily_export = daily_holdings.copy()
        daily_export = daily_export[['Date', 'Ticker', 'Price', 'Allocation', 'Position_Value', 'Portfolio_Total_Value']]
        daily_export['Date'] = daily_export['Date'].dt.strftime('%Y-%m-%d')
        daily_export.to_excel(writer, sheet_name='Daily Holdings', index=False)
        
        # Sheet 2: Portfolio Summary
        portfolio_export = portfolio_summary.copy()
        portfolio_export['Date'] = portfolio_export['Date'].dt.strftime('%Y-%m-%d')
        portfolio_export['Daily_Return'] = portfolio_export['Daily_Return'].fillna(0)
        portfolio_export['Cumulative_Return'] = portfolio_export['Cumulative_Return'].fillna(0)
        portfolio_export.to_excel(writer, sheet_name='Portfolio Summary', index=False)
        
        # Sheet 3: Performance Metrics
        metrics_data = [
            ['Metric', 'Value'],
            ['Total Return', f"{summary_stats['Total_Return']*100:.2f}%"],
            ['Volatility (annualized)', f"{summary_stats['Volatility']*100:.2f}%"],
            ['Sharpe Ratio', f"{summary_stats['Sharpe_Ratio']:.4f}"],
            ['Sortino Ratio', f"{summary_stats['Sortino_Ratio']:.4f}"],
            ['Max Drawdown', f"{summary_stats['Max_Drawdown']*100:.2f}%"]
        ]
        metrics_df = pd.DataFrame(metrics_data[1:], columns=metrics_data[0])
        metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
        
        # Sheet 4: Rebalance Events
        rebalance_events = []
        for rebalance_date, allocation_df in rebalance_schedule.items():
            for _, row in allocation_df.iterrows():
                rebalance_events.append({
                    'Rebalance_Date': rebalance_date.strftime('%Y-%m-%d'),
                    'Ticker': row['Stock'],
                    'Coreness': row['Coreness'],
                    'Allocation': row['Allocation'],
                    'Action': 'SHORT' if row['Allocation'] < 0 else 'LONG'
                })
        
        rebalance_df = pd.DataFrame(rebalance_events)
        rebalance_df.to_excel(writer, sheet_name='Rebalance Events', index=False)
    
    
    print(f"\nExcel report exported: {output_filename}")


def plot_backtest_results(
    portfolio_summary: pd.DataFrame,
    output_dir: str = '.'
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
    axes[0].plot(portfolio_summary['Date'], portfolio_summary['Portfolio_Value'], linewidth=2, color='blue')
    axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Returns
    axes[1].plot(portfolio_summary['Date'], portfolio_summary['Cumulative_Return'] * 100, linewidth=2, color='green')
    axes[1].set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Cumulative Return (%)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 3: Drawdown from Peak (INVERTED - big drawdowns go DOWN)
    running_max = portfolio_summary['Portfolio_Value'].expanding().max()
    # Drawdown = (Current - Peak) / Peak (negative when price drops from peak)
    drawdown = (portfolio_summary['Portfolio_Value'] - running_max) / running_max * 100
    axes[2].fill_between(portfolio_summary['Date'], drawdown, 0, alpha=0.3, color='red')
    axes[2].plot(portfolio_summary['Date'], drawdown, linewidth=2, color='darkred')
    axes[2].set_title('Drawdown from Peak (% - inverted, so down = bad)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Drawdown (%)')
    axes[2].set_ylim([min(drawdown.min(), -1), 1])  # Ensure 0 line is visible
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'backtest_plots.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plots saved: {output_file}")
    plt.close()


def run_backtest(
    ticker_list: List[str],
    start_backtest_date: str,
    end_backtest_date: str,
    lookback_days: int,
    rebalance_interval_days: int,
    cache_file: str = 'india_stocks_history.csv',
    output_excel: str = 'backtest_results.xlsx',
    output_plots: bool = True
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
    
    Returns:
        Portfolio summary DataFrame
    """
    print("\n" + "="*70)
    print("BACKTEST: CORE-PERIPHERY STRATEGY")
    print("="*70)
    print(f"Backtest period: {start_backtest_date} to {end_backtest_date}")
    print(f"Lookback period: {lookback_days} days")
    print(f"Rebalance interval: {rebalance_interval_days} days")
    print(f"Tickers: {len(ticker_list)}")
    
    # Step 1: Fetch price data
    print("\n[1/5] Fetching price data...")
    price_data = rossa.fetch_and_cache_stock_data(ticker_list, cache_file)
    
    # Step 2: Generate rebalance dates
    print("[2/5] Generating rebalance schedule...")
    rebalance_dates = generate_rebalance_dates(
        start_backtest_date,
        end_backtest_date,
        rebalance_interval_days
    )
    print(f"  → {len(rebalance_dates)} rebalance dates generated")
    
    # Step 3: Compute allocations for each rebalance
    print("[3/5] Computing allocations for each rebalance...")
    rebalance_schedule = get_rebalance_allocations(
        ticker_list,
        rebalance_dates,
        lookback_days,
        cache_file
    )
    
    # Step 4: Calculate daily portfolio values
    print("[4/5] Calculating daily portfolio values...")
    daily_holdings = calculate_portfolio_daily_values(
        price_data,
        rebalance_schedule,
        start_backtest_date,
        end_backtest_date
    )
    
    # Step 5: Calculate metrics and export
    print("[5/5] Generating reports...")
    portfolio_summary, summary_stats = calculate_portfolio_metrics(daily_holdings)
    
    # Export Excel
    export_to_excel(daily_holdings, portfolio_summary, rebalance_schedule, summary_stats, output_excel)
    
    # Generate plots
    if output_plots:
        plot_backtest_results(portfolio_summary)
    
    # Print summary statistics
    print("\n" + "-"*70)
    print("BACKTEST SUMMARY")
    print("-"*70)
    print(f"Total Return: {summary_stats['Total_Return']*100:.2f}%")
    print(f"Volatility (annualized): {summary_stats['Volatility']*100:.2f}%")
    print(f"Sharpe Ratio: {summary_stats['Sharpe_Ratio']:.4f}")
    print(f"Sortino Ratio: {summary_stats['Sortino_Ratio']:.4f}")
    print(f"Max Drawdown: {summary_stats['Max_Drawdown']*100:.2f}%")
    print(f"Final Portfolio Value: ${portfolio_summary['Portfolio_Value'].iloc[-1]:.2f}")
    print("="*70 + "\n")
    
    return portfolio_summary


def main() -> None:
    """
    Main entry point. Load tickers and run backtest.
    """
    # Load tickers from file
    if not os.path.exists('tickers.txt'):
        print("Error: tickers.txt not found")
        return
    
    with open('tickers.txt', 'r') as f:
        ticker_list = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(ticker_list)} tickers from tickers.txt")
    
    # Run backtest with default parameters
    # 5 year backtest, 6-month lookback, monthly (21-day) rebalance
    results = run_backtest(
        ticker_list=ticker_list,
        start_backtest_date='2018-04-13',  # 5 years of data
        end_backtest_date='2026-04-13',
        lookback_days=126,  # ~6 months
        rebalance_interval_days=21,  # ~monthly
        cache_file='india_stocks_history.csv',
        output_excel='backtest_results.xlsx',
        output_plots=True
    )


if __name__ == "__main__":
    main()
