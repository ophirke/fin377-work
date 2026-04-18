"""
Backtesting framework for core-periphery stock strategy.

Runs the Rossa core-periphery algorithm at regular intervals over a historical
period, allocating long positions to peripheral stocks and short positions to
core stocks. Tracks daily portfolio values and outputs results to Excel with visualizations.
"""

import logging
import multiprocessing
import os
from dataclasses import dataclass, field, replace
from datetime import timedelta
from multiprocessing import Pool
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

import rossa
from data import clean_price_data, init_worker_lock
from datamarshal import (
    DataConfig,
    load_nasdaq100_constituents,
    load_sp100_constituents,
    load_sp500_constituents,
)
from factor import compute_factor_loadings, load_factor_data

NUM_WORKERS = max(1, min(12, multiprocessing.cpu_count() - 4))

# ============================================================================
# BACKTEST CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy-specific portfolio construction settings."""

    target_net_exposure: float = 1.0
    short_amount: float = 0.90
    periphery_threshold_quantile: float = 0.05
    long_periphery: bool = True
    weighting_method: str = "equal"  # Options: "equal", "coreness_proportional", "markowitz_min_vol"
    max_long_weight: Optional[float] = None
    max_short_weight: Optional[float] = None

    @property
    def effective_periphery_quantile(self) -> float:
        """Return the quantile used to split coreness into portfolio sides."""
        if self.long_periphery:
            return self.periphery_threshold_quantile
        return 1.0 - self.periphery_threshold_quantile

    @property
    def gross_long_exposure(self) -> float:
        """Total positive exposure target."""
        return self.target_net_exposure + self.short_amount

    @property
    def gross_short_exposure(self) -> float:
        """Total absolute short exposure target."""
        return self.short_amount

    @property
    def resolved_max_long_weight(self) -> float:
        """Per-name cap for long positions."""
        if self.max_long_weight is not None:
            return self.max_long_weight
        return self.gross_long_exposure

    @property
    def resolved_max_short_weight(self) -> float:
        """Per-name cap for short positions."""
        if self.max_short_weight is not None:
            return self.max_short_weight
        return self.gross_short_exposure


@dataclass(frozen=True)
class ExcelExportConfig:
    """Controls lossless Excel writer behavior."""

    use_constant_memory: bool = True


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    ticker_list: Union[List[str], Callable[[str], List[str]]]
    start_backtest_date: str
    end_backtest_date: str
    lookback_days: int
    rebalance_interval_days: int
    output_excel: Optional[str] = None
    output_plots: bool = False
    benchmark_tickers: Optional[List[str]] = None
    factor_list: Optional[List[str]] = None
    factor_lookback_days: Optional[int] = None
    factor_data_file: str = "factor_returns.xlsx"
    summary_file: Optional[str] = None
    parallel: bool = False
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    excel_export: ExcelExportConfig = field(default_factory=ExcelExportConfig)


@dataclass
class BacktestResult:
    """Results from a completed backtest run."""

    portfolio_summary: pd.DataFrame
    summary_stats: Dict
    benchmark_data: Optional[Dict[str, pd.DataFrame]] = None
    benchmark_metrics: Optional[Dict[str, Dict]] = None

    @property
    def total_return(self) -> float:
        """Total return of the strategy."""
        return self.summary_stats.get("Total_Return", 0.0)

    @property
    def annualized_return(self) -> float:
        """Annualized geometric return."""
        return self.summary_stats.get("Annualized_Return", 0.0)

    @property
    def arithmetic_annual_return(self) -> float:
        """Annualized arithmetic average return."""
        return self.summary_stats.get("Arithmetic_Annual_Return", 0.0)

    @property
    def volatility(self) -> float:
        """Annualized volatility."""
        return self.summary_stats.get("Volatility", 0.0)

    @property
    def sharpe_ratio(self) -> float:
        """Sharpe ratio."""
        return self.summary_stats.get("Sharpe_Ratio", 0.0)

    @property
    def sortino_ratio(self) -> float:
        """Sortino ratio."""
        return self.summary_stats.get("Sortino_Ratio", 0.0)

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown."""
        return self.summary_stats.get("Max_Drawdown", 0.0)

    @property
    def final_portfolio_value(self) -> float:
        """Final portfolio value."""
        if len(self.portfolio_summary) > 0:
            return self.portfolio_summary["Portfolio_Value"].iloc[-1]
        return 0.0

    @property
    def has_benchmarks(self) -> bool:
        """Whether benchmark data was computed."""
        return self.benchmark_data is not None and len(self.benchmark_data) > 0

    def benchmark_comparison(self) -> Optional[pd.DataFrame]:
        """Create a comparison DataFrame of strategy vs benchmarks."""
        if not self.has_benchmarks:
            return None

        results = []
        results.append(
            {
                "Name": "Strategy",
                "Total Return": f"{self.total_return * 100:.2f}%",
                "Annualized Return": f"{self.annualized_return * 100:.2f}%",
                "Volatility": f"{self.volatility * 100:.2f}%",
                "Sharpe Ratio": f"{self.sharpe_ratio:.4f}",
                "Sortino Ratio": f"{self.sortino_ratio:.4f}",
                "Max Drawdown": f"{self.max_drawdown * 100:.2f}%",
            }
        )

        for ticker, metrics in self.benchmark_metrics.items():
            results.append(
                {
                    "Name": ticker,
                    "Total Return": f"{metrics.get('Total_Return', 0) * 100:.2f}%",
                    "Annualized Return": f"{metrics.get('Annualized_Return', 0) * 100:.2f}%",
                    "Volatility": f"{metrics.get('Volatility', 0) * 100:.2f}%",
                    "Sharpe Ratio": f"{metrics.get('Sharpe_Ratio', 0):.4f}",
                    "Sortino Ratio": f"{metrics.get('Sortino_Ratio', 0):.4f}",
                    "Max Drawdown": f"{metrics.get('Max_Drawdown', 0) * 100:.2f}%",
                }
            )

        return pd.DataFrame(results)


@dataclass
class SingleBacktestRun:
    """A single backtest execution plan."""

    name: str
    config: BacktestConfig


@dataclass
class StepForwardBacktestRun:
    """A step-forward evaluation plan built from repeated backtests."""

    name: str
    base_config: BacktestConfig
    overall_start_date: str
    overall_end_date: str
    eval_lookback: pd.DateOffset
    eval_interval: pd.DateOffset
    summary_plot_filename: Optional[str] = None
    parallel: bool = True


@dataclass
class StepForwardBacktestResult:
    """Results from a full step-forward evaluation."""

    results_by_start_date: Dict[str, BacktestResult]
    summary_stats_over_time: List[Dict]


ConfiguredRun = Union[SingleBacktestRun, StepForwardBacktestRun]


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_LEVEL = logging.INFO

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


RF_RATE = 0.03


def calculate_benchmark_daily_values(
    prices: pd.DataFrame,
    benchmark_tickers: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate daily values for benchmark indices (equally weighted).

    Args:
        prices: DataFrame with all stock prices (index: date, columns: tickers)
        benchmark_tickers: List of benchmark tickers
        start_date: Backtest start (YYYY-MM-DD)
        end_date: Backtest end (YYYY-MM-DD)
        initial_capital: Starting capital (for benchmark comparison)

    Returns:
        Dictionary mapping benchmark_ticker -> daily_metrics_df with columns:
        [Date, Price, Portfolio_Value, Daily_Return, Cumulative_Return]
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    benchmark_data = {}

    for ticker in benchmark_tickers:
        if ticker not in prices.columns:
            logger.warning(f"Benchmark ticker {ticker} not found in price data")
            continue

        # Get benchmark prices for date range
        benchmark_prices = (
            prices[(prices.index >= start) & (prices.index <= end)][[ticker]]
            .ffill()
            .copy()
        )

        benchmark_prices = benchmark_prices.dropna()

        if benchmark_prices.empty:
            logger.warning(f"No data for benchmark {ticker} in date range")
            continue

        # Calculate portfolio value assuming buy-and-hold
        first_price = benchmark_prices.iloc[0, 0]
        benchmark_prices["Portfolio_Value"] = (
            benchmark_prices[ticker] / first_price * initial_capital
        )
        benchmark_prices["Daily_Return"] = benchmark_prices[ticker].pct_change()
        benchmark_prices["Cumulative_Return"] = (
            1 + benchmark_prices["Daily_Return"]
        ).cumprod() - 1

        # Rename columns for consistency
        result_df = benchmark_prices[
            ["Portfolio_Value", "Daily_Return", "Cumulative_Return"]
        ].copy()
        result_df.columns = ["Portfolio_Value", "Daily_Return", "Cumulative_Return"]
        result_df["Date"] = benchmark_prices.index
        result_df = result_df[
            ["Date", "Portfolio_Value", "Daily_Return", "Cumulative_Return"]
        ]
        result_df = result_df.reset_index(drop=True)

        benchmark_data[ticker] = result_df
        logger.info(f"Benchmark {ticker}: {len(result_df)} trading days")

    return benchmark_data


def calculate_benchmark_metrics(
    benchmark_data: Dict[str, pd.DataFrame],
) -> Dict[str, Dict]:
    """
    Calculate metrics for all benchmarks.

    Args:
        benchmark_data: Dictionary mapping benchmark_ticker -> daily_metrics_df

    Returns:
        Dictionary mapping benchmark_ticker -> summary_stats_dict
    """
    benchmark_metrics = {}

    for ticker, daily_metrics in benchmark_data.items():
        daily_returns = daily_metrics["Daily_Return"].dropna()

        if len(daily_returns) == 0:
            logger.warning(f"No valid returns for benchmark {ticker}")
            continue

        # Total return
        total_return = daily_metrics["Cumulative_Return"].iloc[-1]

        # Volatility (annualized)
        daily_volatility = daily_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(252)

        # Annualized return
        annual_return = (1 + total_return) ** (252 / len(daily_returns)) - 1

        # Arithmetic average return (annualized)
        arithmetic_annual_return = daily_returns.mean() * 252

        # Volatility drag
        volatility_drag = (annualized_volatility**2) / 2

        # Sharpe Ratio
        annual_return_excess = annual_return - RF_RATE
        sharpe_ratio = (
            annual_return_excess / annualized_volatility
            if annualized_volatility > 0
            else 0.0
        )

        # Sortino Ratio
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
        running_max = daily_metrics["Portfolio_Value"].expanding().max()
        drawdown_series = (running_max - daily_metrics["Portfolio_Value"]) / running_max
        max_drawdown = drawdown_series.max()

        benchmark_metrics[ticker] = {
            "Total_Return": total_return,
            "Annualized_Return": annual_return,
            "Arithmetic_Annual_Return": arithmetic_annual_return,
            "Volatility": annualized_volatility,
            "Volatility_Drag": volatility_drag,
            "Sharpe_Ratio": sharpe_ratio,
            "Sortino_Ratio": sortino_ratio,
            "Max_Drawdown": max_drawdown,
        }

    return benchmark_metrics


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


def _compute_allocation_for_rebalance_date(
    args: Tuple,
) -> Tuple[pd.Timestamp, pd.DataFrame]:
    """
    Worker function for parallel allocation computation.

    Executed by multiprocessing pool to compute allocations for a single rebalance date.

    Args:
        args: Tuple of (rebalance_date, ticker_list, lookback_days)

    Returns:
        Tuple of (rebalance_date, allocations_df)
    """
    rebalance_date, ticker_list, lookback_days, strategy = args

    # Get tickers for this rebalance date (dynamic or static)
    if callable(ticker_list):
        current_tickers = list(ticker_list(rebalance_date.strftime("%Y-%m-%d")))
    else:
        current_tickers = ticker_list

    # Calculate lookback window
    lookback_start = rebalance_date - timedelta(days=lookback_days)
    # CRITICAL: Shift end date back 1 day to avoid lookahead bias
    analysis_end_date = rebalance_date - timedelta(days=1)

    logger.info(
        f"\nRebalance on {rebalance_date.date()}: lookback [{lookback_start.date()} to {analysis_end_date.date()}]"
    )

    _, log_returns = rossa.load_analysis_price_data(
        ticker_list=current_tickers,
        price_history_start_date=lookback_start.strftime("%Y-%m-%d"),
        price_history_end_date=analysis_end_date.strftime("%Y-%m-%d"),
    )
    A, ticker_names = rossa.build_adjacency_matrix(log_returns)
    results = rossa.rossa_core_periphery(A, ticker_names)

    # Get allocations based on coreness
    allocations = allocate_by_coreness(
        results, strategy_config=strategy, log_returns=log_returns
    )

    # Add rebalance metadata
    allocations["RebalanceDate"] = rebalance_date
    allocations["Coreness_Rank"] = range(1, len(allocations) + 1)  # 1 = most peripheral

    logger.info(f"  → {len(allocations)} stocks allocated")

    return rebalance_date, allocations


def _allocate_by_coreness_equal_legacy(
    results_df: pd.DataFrame,
    strategy_config: Optional[StrategyConfig] = None,
) -> pd.DataFrame:
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

    # Use 20th percentile as threshold: stocks in lowest 20% are PERIPHERAL
    strategy = strategy_config or StrategyConfig()
    quantile_prop = strategy.effective_periphery_quantile
    coreness_threshold = results_df["Coreness"].quantile(quantile_prop)

    # Count how many core and peripheral stocks
    is_core = results_df["Coreness"] >= coreness_threshold
    n_core = is_core.sum()
    n_peripheral = (~is_core).sum()
    logger.info(
        f"  → {n_peripheral} peripheral stocks (bottom {quantile_prop * 100:.0f}%), {n_core} core stocks (top {(1 - quantile_prop) * 100:.0f}%) at coreness threshold {coreness_threshold:.4f}"
    )

    allocations = []

    for _, row in results_df.iterrows():
        core_alloc = -strategy.short_amount
        periphery_alloc = 1.0 - core_alloc
        if not strategy.long_periphery:
            core_alloc, periphery_alloc = periphery_alloc, core_alloc
        core_alloc = core_alloc / n_core if n_core > 0 else 0.0
        periphery_alloc = periphery_alloc / n_peripheral if n_peripheral > 0 else 0.0
        allocation = (
            core_alloc if row["Coreness"] >= coreness_threshold else periphery_alloc
        )

        allocations.append(
            {
                "Stock": row["Stock"],
                "Coreness": row["Coreness"],
                "Allocation": allocation,
            }
        )

    # check that total allocation sums to 1.0 (or very close due to rounding)
    total_alloc = sum(a["Allocation"] for a in allocations)
    if not np.isclose(total_alloc, 1.0):
        raise ValueError(f"Total allocation sums to {total_alloc:.4f}, expected 1.0")

    # Sort by coreness ascending (rank 1 = lowest coreness = most peripheral)
    allocations_df = pd.DataFrame(allocations)
    allocations_df = allocations_df.sort_values("Coreness", ascending=True).reset_index(
        drop=True
    )
    allocations_df["Coreness_Rank"] = range(1, len(allocations_df) + 1)

    return allocations_df


def _get_side_exposures(strategy_config: StrategyConfig) -> Tuple[float, float]:
    """Return target gross exposures for the long and short books."""
    return strategy_config.gross_long_exposure, strategy_config.gross_short_exposure


def _classify_coreness_buckets(
    results_df: pd.DataFrame,
    strategy_config: StrategyConfig,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, float]:
    """Split stocks into long and short books based on coreness."""
    quantile_prop = strategy_config.effective_periphery_quantile
    coreness_threshold = results_df["Coreness"].quantile(quantile_prop)
    is_core = results_df["Coreness"] >= coreness_threshold

    if strategy_config.long_periphery:
        is_long = ~is_core
        is_short = is_core
    else:
        is_long = is_core
        is_short = ~is_core

    return results_df.copy(), is_long, is_short, coreness_threshold


def _validate_weight_caps(
    n_long: int,
    n_short: int,
    strategy_config: StrategyConfig,
) -> None:
    """Ensure the configured per-stock caps can satisfy target exposures."""
    long_total, short_total = _get_side_exposures(strategy_config)
    max_long_weight = strategy_config.resolved_max_long_weight
    max_short_weight = strategy_config.resolved_max_short_weight

    if n_long <= 0 or n_short <= 0:
        raise ValueError(
            f"Need at least one long and one short stock, got {n_long} longs and {n_short} shorts"
        )

    if max_long_weight * n_long + 1e-12 < long_total:
        raise ValueError(
            f"max_long_weight={max_long_weight:.4f} is too small for {n_long} long stocks; "
            f"need at least {long_total / n_long:.4f} per long to satisfy target exposure"
        )

    if max_short_weight * n_short + 1e-12 < short_total:
        raise ValueError(
            f"max_short_weight={max_short_weight:.4f} is too small for {n_short} short stocks; "
            f"need at least {short_total / n_short:.4f} per short to satisfy target exposure"
        )


def _allocate_equal_weights(
    results_df: pd.DataFrame,
    is_long: pd.Series,
    is_short: pd.Series,
    strategy_config: StrategyConfig,
) -> pd.DataFrame:
    """Allocate equal weights within the long and short books."""
    n_long = int(is_long.sum())
    n_short = int(is_short.sum())
    _validate_weight_caps(n_long, n_short, strategy_config)

    long_total, short_total = _get_side_exposures(strategy_config)
    long_weight = long_total / n_long
    short_weight = -short_total / n_short
    max_long_weight = strategy_config.resolved_max_long_weight
    max_short_weight = strategy_config.resolved_max_short_weight

    if long_weight > max_long_weight + 1e-12:
        raise ValueError(
            f"Equal long weight {long_weight:.4f} exceeds max_long_weight={max_long_weight:.4f}"
        )
    if abs(short_weight) > max_short_weight + 1e-12:
        raise ValueError(
            f"Equal short weight {abs(short_weight):.4f} exceeds max_short_weight={max_short_weight:.4f}"
        )

    allocations_df = results_df.copy()
    allocations_df["Allocation"] = np.where(is_long, long_weight, short_weight)
    return allocations_df


def _allocate_capped_proportional_weights(
    strengths: pd.Series,
    total_exposure: float,
    max_weight: float,
) -> pd.Series:
    """Allocate exposure proportionally to strengths while respecting a cap."""
    if len(strengths) == 0:
        return strengths.copy()

    if total_exposure < 0:
        raise ValueError("total_exposure must be non-negative")
    if max_weight <= 0:
        raise ValueError("max_weight must be positive")
    if max_weight * len(strengths) + 1e-12 < total_exposure:
        raise ValueError("max_weight is too small to satisfy total_exposure")

    positive_strengths = strengths.astype(float).clip(lower=0.0)
    if positive_strengths.sum() <= 0:
        positive_strengths = pd.Series(1.0, index=strengths.index, dtype=float)

    allocations = pd.Series(0.0, index=positive_strengths.index, dtype=float)
    remaining = positive_strengths.copy()
    remaining_exposure = float(total_exposure)

    while len(remaining) > 0 and remaining_exposure > 1e-12:
        scaled = remaining / remaining.sum() * remaining_exposure
        capped_mask = scaled >= max_weight - 1e-12

        if not capped_mask.any():
            allocations.loc[remaining.index] += scaled
            remaining_exposure = 0.0
            break

        capped_names = scaled.index[capped_mask]
        allocations.loc[capped_names] = max_weight
        remaining_exposure -= max_weight * len(capped_names)
        remaining = remaining.loc[~remaining.index.isin(capped_names)]

        if remaining_exposure < -1e-9:
            raise ValueError("Exposure allocation overshot while applying caps")

    if not np.isclose(allocations.sum(), total_exposure):
        raise ValueError(
            f"Allocated exposure sums to {allocations.sum():.6f}, expected {total_exposure:.6f}"
        )

    return allocations


def _allocate_coreness_proportional(
    results_df: pd.DataFrame,
    is_long: pd.Series,
    is_short: pd.Series,
    coreness_threshold: float,
    strategy_config: StrategyConfig,
) -> pd.DataFrame:
    """Allocate exposures continuously from coreness distance within each side."""
    long_total, short_total = _get_side_exposures(strategy_config)
    max_long_weight = strategy_config.resolved_max_long_weight
    max_short_weight = strategy_config.resolved_max_short_weight
    long_coreness = results_df.loc[is_long, ["Stock", "Coreness"]].set_index("Stock")[
        "Coreness"
    ]
    short_coreness = results_df.loc[is_short, ["Stock", "Coreness"]].set_index("Stock")[
        "Coreness"
    ]

    if strategy_config.long_periphery:
        long_strengths = coreness_threshold - long_coreness
        short_strengths = short_coreness - coreness_threshold
    else:
        long_strengths = long_coreness - coreness_threshold
        short_strengths = coreness_threshold - short_coreness

    long_weights = _allocate_capped_proportional_weights(
        long_strengths,
        total_exposure=long_total,
        max_weight=max_long_weight,
    )
    short_weights = _allocate_capped_proportional_weights(
        short_strengths,
        total_exposure=short_total,
        max_weight=max_short_weight,
    )

    allocations_df = results_df.copy()
    allocations_df["Allocation"] = 0.0
    allocations_df.loc[is_long, "Allocation"] = allocations_df.loc[is_long, "Stock"].map(
        long_weights
    )
    allocations_df.loc[is_short, "Allocation"] = -allocations_df.loc[
        is_short, "Stock"
    ].map(short_weights)
    return allocations_df


def _allocate_markowitz_min_vol(
    results_df: pd.DataFrame,
    is_long: pd.Series,
    is_short: pd.Series,
    log_returns: pd.DataFrame,
    strategy_config: StrategyConfig,
) -> pd.DataFrame:
    """Minimize portfolio variance subject to long/short exposure and cap constraints."""
    long_tickers = results_df.loc[is_long, "Stock"].tolist()
    short_tickers = results_df.loc[is_short, "Stock"].tolist()
    n_long = len(long_tickers)
    n_short = len(short_tickers)
    _validate_weight_caps(n_long, n_short, strategy_config)

    long_total, short_total = _get_side_exposures(strategy_config)
    max_long_weight = strategy_config.resolved_max_long_weight
    max_short_weight = strategy_config.resolved_max_short_weight
    ordered_tickers = long_tickers + short_tickers
    returns_subset = log_returns[ordered_tickers].dropna(how="any")
    if len(returns_subset) < 2:
        raise ValueError(
            "Not enough overlapping return observations for Markowitz optimization"
        )

    cov = returns_subset.cov().to_numpy(dtype=float)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov += np.eye(len(ordered_tickers)) * 1e-10

    def objective(x: np.ndarray) -> float:
        signed_weights = np.concatenate([x[:n_long], -x[n_long:]])
        return float(signed_weights @ cov @ signed_weights)

    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:n_long]) - long_total},
        {"type": "eq", "fun": lambda x: np.sum(x[n_long:]) - short_total},
    ]
    bounds = [(0.0, max_long_weight)] * n_long + [(0.0, max_short_weight)] * n_short
    x0 = np.array(
        [long_total / n_long] * n_long + [short_total / n_short] * n_short,
        dtype=float,
    )

    optimization = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not optimization.success:
        raise ValueError(f"Markowitz optimization failed: {optimization.message}")

    allocations_map = dict(zip(long_tickers, optimization.x[:n_long]))
    allocations_map.update(zip(short_tickers, -optimization.x[n_long:]))

    allocations_df = results_df.copy()
    allocations_df["Allocation"] = allocations_df["Stock"].map(allocations_map)
    return allocations_df


def allocate_by_coreness(
    results_df: pd.DataFrame,
    strategy_config: StrategyConfig,
    log_returns: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Map coreness scores to portfolio allocations.

    Strategy:
    - Split stocks into long and short books using coreness.
    - Apply the selected weighting method within those books.

    Ranking: Stocks are ranked by coreness (1 = most peripheral, N = most core).

    Args:
        results_df: DataFrame with columns ['Stock', 'Coreness']
        log_returns: Lookback-window return matrix for Markowitz optimization

    Returns:
        DataFrame with columns ['Stock', 'Coreness', 'Allocation', 'Coreness_Rank']
    """
    allocations_df, is_long, is_short, coreness_threshold = _classify_coreness_buckets(
        results_df, strategy_config
    )
    n_long = int(is_long.sum())
    n_short = int(is_short.sum())
    logger.info(
        f"  -> {n_long} long stocks, {n_short} short stocks using {strategy_config.weighting_method} "
        f"weighting at coreness threshold {coreness_threshold:.4f}"
    )

    if strategy_config.weighting_method == "equal":
        allocations_df = _allocate_equal_weights(
            allocations_df, is_long, is_short, strategy_config
        )
    elif strategy_config.weighting_method == "coreness_proportional":
        allocations_df = _allocate_coreness_proportional(
            allocations_df,
            is_long,
            is_short,
            coreness_threshold,
            strategy_config,
        )
    elif strategy_config.weighting_method == "markowitz_min_vol":
        if log_returns is None:
            raise ValueError("log_returns are required for Markowitz weighting")
        allocations_df = _allocate_markowitz_min_vol(
            allocations_df, is_long, is_short, log_returns, strategy_config
        )
    else:
        raise ValueError(
            f"Unknown weighting_method={strategy_config.weighting_method!r}; expected "
            f"'equal', 'coreness_proportional', or 'markowitz_min_vol'"
        )

    total_alloc = allocations_df["Allocation"].sum()
    if not np.isclose(total_alloc, strategy_config.target_net_exposure):
        raise ValueError(f"Total allocation sums to {total_alloc:.4f}, expected {strategy_config.target_net_exposure:.4f}")

    allocations_df = allocations_df.sort_values("Coreness", ascending=True).reset_index(
        drop=True
    )
    allocations_df["Coreness_Rank"] = range(1, len(allocations_df) + 1)

    return allocations_df


def get_rebalance_allocations(
    ticker_list: Union[List[str], Callable[[str], List[str]]],
    rebalance_dates: List[pd.Timestamp],
    lookback_days: int,
    strategy_config: StrategyConfig,
    parallel: bool = False,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Compute allocations for each rebalance date using Rossa algorithm.

    Args:
        ticker_list: List of stock tickers OR callable that takes date string and returns ticker list
        rebalance_dates: List of rebalance dates
        lookback_days: Number of days of history to use for Rossa
        strategy_config: Portfolio construction settings for the backtest
        parallel: If True, uses multiprocessing pool for parallel execution. Default False (sequential).

    Returns:
        Dictionary mapping rebalance_date -> allocation DataFrame
    """
    args_list = [
        (date, ticker_list, lookback_days, strategy_config) for date in rebalance_dates
    ]

    if parallel and len(rebalance_dates) > 1:
        # Use multiprocessing pool for parallel execution
        logger.info(f"Using parallel execution with multiprocessing pool...")
        with Pool(NUM_WORKERS) as pool:
            results = pool.map(_compute_allocation_for_rebalance_date, args_list)
    else:
        # Sequential execution - reuse worker function logic
        results = [_compute_allocation_for_rebalance_date(args) for args in args_list]

    rebalance_allocations = dict(results)
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
    Optimized via matrix vectorization to calculate periods between rebalances
    simultaneously, bypassing slow day-by-day iteration.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    # Fix lookahead bias: Only use ffill(), never bfill()
    prices_filtered = (
        prices[(prices.index >= start) & (prices.index <= end)].ffill().copy()
    )

    # CRITICAL FIX: Strip out any duplicate columns before matrix operations
    prices_filtered = prices_filtered.loc[:, ~prices_filtered.columns.duplicated()]

    # Extract sorted rebalance keys
    reb_keys = pd.Series(sorted(list(rebalance_schedule.keys())))
    reb_keys = reb_keys[reb_keys <= end]

    if reb_keys.empty:
        return pd.DataFrame(
            columns=[
                "Date",
                "Ticker",
                "Price",
                "Position_Value",
                "Portfolio_Total_Value",
            ]
        )

    # Pre-process allocations into a dictionary of Series for O(1) lookups
    allocations = {
        k: v.set_index("Stock")["Allocation"] for k, v in rebalance_schedule.items()
    }

    # Map each date in prices_filtered to its currently active rebalance date
    idx = np.searchsorted(reb_keys, prices_filtered.index, side="right") - 1
    valid_mask = idx >= 0

    # Drop dates before the first active rebalance date
    if not valid_mask.any():
        return pd.DataFrame(
            columns=[
                "Date",
                "Ticker",
                "Price",
                "Position_Value",
                "Portfolio_Total_Value",
            ]
        )

    prices_filtered = prices_filtered[valid_mask]
    active_reb_keys = reb_keys.iloc[idx[valid_mask]].values

    # Find the indices where the active rebalance key changes (chunk boundaries)
    chunk_start_indices = np.where(active_reb_keys[:-1] != active_reb_keys[1:])[0] + 1
    chunk_start_indices = np.insert(chunk_start_indices, 0, 0)
    chunk_end_indices = np.append(chunk_start_indices[1:], len(prices_filtered))

    portfolio_value = initial_capital
    all_chunks = []

    # Loop through REBALANCE PERIODS instead of individual days
    for start_idx, end_idx in zip(chunk_start_indices, chunk_end_indices):
        reb_key = active_reb_keys[start_idx]

        # Get the slice of price data for this entire rebalance period
        chunk_prices = prices_filtered.iloc[start_idx:end_idx]
        rebalance_prices = chunk_prices.iloc[0]

        alloc_series = allocations[reb_key]
        alloc_series = alloc_series[alloc_series.index.isin(rebalance_prices.index)]

        target_dollars = alloc_series * portfolio_value
        prices_at_reb = rebalance_prices[alloc_series.index]

        # Filter valid prices (not NaN, not 0)
        valid_price_mask = prices_at_reb.notna() & (prices_at_reb != 0)
        valid_tickers = prices_at_reb.index[valid_price_mask]

        if len(valid_tickers) == 0:
            continue

        # Calculate shares for the entire period
        shares = target_dollars[valid_tickers] / prices_at_reb[valid_tickers]
        shares = shares[shares != 0]
        # Remove duplicate index labels (keep first occurrence)
        shares = shares[~shares.index.duplicated(keep="first")]
        valid_tickers = shares.index

        if len(valid_tickers) == 0:
            continue

        invested_capital = (shares * prices_at_reb[valid_tickers]).sum()
        cash = portfolio_value - invested_capital

        # FAST VECTORIZED MATH: Calculate all daily values for this chunk at once
        chunk_positions = chunk_prices[valid_tickers].mul(shares, axis=1)
        chunk_port_value = chunk_positions.sum(axis=1) + cash

        # Flatten (melt) the dense chunk into the required long format output
        melted_pos = chunk_positions.reset_index().melt(
            id_vars="index", var_name="Ticker", value_name="Position_Value"
        )
        melted_prices = (
            chunk_prices[valid_tickers]
            .reset_index()
            .melt(id_vars="index", var_name="Ticker", value_name="Price")
        )

        melted_pos.rename(columns={"index": "Date"}, inplace=True)
        melted_pos["Price"] = melted_prices["Price"]
        melted_pos["Portfolio_Total_Value"] = melted_pos["Date"].map(chunk_port_value)

        # Drop rows where price is NaN to match your original fail-safe logic
        melted_pos = melted_pos.dropna(subset=["Price"])

        all_chunks.append(melted_pos)

        # The ending portfolio value of this chunk becomes the starting capital of the next
        portfolio_value = chunk_port_value.iloc[-1]

    if not all_chunks:
        return pd.DataFrame(
            columns=[
                "Date",
                "Ticker",
                "Price",
                "Position_Value",
                "Portfolio_Total_Value",
            ]
        )

    # Combine all chunks and sort chronologically
    final_df = pd.concat(all_chunks, ignore_index=True)
    final_df = final_df[
        ["Date", "Ticker", "Price", "Position_Value", "Portfolio_Total_Value"]
    ]
    final_df = final_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    return final_df


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


def export_benchmark_data_to_excel(
    benchmark_data: Dict[str, pd.DataFrame],
    benchmark_metrics: Dict[str, Dict],
    output_filename: str = "backtest_results.xlsx",
    export_config: Optional[ExcelExportConfig] = None,
) -> None:
    """
    Export benchmark data and metrics to Excel sheets.

    Creates sheets for each benchmark:
    - "Benchmark_[TICKER]": Daily performance data
    - "Benchmark_Metrics": Comparison of all benchmark metrics

    Args:
        benchmark_data: Dictionary mapping benchmark_ticker -> daily_metrics_df
        benchmark_metrics: Dictionary mapping benchmark_ticker -> summary_stats_dict
        output_filename: Output Excel filename (will append to existing)
        export_config: Lossless Excel writer settings
    """
    export_config = export_config or ExcelExportConfig()
    # Append to existing Excel file
    with pd.ExcelWriter(
        output_filename, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        _write_benchmark_sheets(writer, benchmark_data, benchmark_metrics, export_config)

    logger.info(f"Benchmark data exported to Excel: {output_filename}")


def _write_benchmark_sheets(
    writer: pd.ExcelWriter,
    benchmark_data: Optional[Dict[str, pd.DataFrame]],
    benchmark_metrics: Optional[Dict[str, Dict]],
    export_config: ExcelExportConfig,
) -> None:
    """Write benchmark sheets into an open Excel workbook."""
    if benchmark_data:
        for ticker, daily_metrics in benchmark_data.items():
            _write_dataframe_sheet(
                writer, daily_metrics, sheet_name=f"Benchmark_{ticker}", index=False
            )

    if benchmark_metrics:
        metrics_rows = []
        for ticker, metrics in benchmark_metrics.items():
            row = {"Benchmark": ticker}
            row.update(metrics)
            metrics_rows.append(row)

        _write_dataframe_sheet(
            writer,
            pd.DataFrame(metrics_rows),
            sheet_name="Benchmark_Metrics",
            index=False,
        )


def _is_xlsxwriter_writer(writer: pd.ExcelWriter) -> bool:
    """Return whether the active Excel writer uses xlsxwriter."""
    return writer.engine == "xlsxwriter"


def _write_dataframe_sheet(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    sheet_name: str,
    index: bool = False,
) -> None:
    """Write a dataframe, using direct worksheet writes for xlsxwriter."""
    if not _is_xlsxwriter_writer(writer):
        df.to_excel(writer, sheet_name=sheet_name, index=index)
        return

    worksheet = writer.book.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet
    date_format = writer.book.add_format({"num_format": "yyyy-mm-dd"})
    datetime_format = writer.book.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"})

    export_df = df.reset_index() if index else df
    headers = export_df.columns.tolist()
    worksheet.write_row(0, 0, headers)

    for row_idx, row in enumerate(export_df.itertuples(index=False, name=None), start=1):
        for col_idx, value in enumerate(row):
            if pd.isna(value):
                continue
            if isinstance(value, pd.Timestamp):
                if value.time() == pd.Timestamp(value.date()).time():
                    worksheet.write_datetime(row_idx, col_idx, value.to_pydatetime(), date_format)
                else:
                    worksheet.write_datetime(
                        row_idx, col_idx, value.to_pydatetime(), datetime_format
                    )
            else:
                worksheet.write(row_idx, col_idx, value)


def _write_wide_timeseries_sheet(
    writer: pd.ExcelWriter,
    wide_df: pd.DataFrame,
    sheet_name: str,
) -> None:
    """Write a date-indexed wide dataframe efficiently and losslessly."""
    if not _is_xlsxwriter_writer(writer):
        wide_df.to_excel(writer, sheet_name=sheet_name)
        return

    worksheet = writer.book.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet
    date_format = writer.book.add_format({"num_format": "yyyy-mm-dd"})

    headers = ["Date"] + [str(col) for col in wide_df.columns]
    worksheet.write_row(0, 0, headers)

    values = wide_df.to_numpy()
    dates = pd.to_datetime(wide_df.index)
    for row_idx, (date_value, row_values) in enumerate(zip(dates, values), start=1):
        worksheet.write_datetime(row_idx, 0, date_value.to_pydatetime(), date_format)
        for col_idx, value in enumerate(row_values, start=1):
            if pd.isna(value):
                continue
            worksheet.write(row_idx, col_idx, value)


def _write_backtest_sheets(
    writer: pd.ExcelWriter,
    daily_holdings: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    rebalance_schedule: Dict[pd.Timestamp, pd.DataFrame],
    summary_stats: Dict,
    benchmark_data: Optional[Dict[str, pd.DataFrame]],
    benchmark_metrics: Optional[Dict[str, Dict]],
    export_config: ExcelExportConfig,
) -> None:
    """Write all backtest result sheets into an open Excel workbook."""
    # Sheet 1: Daily Holdings (pivoted by ticker)
    daily_export = daily_holdings.copy()
    daily_export["Date"] = pd.to_datetime(daily_export["Date"])

    # Group once per Date/Ticker to preserve previous "first value wins" behavior.
    position_pivot = (
        daily_export.groupby(["Date", "Ticker"], sort=False)["Position_Value"]
        .first()
        .unstack()
    )
    _write_wide_timeseries_sheet(writer, position_pivot, sheet_name="Position Values ($)")

    # Pivot prices by ticker (optional, for reference)
    price_pivot = (
        daily_export.groupby(["Date", "Ticker"], sort=False)["Price"].first().unstack()
    )
    _write_wide_timeseries_sheet(writer, price_pivot, sheet_name="Prices ($)")

    # Add portfolio total value (same for all tickers on each date)
    portfolio_total = (
        daily_export[["Date", "Portfolio_Total_Value"]]
        .drop_duplicates(subset=["Date"])
        .set_index("Date")
    )
    _write_dataframe_sheet(
        writer, portfolio_total.reset_index(), sheet_name="Portfolio Total", index=False
    )

    # Sheet 2: Portfolio Metrics (separate sheet for each metric)
    portfolio_export = portfolio_summary.copy()
    portfolio_export["Date"] = pd.to_datetime(portfolio_export["Date"])
    portfolio_export["Daily_Return"] = portfolio_export["Daily_Return"].fillna(0)
    portfolio_export["Cumulative_Return"] = portfolio_export[
        "Cumulative_Return"
    ].fillna(0)

    # Create separate sheets for each metric
    metrics_to_export = ["Daily_Return", "Cumulative_Return"]
    for metric in metrics_to_export:
        if metric in portfolio_export.columns:
            metric_df = portfolio_export[["Date", metric]].copy()
            sheet_name = metric.replace("_", " ")
            _write_dataframe_sheet(writer, metric_df, sheet_name=sheet_name, index=False)

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
    _write_dataframe_sheet(
        writer, metrics_df, sheet_name="Performance Metrics", index=False
    )

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
    _write_dataframe_sheet(
        writer, rebalance_df, sheet_name="Rebalance Events", index=False
    )

    _write_benchmark_sheets(writer, benchmark_data, benchmark_metrics, export_config)


def export_to_excel(
    daily_holdings: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    rebalance_schedule: Dict[pd.Timestamp, pd.DataFrame],
    summary_stats: Dict,
    benchmark_data: Optional[Dict[str, pd.DataFrame]] = None,
    benchmark_metrics: Optional[Dict[str, Dict]] = None,
    output_filename: str = "backtest_results.xlsx",
    export_config: Optional[ExcelExportConfig] = None,
) -> None:
    """
    Export backtest results to Excel workbook with separate sheets for each metric.

    Creates multiple sheets:
    - "Position Values ($)": Daily position values by ticker (pivoted)
    - "Prices ($)": Daily prices by ticker (pivoted)
    - "Portfolio Total": Daily portfolio total value
    - "Daily Return": Daily portfolio returns over time
    - "Cumulative Return": Cumulative portfolio returns over time
    - "Performance Metrics": Key risk/return summary statistics
    - "Rebalance Events": Rebalance allocation decisions

    Args:
        daily_holdings: DataFrame from calculate_portfolio_daily_values
        portfolio_summary: DataFrame from calculate_portfolio_metrics
        rebalance_schedule: Dictionary of rebalance allocations
        summary_stats: Dict with performance metrics from calculate_portfolio_metrics
        benchmark_data: Optional benchmark daily data to write in the same workbook
        benchmark_metrics: Optional benchmark summary stats to write in the same workbook
        output_filename: Output Excel filename
        export_config: Lossless Excel writer settings
    """
    export_config = export_config or ExcelExportConfig()

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

    try:
        with pd.ExcelWriter(
            output_filename,
            engine="xlsxwriter",
            datetime_format="yyyy-mm-dd",
            engine_kwargs={
                "options": {"constant_memory": export_config.use_constant_memory}
            },
        ) as writer:
            _write_backtest_sheets(
                writer,
                daily_holdings,
                portfolio_summary,
                rebalance_schedule,
                summary_stats,
                benchmark_data,
                benchmark_metrics,
                export_config,
            )
    except ModuleNotFoundError:
        logger.warning(
            "xlsxwriter is not installed; falling back to openpyxl for Excel export"
        )
        with pd.ExcelWriter(output_filename, engine="openpyxl") as writer:
            _write_backtest_sheets(
                writer,
                daily_holdings,
                portfolio_summary,
                rebalance_schedule,
                summary_stats,
                benchmark_data,
                benchmark_metrics,
                export_config,
            )

    logger.info(f"Excel report exported: {output_filename}")


def plot_backtest_results(
    portfolio_summary: pd.DataFrame,
    benchmark_data: Optional[Dict[str, pd.DataFrame]] = None,
    output_dir: str = ".",
) -> None:
    """
    Create visualization plots for backtest results, optionally with benchmarks.

    Generates:
    - Portfolio value over time (with benchmark comparison)
    - Cumulative returns (with benchmark comparison)
    - Drawdown from peak (inverted so big drawdowns go DOWN on graph)

    Args:
        portfolio_summary: DataFrame from calculate_portfolio_metrics
        benchmark_data: Optional dict mapping benchmark_ticker -> daily_metrics_df
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Portfolio Value
    axes[0].plot(
        portfolio_summary["Date"],
        portfolio_summary["Portfolio_Value"],
        linewidth=2.5,
        color="blue",
        label="Strategy",
        zorder=10,
    )

    # Add benchmark lines
    if benchmark_data:
        colors = plt.cm.Set2(np.linspace(0, 1, len(benchmark_data)))
        for (ticker, bench_df), color in zip(benchmark_data.items(), colors):
            axes[0].plot(
                bench_df["Date"],
                bench_df["Portfolio_Value"],
                linewidth=1.5,
                color=color,
                label=ticker,
                alpha=0.7,
            )

    axes[0].set_title("Portfolio Value Over Time", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9)

    # Plot 2: Cumulative Returns
    axes[1].plot(
        portfolio_summary["Date"],
        portfolio_summary["Cumulative_Return"] * 100,
        linewidth=2.5,
        color="green",
        label="Strategy",
        zorder=10,
    )

    # Add benchmark lines
    if benchmark_data:
        colors = plt.cm.Set2(np.linspace(0, 1, len(benchmark_data)))
        for (ticker, bench_df), color in zip(benchmark_data.items(), colors):
            axes[1].plot(
                bench_df["Date"],
                bench_df["Cumulative_Return"] * 100,
                linewidth=1.5,
                color=color,
                label=ticker,
                alpha=0.7,
            )

    axes[1].set_title("Cumulative Returns (%)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Cumulative Return (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper left", fontsize=9)

    # Plot 3: Drawdown from Peak (INVERTED - big drawdowns go DOWN)
    running_max = portfolio_summary["Portfolio_Value"].expanding().max()
    # Drawdown = (Current - Peak) / Peak (negative when price drops from peak)
    drawdown = (portfolio_summary["Portfolio_Value"] - running_max) / running_max * 100
    axes[2].fill_between(portfolio_summary["Date"], drawdown, 0, alpha=0.3, color="red")
    axes[2].plot(
        portfolio_summary["Date"],
        drawdown,
        linewidth=2.5,
        color="darkred",
        label="Strategy",
        zorder=10,
    )

    # Add benchmark drawdown lines
    if benchmark_data:
        colors = plt.cm.Set2(np.linspace(0, 1, len(benchmark_data)))
        for (ticker, bench_df), color in zip(benchmark_data.items(), colors):
            bench_running_max = bench_df["Portfolio_Value"].expanding().max()
            bench_drawdown = (
                (bench_df["Portfolio_Value"] - bench_running_max)
                / bench_running_max
                * 100
            )
            axes[2].plot(
                bench_df["Date"],
                bench_drawdown,
                linewidth=1.5,
                color=color,
                label=ticker,
                alpha=0.7,
            )

    axes[2].set_title(
        "Drawdown from Peak (% - inverted, so down = bad)",
        fontsize=14,
        fontweight="bold",
    )
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Drawdown (%)")
    all_drawdowns = (
        [drawdown]
        if benchmark_data is None
        else [drawdown]
        + [
            (
                bench_df["Portfolio_Value"].expanding().max()
                - bench_df["Portfolio_Value"]
            )
            / bench_df["Portfolio_Value"].expanding().max()
            * 100
            for bench_df in benchmark_data.values()
        ]
    )
    min_dd = min([dd.min() for dd in all_drawdowns])
    axes[2].set_ylim([min(min_dd, -1), 1])  # Ensure 0 line is visible
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color="black", linestyle="-", alpha=0.3, linewidth=0.5)
    axes[2].legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "backtest_plots.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Plots saved: {output_file}")
    plt.close()


def plot_backtest_results_log(
    portfolio_summary: pd.DataFrame,
    benchmark_data: Optional[Dict[str, pd.DataFrame]] = None,
    output_dir: str = ".",
) -> None:
    """
    Create log-scale performance plots for easier long-horizon comparison.

    Generates:
    - Portfolio value over time on a log y-axis
    - Growth of $1 on a log y-axis
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    strategy_growth = portfolio_summary["Portfolio_Value"] / portfolio_summary[
        "Portfolio_Value"
    ].iloc[0]

    axes[0].plot(
        portfolio_summary["Date"],
        portfolio_summary["Portfolio_Value"],
        linewidth=2.5,
        color="blue",
        label="Strategy",
        zorder=10,
    )
    axes[1].plot(
        portfolio_summary["Date"],
        strategy_growth,
        linewidth=2.5,
        color="green",
        label="Strategy",
        zorder=10,
    )

    if benchmark_data:
        colors = plt.cm.Set2(np.linspace(0, 1, len(benchmark_data)))
        for (ticker, bench_df), color in zip(benchmark_data.items(), colors):
            benchmark_growth = bench_df["Portfolio_Value"] / bench_df["Portfolio_Value"].iloc[0]
            axes[0].plot(
                bench_df["Date"],
                bench_df["Portfolio_Value"],
                linewidth=1.5,
                color=color,
                label=ticker,
                alpha=0.7,
            )
            axes[1].plot(
                bench_df["Date"],
                benchmark_growth,
                linewidth=1.5,
                color=color,
                label=ticker,
                alpha=0.7,
            )

    axes[0].set_yscale("log")
    axes[0].set_title("Portfolio Value Over Time (Log Scale)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Portfolio Value ($, log)")
    axes[0].grid(True, alpha=0.3, which="both")
    axes[0].legend(loc="upper left", fontsize=9)

    axes[1].set_yscale("log")
    axes[1].set_title("Growth of $1 (Log Scale)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Growth Multiple (log)")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    output_file = os.path.join(output_dir, "backtest_plots_log.png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    logger.info(f"Log-scale plots saved: {output_file}")
    plt.close()


def export_summary_txt(
    summary_stats: Dict,
    portfolio_summary: pd.DataFrame,
    benchmark_metrics: Optional[Dict[str, Dict]] = None,
    output_filename: str = "summary.txt",
) -> None:
    """
    Export backtest summary statistics to a simple text file, optionally with benchmarks.

    Args:
        summary_stats: Dict with performance metrics from calculate_portfolio_metrics
        portfolio_summary: DataFrame from calculate_portfolio_metrics
        benchmark_metrics: Optional dict mapping benchmark_ticker -> summary_stats_dict
        output_filename: Output text filename
    """
    with open(output_filename, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("BACKTEST SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("STRATEGY PERFORMANCE METRICS\n")
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

        # Add benchmark comparison if available
        if benchmark_metrics:
            f.write("=" * 70 + "\n")
            f.write("BENCHMARK COMPARISON\n")
            f.write("=" * 70 + "\n\n")

            for ticker, metrics in benchmark_metrics.items():
                f.write(f"{ticker}\n")
                f.write("-" * 70 + "\n")
                f.write(
                    f"Total Return:                    {metrics['Total_Return'] * 100:>12.2f}%\n"
                )
                f.write(
                    f"Annualized Return (geometric):   {metrics['Annualized_Return'] * 100:>12.2f}%\n"
                )
                f.write(
                    f"Arithmetic Average Return:       {metrics['Arithmetic_Annual_Return'] * 100:>12.2f}%\n"
                )
                f.write(
                    f"Volatility (annualized):         {metrics['Volatility'] * 100:>12.2f}%\n"
                )
                f.write(
                    f"Sharpe Ratio:                    {metrics['Sharpe_Ratio']:>12.4f}\n"
                )
                f.write(
                    f"Sortino Ratio:                   {metrics['Sortino_Ratio']:>12.4f}\n"
                )
                f.write(
                    f"Max Drawdown:                    {metrics['Max_Drawdown'] * 100:>12.2f}%\n"
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


def run_backtest(config: BacktestConfig) -> BacktestResult:
    """
    Run complete backtest of core-periphery strategy with optional benchmark comparison.

    Args:
        config: BacktestConfig instance specifying all backtest parameters

    Returns:
        BacktestResult instance with portfolio_summary, summary_stats, and optional benchmark data
    """
    logger.info("=" * 70)
    logger.info("BACKTEST: CORE-PERIPHERY STRATEGY")
    logger.info("=" * 70)
    logger.info(
        f"Backtest period: {config.start_backtest_date} to {config.end_backtest_date}"
    )
    logger.info(f"Lookback period: {config.lookback_days} days")
    logger.info(f"Rebalance interval: {config.rebalance_interval_days} days")

    # Determine if using dynamic or static ticker list
    is_dynamic = callable(config.ticker_list)
    if is_dynamic:
        logger.info(
            "Using dynamic S&P 500 constituent selection (avoiding survivorship bias)"
        )
    else:
        logger.info(f"Using static ticker list: {len(config.ticker_list)} tickers")

    # Step 1: Generate rebalance dates first (needed for dynamic ticker fetching)
    logger.info("[1/6] Generating rebalance schedule...")
    rebalance_dates = generate_rebalance_dates(
        config.start_backtest_date,
        config.end_backtest_date,
        config.rebalance_interval_days,
    )
    logger.info(f"  → {len(rebalance_dates)} rebalance dates generated")

    # Step 2: Fetch price data
    # For dynamic mode, fetch all unique tickers across all rebalance dates
    if is_dynamic:
        all_tickers_set = set()
        for rebalance_date in rebalance_dates:
            tickers = config.ticker_list(rebalance_date.strftime("%Y-%m-%d"))
            all_tickers_set.update(tickers)
        tickers_to_fetch = list(all_tickers_set)
        logger.info(
            f"Dynamic mode will use {len(all_tickers_set)} unique tickers across rebalance dates"
        )
    else:
        tickers_to_fetch = list(config.ticker_list)

    if config.benchmark_tickers:
        tickers_to_fetch.extend(config.benchmark_tickers)

    # CRITICAL FIX: Remove duplicates to prevent duplicate columns in price_data
    tickers_to_fetch = list(set(tickers_to_fetch))

    price_data = rossa.fetch_and_cache_stock_data(tuple(tickers_to_fetch))

    # CRITICAL FIX: Clean price data (remove penny stocks & data glitches)
    price_data = clean_price_data(price_data, min_price=1.0, max_price=10000.0)

    # Verify data covers the required backtest period
    actual_start = price_data.index[0]
    actual_end = price_data.index[-1]
    logger.info(f"Data covers period: {actual_start.date()} to {actual_end.date()}")
    logger.info(
        f"Backtest will use: {config.start_backtest_date} to {config.end_backtest_date}"
    )

    # Step 3: Compute allocations for each rebalance
    logger.info("[3/6] Computing allocations for each rebalance...")
    rebalance_schedule = get_rebalance_allocations(
        config.ticker_list,
        rebalance_dates,
        config.lookback_days,
        config.strategy,
        parallel=config.parallel,
    )

    # Step 4: Calculate daily portfolio values
    logger.info("[4/6] Calculating daily portfolio values...")
    daily_holdings = calculate_portfolio_daily_values(
        price_data,
        rebalance_schedule,
        config.start_backtest_date,
        config.end_backtest_date,
    )

    # Step 5: Calculate metrics
    logger.info("[5/6] Calculating portfolio metrics...")
    portfolio_summary, summary_stats = calculate_portfolio_metrics(daily_holdings)

    # Fetch benchmark data if requested
    benchmark_data = None
    benchmark_metrics = None

    if config.benchmark_tickers:
        logger.info("[5.5/6] Fetching benchmark data...")
        benchmark_data = calculate_benchmark_daily_values(
            price_data,
            config.benchmark_tickers,
            config.start_backtest_date,
            config.end_backtest_date,
        )
        benchmark_metrics = calculate_benchmark_metrics(benchmark_data)

    # Step 6: Export reports
    logger.info("[5.6/6] Generating reports...")

    # Export Excel
    if config.output_excel is not None:
        os.makedirs(str(DataConfig.OUTPUT_DIR), exist_ok=True)
        output_excel_path = os.path.join(
            str(DataConfig.OUTPUT_DIR), config.output_excel
        )
        export_to_excel(
            daily_holdings,
            portfolio_summary,
            rebalance_schedule,
            summary_stats,
            benchmark_data,
            benchmark_metrics,
            output_excel_path,
            config.excel_export,
        )

    # Export summary text file
    if config.summary_file is not None:
        os.makedirs(str(DataConfig.OUTPUT_DIR), exist_ok=True)
        summary_file_path = os.path.join(
            str(DataConfig.OUTPUT_DIR), config.summary_file
        )
        export_summary_txt(
            summary_stats, portfolio_summary, benchmark_metrics, summary_file_path
        )

    # Generate plots (with benchmarks if available)
    if config.output_plots:
        os.makedirs(str(DataConfig.OUTPUT_DIR), exist_ok=True)
        plot_backtest_results(
            portfolio_summary, benchmark_data, output_dir=str(DataConfig.OUTPUT_DIR)
        )
        plot_backtest_results_log(
            portfolio_summary, benchmark_data, output_dir=str(DataConfig.OUTPUT_DIR)
        )

        # Compute and plot benchmark summary stats over time
        if benchmark_data and rebalance_dates:
            benchmark_stats_over_time = compute_benchmark_summary_stats_over_time(
                benchmark_data, rebalance_dates, lookback_days=config.lookback_days
            )
            plot_benchmark_summary_over_time(
                benchmark_stats_over_time, output_dir=str(DataConfig.OUTPUT_DIR)
            )

    # Print summary statistics
    print_backtest_summary(summary_stats, portfolio_summary, benchmark_metrics)

    # Step 6: Factor analysis (if requested)
    factor_loadings_df = None
    rsquared_series = None

    if config.factor_lookback_days is not None and config.factor_lookback_days > 0:
        logger.info("[6/6] Computing factor analysis...")
        factor_loadings_df, rsquared_series = compute_factor_loadings_over_time(
            portfolio_summary=portfolio_summary,
            rebalance_dates=rebalance_dates,
            factor_lookback_days=config.factor_lookback_days,
            factor_list=config.factor_list,
            factor_data_file=config.factor_data_file,
        )

        if factor_loadings_df is not None:
            os.makedirs(str(DataConfig.OUTPUT_DIR), exist_ok=True)
            plot_factor_loadings_multiline(
                factor_loadings_df,
                os.path.join(
                    str(DataConfig.OUTPUT_DIR), "factor_loadings_multiline.png"
                ),
            )
            plot_factor_loadings_subplots(
                factor_loadings_df,
                os.path.join(
                    str(DataConfig.OUTPUT_DIR), "factor_loadings_subplots.png"
                ),
            )
            plot_factor_rsquared(
                rsquared_series,
                os.path.join(str(DataConfig.OUTPUT_DIR), "factor_rsquared.png"),
            )
            export_factor_analysis_to_excel(
                factor_loadings_df,
                rsquared_series,
                os.path.join(str(DataConfig.OUTPUT_DIR), "factor_analysis.xlsx"),
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

    return BacktestResult(
        portfolio_summary=portfolio_summary,
        summary_stats=summary_stats,
        benchmark_data=benchmark_data,
        benchmark_metrics=benchmark_metrics,
    )


def _backtest_worker(config: BacktestConfig) -> Tuple[str, BacktestResult]:
    """
    Worker function for parallel backtest execution.
    Returns: (start_date_str, BacktestResult)
    """
    # Disable logging in worker threads
    logging.disable(logging.CRITICAL)

    result = run_backtest(config)
    return (config.start_backtest_date, result)


def load_tickers_from_file(ticker_file: Union[str, os.PathLike]) -> List[str]:
    """Load a plain-text ticker list from disk."""
    ticker_path = str(ticker_file)
    if not os.path.exists(ticker_path):
        raise FileNotFoundError(f"{ticker_path} not found")

    with open(ticker_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def generate_step_forward_date_pairs(
    overall_start_date: str,
    overall_end_date: str,
    eval_lookback: pd.DateOffset,
    eval_interval: pd.DateOffset,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate rolling step-forward evaluation windows."""
    start_bound = pd.to_datetime(overall_start_date)
    end_bound = pd.to_datetime(overall_end_date)

    last_end_date = end_bound
    last_start_date = last_end_date - eval_lookback + pd.DateOffset(days=1)
    date_pairs = [(last_start_date, last_end_date)]

    while date_pairs[-1][0] > start_bound:
        prev_start = date_pairs[-1][0] - eval_interval
        prev_end = date_pairs[-1][1] - eval_interval
        date_pairs.append((prev_start, prev_end))

    filtered_pairs = []
    for start_date, end_date in reversed(date_pairs):
        if start_date < start_bound:
            start_date = start_bound
        if start_date <= end_date:
            filtered_pairs.append((start_date, end_date))

    return filtered_pairs


def build_step_forward_configs(plan: StepForwardBacktestRun) -> List[BacktestConfig]:
    """Create child backtest configs for a step-forward evaluation plan."""
    date_pairs = generate_step_forward_date_pairs(
        plan.overall_start_date,
        plan.overall_end_date,
        plan.eval_lookback,
        plan.eval_interval,
    )

    configs = []
    for start_date, end_date in date_pairs:
        configs.append(
            replace(
                plan.base_config,
                start_backtest_date=start_date.strftime("%Y-%m-%d"),
                end_backtest_date=end_date.strftime("%Y-%m-%d"),
            )
        )
    return configs


def run_step_forward_evaluation(
    plan: StepForwardBacktestRun,
) -> StepForwardBacktestResult:
    """Run a rolling step-forward evaluation."""
    backtest_configs = build_step_forward_configs(plan)
    logger.info(
        f"Running {len(backtest_configs)} step-forward backtests for {plan.name}"
    )

    if plan.parallel and len(backtest_configs) > 1:
        manager = multiprocessing.Manager()
        shared_lock = manager.Lock()
        with Pool(
            NUM_WORKERS, initializer=init_worker_lock, initargs=(shared_lock,)
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_backtest_worker, backtest_configs),
                    total=len(backtest_configs),
                    desc=f"Running {plan.name}",
                )
            )
    else:
        results = [_backtest_worker(config) for config in backtest_configs]

    results_dict = {start_date: result for start_date, result in results}
    summary_stats_over_time = []

    for config in backtest_configs:
        date_key = config.start_backtest_date
        if date_key not in results_dict:
            continue

        result = results_dict[date_key]
        summary_row = dict(result.summary_stats)
        summary_row["Date"] = pd.Timestamp(config.end_backtest_date)
        summary_row["Window_Start"] = pd.Timestamp(config.start_backtest_date)
        summary_row["Window_End"] = pd.Timestamp(config.end_backtest_date)
        summary_stats_over_time.append(summary_row)

    if plan.summary_plot_filename and summary_stats_over_time:
        os.makedirs(str(DataConfig.OUTPUT_DIR), exist_ok=True)
        plot_backtest_summary_over_time(
            summary_stats_over_time,
            os.path.join(str(DataConfig.OUTPUT_DIR), plan.summary_plot_filename),
        )

    return StepForwardBacktestResult(
        results_by_start_date=results_dict,
        summary_stats_over_time=summary_stats_over_time,
    )


def print_backtest_summary(
    summary_stats: Dict,
    portfolio_summary: pd.DataFrame,
    benchmark_metrics: Optional[Dict[str, Dict]] = None,
) -> None:
    logger.info("-" * 70)
    logger.info("BACKTEST SUMMARY - STRATEGY")
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

    # Print benchmark comparison if available
    if benchmark_metrics:
        logger.info("-" * 70)
        logger.info("BENCHMARK COMPARISON")
        logger.info("-" * 70)
        for ticker, metrics in benchmark_metrics.items():
            logger.info(f"\n{ticker}:")
            logger.info(f"  Total Return: {metrics['Total_Return'] * 100:.2f}%")
            logger.info(
                f"  Annualized Return: {metrics['Annualized_Return'] * 100:.2f}%"
            )
            logger.info(f"  Volatility: {metrics['Volatility'] * 100:.2f}%")
            logger.info(f"  Sharpe Ratio: {metrics['Sharpe_Ratio']:.4f}")
            logger.info(f"  Max Drawdown: {metrics['Max_Drawdown'] * 100:.2f}%")
        logger.info("=" * 70)


def compute_benchmark_summary_stats_over_time(
    benchmark_data: Dict[str, pd.DataFrame],
    rebalance_dates: List[pd.Timestamp],
    lookback_days: int = 252,
) -> Dict[str, List[Dict]]:
    """
    Compute summary statistics over time for each benchmark at each rebalance date.

    For each benchmark and rebalance date, computes annualized return, volatility,
    and Sharpe ratio using a rolling lookback window.

    Args:
        benchmark_data: Dict mapping benchmark_ticker -> daily_metrics_df
        rebalance_dates: List of rebalance dates
        lookback_days: Lookback window in days (default 252 = 1 year)

    Returns:
        Dict mapping benchmark_ticker -> List[Dict with Date, Annualized_Return, Volatility, Sharpe_Ratio]
    """
    RF_RATE = 0.03  # 3% annual risk-free rate

    benchmark_stats_over_time = {}

    for ticker, bench_df in benchmark_data.items():
        stats_list = []

        for rebalance_date in rebalance_dates:
            # Get data within lookback window ending at rebalance_date
            window_start = rebalance_date - pd.Timedelta(days=lookback_days)
            window_data = bench_df[
                (bench_df["Date"] >= window_start)
                & (bench_df["Date"] <= rebalance_date)
            ].copy()

            if len(window_data) < 2:
                continue

            # Compute daily returns
            daily_values = window_data["Portfolio_Value"].values
            daily_returns = np.diff(daily_values) / daily_values[:-1]

            # Annualized return (geometric)
            total_return = (daily_values[-1] - daily_values[0]) / daily_values[0]
            days_elapsed = (
                window_data["Date"].iloc[-1] - window_data["Date"].iloc[0]
            ).days
            years_elapsed = max(days_elapsed / 365.0, 1.0 / 252.0)  # At least 1 day
            annual_return = ((1 + total_return) ** (1 / years_elapsed)) - 1

            # Annualized volatility
            daily_volatility = np.std(daily_returns)
            annualized_volatility = daily_volatility * np.sqrt(252)

            # Sharpe ratio
            annual_return_excess = annual_return - RF_RATE
            sharpe_ratio = (
                annual_return_excess / annualized_volatility
                if annualized_volatility > 0
                else 0.0
            )

            stats_list.append(
                {
                    "Date": rebalance_date,
                    "Annualized_Return": annual_return,
                    "Volatility": annualized_volatility,
                    "Sharpe_Ratio": sharpe_ratio,
                }
            )

        benchmark_stats_over_time[ticker] = stats_list

    return benchmark_stats_over_time


def plot_benchmark_summary_over_time(
    benchmark_stats_over_time: Dict[str, List[Dict]],
    output_dir: str = ".",
) -> None:
    """
    Plot summary statistics over time for each benchmark.

    Creates individual 3-subplot figures for each benchmark showing:
    - Annualized Return Over Time
    - Volatility Over Time
    - Sharpe Ratio Over Time

    Args:
        benchmark_stats_over_time: Dict mapping benchmark_ticker -> List[Dict with stats]
        output_dir: Directory to save plots
    """
    for ticker, stats_list in benchmark_stats_over_time.items():
        if not stats_list or len(stats_list) < 2:
            logger.warning(f"Not enough data points for {ticker} summary stats plot.")
            continue

        # Convert list of dicts to DataFrame
        summary_df = pd.DataFrame(stats_list)
        summary_df["Date"] = pd.to_datetime(summary_df["Date"])
        summary_df = summary_df.sort_values("Date")

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
        axes[0].set_title(
            f"{ticker} - Annualized Return Over Time",
            fontsize=12,
            fontweight="bold",
        )
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
        axes[1].set_title(
            f"{ticker} - Volatility Over Time",
            fontsize=12,
            fontweight="bold",
        )
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
        axes[2].set_title(
            f"{ticker} - Sharpe Ratio Over Time",
            fontsize=12,
            fontweight="bold",
        )
        axes[2].set_xlabel("Date")
        axes[2].set_ylabel("Sharpe Ratio")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(
            output_dir, f"benchmark_summary_over_time_{ticker}.png"
        )
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        logger.info(f"Benchmark summary plot saved: {output_file}")
        plt.close()


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


def execute_run(
    plan: ConfiguredRun,
) -> Union[BacktestResult, StepForwardBacktestResult]:
    """Execute a configured run plan."""
    logger.info("=" * 70)
    logger.info(f"RUN PLAN: {plan.name}")
    logger.info("=" * 70)

    if hasattr(plan, "config") and not hasattr(plan, "base_config"):
        return run_backtest(plan.config)
    if hasattr(plan, "base_config"):
        return run_step_forward_evaluation(plan)

    raise TypeError(f"Unsupported run plan type: {type(plan)!r}")


def execute_runs(
    plans: List[ConfiguredRun],
) -> Dict[str, Union[BacktestResult, StepForwardBacktestResult]]:
    """Execute all configured runs and return results keyed by plan name."""
    results = {}
    for plan in plans:
        results[plan.name] = execute_run(plan)
    return results


def main() -> None:
    from runconfig import create_backtests

    plans = create_backtests()
    if not plans:
        logger.warning("No backtests configured in runconfig.create_backtests()")
        return

    execute_runs(plans)


if __name__ == "__main__":
    main()
