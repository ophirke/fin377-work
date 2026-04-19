"""
Year-by-year Rossa algorithm analysis for all tickers from tickers-all.txt (2001-present).

For each year, computes the core-periphery structure using the Rossa algorithm
with 365-day lookback, plots scatter plots of coreness vs forward returns,
fits a regression line, and tracks how the slope changes over time.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import rossa
from data import fetch_and_cache_stock_data, clean_price_data

# PARAMETERS
START_YEAR = 2001
END_YEAR = None  # None means current year
LOOKBACK_DAYS = 365
CORRELATION_THRESHOLD = 0.0  # Keep edges where |correlation| >= this threshold
SELECT_TOP_N = None  # Select top N tickers by rank (None = no limit)
SELECT_BOTTOM_N = 500  # Select bottom N tickers by rank (None = no limit)

# LOGGING
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# OUTPUT DIRECTORIES
def get_output_dir_name():
    """Generate output directory name based on rank filters."""
    name = "rossa_analysis_all_tickers"
    if SELECT_TOP_N:
        name += f"_top{SELECT_TOP_N}"
    elif SELECT_BOTTOM_N:
        name += f"_bottom{SELECT_BOTTOM_N}"
    return name

OUTPUT_DIR = Path(__file__).parent / get_output_dir_name()
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "yearly_plots"
PLOTS_DIR.mkdir(exist_ok=True)


def filter_tickers_by_rank(tickers: List[str]) -> List[str]:
    """Filter tickers based on rank parameters (SELECT_TOP_N or SELECT_BOTTOM_N)."""
    if SELECT_TOP_N:
        filtered = tickers[:SELECT_TOP_N]
        logger.info(f"Filtering to top {SELECT_TOP_N} tickers")
        return filtered
    elif SELECT_BOTTOM_N:
        filtered = tickers[-SELECT_BOTTOM_N:]
        logger.info(f"Filtering to bottom {SELECT_BOTTOM_N} tickers")
        return filtered
    return tickers


def load_all_tickers() -> List[str]:
    """Load tickers from data/tickers-all.txt file."""
    tickers_file = Path(__file__).parent.parent / "data" / "tickers-all.txt"
    
    if not tickers_file.exists():
        logger.error(f"Tickers file not found: {tickers_file}")
        return []
    
    with open(tickers_file, 'r') as f:
        tickers = [line.strip().upper() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(tickers)} tickers from {tickers_file}")
    return tickers


def apply_correlation_threshold(A: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """
    Filter adjacency matrix by correlation threshold.
    Keep only edges where |correlation| >= threshold.
    
    Args:
        A: Adjacency matrix (correlation-based)
        threshold: Minimum absolute correlation to keep an edge
    
    Returns:
        Filtered adjacency matrix
    """
    A_filtered = A * (np.abs(A) >= threshold).astype(float)
    return A_filtered


def analyze_year(
    year: int,
    tickers: List[str],
    lookback_days: int = 365,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run Rossa algorithm for a single year on all tickers and relate coreness to returns.
    
    Args:
        year: Analysis year
        tickers: List of ticker symbols to analyze
        lookback_days: Number of days of history to use (default 365)
    
    Returns:
        Tuple of (results_df, diagnostics_dict) where:
        - results_df: DataFrame with Stock, Coreness, and Returns columns
        - diagnostics_dict: Contains slope, r_squared, p_value, n_stocks, date_range
    """
    # For analysis: use year end date
    analysis_end_date = pd.Timestamp(year=year, month=12, day=31)
    if analysis_end_date > pd.Timestamp.today():
        analysis_end_date = pd.Timestamp.today()
    
    # For lookback window: go back lookback_days from analysis end date
    analysis_start_date = analysis_end_date - timedelta(days=lookback_days)
    
    # For returns calculation: use the forward lookback_days from analysis end date
    # (forward returns start FROM the analysis date, not after it)
    returns_start_date = analysis_end_date
    returns_end_date = returns_start_date + timedelta(days=lookback_days)
    if returns_end_date > pd.Timestamp.today():
        returns_end_date = pd.Timestamp.today()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Year {year}")
    logger.info(f"  Analysis window: {analysis_start_date.date()} to {analysis_end_date.date()}")
    logger.info(f"  Returns window: {returns_start_date.date()} to {returns_end_date.date()}")
    logger.info(f"{'='*70}")
    
    try:
        # Create a mapping of ticker to its rank (order in the list)
        ticker_to_rank = {ticker: i + 1 for i, ticker in enumerate(tickers)}
        
        logger.info(f"Analyzing {len(tickers)} tickers")
        
        # Fetch price data for BOTH analysis and returns windows
        logger.info(f"Fetching price data for {len(tickers)} tickers...")
        price_data = fetch_and_cache_stock_data(tuple(tickers))
        
        # Split into analysis and returns windows
        price_data_analysis = price_data[
            (price_data.index >= analysis_start_date) & (price_data.index <= analysis_end_date)
        ]
        price_data_returns = price_data[
            (price_data.index >= returns_start_date) & (price_data.index <= returns_end_date)
        ]
        
        if price_data_analysis.empty or price_data_returns.empty:
            logger.error(f"  No price data available for {year}")
            return None, {"error": "No price data"}
        
        logger.info(f"  Analysis data: {price_data_analysis.shape[1]} tickers, {price_data_analysis.shape[0]} days")
        logger.info(f"  Returns data: {price_data_returns.shape[1]} tickers, {price_data_returns.shape[0]} days")
        
        # Clean both windows
        logger.info("Cleaning price data...")
        price_data_analysis = clean_price_data(price_data_analysis, min_price=1.0, max_price=10000.0)
        price_data_returns = clean_price_data(price_data_returns, min_price=1.0, max_price=10000.0)
        
        # Prepare analysis data for Rossa
        logger.info("Preparing analysis data...")
        price_data_analysis_clean, log_returns_analysis = rossa.prepare_price_data(price_data_analysis)
        
        if log_returns_analysis.empty or log_returns_analysis.shape[1] < 2:
            logger.error(f"  Insufficient overlapping data for {year}")
            return None, {"error": "Insufficient data"}
        
        n_stocks_analysis = log_returns_analysis.shape[1]
        logger.info(f"  → Analyzing {n_stocks_analysis} stocks with overlapping data")
        
        # Build adjacency matrix from analysis period
        logger.info("Building adjacency matrix...")
        A, ticker_names = rossa.build_adjacency_matrix(log_returns_analysis)
        
        # Apply correlation threshold filtering
        logger.info(f"Applying correlation threshold (|r| >= {CORRELATION_THRESHOLD})...")
        n_edges_before = np.sum(A > 0) / 2  # Divide by 2 because matrix is symmetric
        A = apply_correlation_threshold(A, threshold=CORRELATION_THRESHOLD)
        n_edges_after = np.sum(A > 0) / 2
        logger.info(f"  Correlation filtering: {int(n_edges_before)} edges → {int(n_edges_after)} edges")
        
        # Run Rossa algorithm
        logger.info("Running Rossa algorithm on correlation-filtered network...")
        results = rossa.rossa_core_periphery(A, ticker_names)
        
        # Calculate returns for each stock in the RETURNS window
        logger.info("Calculating returns...")
        price_data_returns_clean, log_returns_forward = rossa.prepare_price_data(price_data_returns)
        
        # Calculate total return for each stock over the returns window
        returns_dict = {}
        for ticker in results["Stock"]:
            if ticker in log_returns_forward.columns:
                # Total return over the period
                total_return = log_returns_forward[ticker].sum()
                returns_dict[ticker] = total_return
            else:
                returns_dict[ticker] = np.nan
        
        results["Returns"] = results["Stock"].map(returns_dict)
        
        # Add rank for each stock
        results["Rank"] = results["Stock"].map(ticker_to_rank)
        
        # Remove stocks with missing returns
        results_clean = results.dropna(subset=["Returns"]).copy()
        
        if len(results_clean) < 2:
            logger.error(f"  Not enough stocks with both coreness and returns data")
            return None, {"error": "Insufficient data"}
        
        logger.info(f"  → {len(results_clean)} stocks have both coreness and returns data")
        
        # Fit regression: Returns ~ Coreness
        X = results_clean["Coreness"].values
        y = results_clean["Returns"].values
        
        # Fit OLS regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        r_squared = r_value ** 2
        
        diagnostics = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
            "n_stocks": len(results_clean),
            "date_range_analysis": f"{analysis_start_date.date()} to {analysis_end_date.date()}",
            "date_range_returns": f"{returns_start_date.date()} to {returns_end_date.date()}",
        }
        
        logger.info(f"Regression results (Coreness vs Forward Returns):")
        logger.info(f"  Slope: {slope:.6f} (return per unit coreness)")
        logger.info(f"  R²: {r_squared:.6f}")
        logger.info(f"  P-value: {p_value:.2e}")
        logger.info(f"  N stocks: {len(results_clean)}")
        
        return results_clean, diagnostics
        
    except Exception as e:
        logger.error(f"Error analyzing year {year}: {e}")
        import traceback
        traceback.print_exc()
        return None, {"error": str(e)}


def plot_yearly_scatter(
    results: pd.DataFrame,
    year: int,
    diagnostics: Dict[str, float],
) -> None:
    """Plot scatter plot of coreness vs returns for a single year with stock labels, colored by ticker rank."""
    if results is None or "error" in diagnostics:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create colormap from green (rank 1) to purple (high rank)
    cmap = mcolors.LinearSegmentedColormap.from_list('green_purple', ['green', 'purple'], N=500)
    
    # Normalize ranks to [0, 1] for colormap
    ranks = results["Rank"].values
    max_rank = ranks.max()
    norm = mcolors.Normalize(vmin=1, vmax=max(500, max_rank))
    colors = cmap(norm(ranks))
    
    # Scatter plot with colors based on rank
    ax.scatter(
        results["Coreness"],
        results["Returns"],
        alpha=0.6,
        s=50,
        c=colors,
    )
    
    # Add labels for each point
    for idx, row in results.iterrows():
        ax.annotate(
            row["Stock"],
            (row["Coreness"], row["Returns"]),
            fontsize=8,
            alpha=0.7,
            xytext=(5, 5),
            textcoords="offset points",
        )
    
    # Regression line
    X = results["Coreness"].values
    y_pred = diagnostics["slope"] * X + diagnostics["intercept"]
    ax.plot(X, y_pred, "r-", linewidth=2, label="Regression line")
    
    ax.set_xlabel("Coreness Score", fontsize=11)
    ax.set_ylabel("Forward Returns (Next 365 Days)", fontsize=11)
    
    # Generate title with rank filter info
    ticker_desc = "All Tickers"
    if SELECT_TOP_N:
        ticker_desc = f"Top {SELECT_TOP_N} Tickers"
    elif SELECT_BOTTOM_N:
        ticker_desc = f"Bottom {SELECT_BOTTOM_N} Tickers"
    
    ax.set_title(
        f"Coreness vs Forward Returns - {year} ({ticker_desc}, Correlation-filtered)\\n"
        f"R² = {diagnostics['r_squared']:.4f}, "
        f"Slope = {diagnostics['slope']:.6f}, "
        f"P-value = {diagnostics['p_value']:.2e}",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add colorbar to show rank scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Ticker Rank")
    
    # Save
    output_file = PLOTS_DIR / f"coreness_scatter_{year}.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {output_file}")


def plot_slope_over_time(
    years: List[int],
    slopes: List[float],
    r_squared_values: List[float],
    p_values: List[float],
) -> None:
    """Plot how regression slope changes over time."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Slope over time
    axes[0].plot(years, slopes, marker="o", linewidth=2, markersize=6, color="steelblue")
    axes[0].axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    axes[0].set_ylabel("Slope (Returns vs Coreness)", fontsize=11)
    
    # Generate title with rank filter info
    ticker_desc = "All Tickers"
    if SELECT_TOP_N:
        ticker_desc = f"Top {SELECT_TOP_N} Tickers"
    elif SELECT_BOTTOM_N:
        ticker_desc = f"Bottom {SELECT_BOTTOM_N} Tickers"
    
    axes[0].set_title(f"Regression Slope Over Time ({ticker_desc}, Correlation-filtered)", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # R² over time
    axes[1].plot(years, r_squared_values, marker="s", linewidth=2, markersize=6, color="darkgreen")
    axes[1].set_ylabel("R² (Goodness of Fit)", fontsize=11)
    axes[1].set_title("Regression R² Over Time", fontsize=12)
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    # P-value over time (log scale)
    axes[2].semilogy(years, p_values, marker="^", linewidth=2, markersize=6, color="darkred")
    axes[2].axhline(y=0.05, color="k", linestyle="--", linewidth=1, alpha=0.5, label="α=0.05")
    axes[2].set_xlabel("Year", fontsize=11)
    axes[2].set_ylabel("P-value (log scale)", fontsize=11)
    axes[2].set_title("Regression P-value Over Time (Significance)", fontsize=12)
    axes[2].grid(True, alpha=0.3, which="both")
    axes[2].legend()
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "slope_over_time.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {output_file}")


def main():
    """Run year-by-year Rossa analysis."""
    # Load tickers from file
    tickers = load_all_tickers()
    
    if not tickers:
        logger.error("No tickers loaded")
        return
    
    # Apply rank filtering
    tickers = filter_tickers_by_rank(tickers)
    
    if not tickers:
        logger.error("No tickers after filtering")
        return
    
    logger.info(f"Starting year-by-year Rossa analysis for {len(tickers)} tickers ({START_YEAR}-present)")
    
    # Determine year range
    current_year = datetime.now().year
    end_year = END_YEAR or current_year
    
    # Storage for results across years
    years = []
    slopes = []
    r_squared_values = []
    p_values = []
    n_stocks_list = []
    all_results = {}
    
    # Analyze each year
    for year in range(START_YEAR, end_year + 1):
        results, diagnostics = analyze_year(year, tickers=tickers, lookback_days=LOOKBACK_DAYS)
        
        if results is not None and "error" not in diagnostics:
            years.append(year)
            slopes.append(diagnostics["slope"])
            r_squared_values.append(diagnostics["r_squared"])
            p_values.append(diagnostics["p_value"])
            n_stocks_list.append(diagnostics["n_stocks"])
            all_results[year] = results
            
            # Plot scatter for this year
            plot_yearly_scatter(results, year, diagnostics)
        else:
            logger.warning(f"Skipping year {year}: {diagnostics.get('error', 'Unknown error')}")
    
    if not years:
        logger.error("No years analyzed successfully")
        return
    
    # Plot slope over time
    plot_slope_over_time(years, slopes, r_squared_values, p_values)
    
    # Summary table
    summary_df = pd.DataFrame({
        "Year": years,
        "Slope": slopes,
        "R²": r_squared_values,
        "P-value": p_values,
        "N_Stocks": n_stocks_list,
    })
    
    logger.info("\n" + "="*70)
    summary_title = "SUMMARY STATISTICS"
    if SELECT_TOP_N:
        summary_title += f" (Top {SELECT_TOP_N} Tickers)"
    elif SELECT_BOTTOM_N:
        summary_title += f" (Bottom {SELECT_BOTTOM_N} Tickers)"
    else:
        summary_title += " (All Tickers)"
    logger.info(summary_title)
    logger.info("="*70)
    print(summary_df.to_string(index=False))
    
    # Save summary to CSV
    summary_file = OUTPUT_DIR / "rossa_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"\nSaved summary: {summary_file}")
    
    # Save detailed results for each year
    for year, results in all_results.items():
        year_file = OUTPUT_DIR / f"rossa_results_{year}.csv"
        results.to_csv(year_file, index=False)
    logger.info(f"Saved yearly results to {OUTPUT_DIR}/rossa_results_*.csv")
    
    logger.info(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
