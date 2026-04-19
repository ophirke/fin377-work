"""
Year-by-year Rossa algorithm analysis (2001-present).

For each year, computes the core-periphery structure using the Rossa algorithm
with 365-day lookback, plots scatter plots of coreness scores, fits a regression
line, and tracks how the slope changes over time.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import rossa
from datamarshal import load_sp500_constituents
from data import fetch_and_cache_stock_data, clean_price_data

# LOGGING
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# OUTPUT DIRECTORIES
OUTPUT_DIR = Path(__file__).parent / "rossa_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = OUTPUT_DIR / "yearly_plots"
PLOTS_DIR.mkdir(exist_ok=True)


def analyze_year(
    year: int,
    lookback_days: int = 365,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run Rossa algorithm for a single year and relate coreness to returns.
    
    Args:
        year: Analysis year
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
    
    # For returns calculation: use the YEAR after analysis (next 365 days)
    returns_start_date = analysis_end_date + timedelta(days=1)
    returns_end_date = returns_start_date + timedelta(days=lookback_days)
    if returns_end_date > pd.Timestamp.today():
        returns_end_date = pd.Timestamp.today()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Year {year}")
    logger.info(f"  Analysis window: {analysis_start_date.date()} to {analysis_end_date.date()}")
    logger.info(f"  Returns window: {returns_start_date.date()} to {returns_end_date.date()}")
    logger.info(f"{'='*70}")
    
    try:
        # Get SP500 constituents as of end of year
        tickers_date = analysis_end_date.strftime("%Y-%m-%d")
        tickers = load_sp500_constituents(tickers_date)
        logger.info(f"Loaded {len(tickers)} SP500 constituents")
        
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
        
        # Run Rossa algorithm
        logger.info("Running Rossa algorithm...")
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
    """Plot scatter plot of coreness vs returns for a single year with stock labels."""
    if results is None or "error" in diagnostics:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Scatter plot
    ax.scatter(
        results["Coreness"],
        results["Returns"],
        alpha=0.6,
        s=50,
        color="steelblue",
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
    ax.set_title(
        f"Coreness vs Forward Returns - {year}\n"
        f"R² = {diagnostics['r_squared']:.4f}, "
        f"Slope = {diagnostics['slope']:.6f}, "
        f"P-value = {diagnostics['p_value']:.2e}",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    
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
    axes[0].set_ylabel("Slope (Coreness vs Rank)", fontsize=11)
    axes[0].set_title("Regression Slope Over Time", fontsize=12)
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
    logger.info("Starting year-by-year Rossa analysis (2001-present)")
    
    # Determine year range
    current_year = datetime.now().year
    start_year = 2001
    
    # Storage for results across years
    years = []
    slopes = []
    r_squared_values = []
    p_values = []
    n_stocks_list = []
    all_results = {}
    
    # Analyze each year
    for year in range(start_year, current_year + 1):
        results, diagnostics = analyze_year(year, lookback_days=365)
        
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
    logger.info("SUMMARY STATISTICS")
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
