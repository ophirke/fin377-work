"""
Rolling window Rossa analysis: 60-day coreness window, 10-day forward returns.

For each year, analyzes every 10-day interval:
- Uses last 60 days to compute coreness via Rossa
- Computes forward 10-day returns
- Creates scatter plots for each interval
- Generates a video showing evolution throughout the year
- Uses multiprocessing for speed
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Pool, cpu_count
import imageio

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import rossa
from datamarshal import load_sp500_constituents
from data import fetch_and_cache_stock_data, clean_price_data

# PARAMETERS
N_STOCKS = 500  # Bottom N stocks from S&P 500
START_YEAR = 2001
END_YEAR = None  # None means current year
LOOKBACK_DAYS = 60
FORWARD_DAYS = 10
INTERVAL_DAYS = 10

# MULTIPROCESSING
NUM_WORKERS = max(1, min(12, cpu_count() - 4))

# LOGGING
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

# OUTPUT DIRECTORIES
OUTPUT_DIR = Path(__file__).parent / f"rossa_rolling_sp500_bottom{N_STOCKS}"
OUTPUT_DIR.mkdir(exist_ok=True)
FRAMES_DIR = OUTPUT_DIR / "frames"
FRAMES_DIR.mkdir(exist_ok=True)
VIDEOS_DIR = OUTPUT_DIR / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)


def analyze_window(
    analysis_end_date: pd.Timestamp,
    lookback_days: int,
    forward_days: int,
    selected_tickers: List[str],
    ticker_to_rank: Dict[str, int],
    price_data_full: pd.DataFrame,
    year: int,
    interval_num: int,
) -> Optional[Tuple[pd.DataFrame, Dict[str, float], pd.Timestamp]]:
    """
    Analyze a single window: use last lookback_days to compute coreness,
    then predict forward_days returns.
    
    Returns: (results_df, diagnostics, analysis_end_date)
    """
    # Lookback window for coreness
    analysis_start_date = analysis_end_date - timedelta(days=lookback_days)
    
    # Forward window for returns
    returns_start_date = analysis_end_date
    returns_end_date = returns_start_date + timedelta(days=forward_days)
    
    try:
        # Extract price data for this window
        price_data_analysis = price_data_full[
            (price_data_full.index >= analysis_start_date) & (price_data_full.index <= analysis_end_date)
        ]
        price_data_returns = price_data_full[
            (price_data_full.index >= returns_start_date) & (price_data_full.index <= returns_end_date)
        ]
        
        if price_data_analysis.empty or len(price_data_analysis) < 10:
            return None
        
        # Prepare analysis data
        price_data_analysis_clean, log_returns_analysis = rossa.prepare_price_data(price_data_analysis)
        
        if log_returns_analysis.empty or log_returns_analysis.shape[1] < 2:
            return None
        
        # Build adjacency and run Rossa
        A, ticker_names = rossa.build_adjacency_matrix(log_returns_analysis)
        results = rossa.rossa_core_periphery(A, ticker_names)
        
        # Calculate forward returns
        if not price_data_returns.empty:
            price_data_returns_clean, log_returns_forward = rossa.prepare_price_data(price_data_returns)
            
            returns_dict = {}
            for ticker in results["Stock"]:
                if ticker in log_returns_forward.columns and len(log_returns_forward) > 0:
                    total_return = log_returns_forward[ticker].sum()
                    returns_dict[ticker] = total_return
                else:
                    returns_dict[ticker] = np.nan
        else:
            returns_dict = {ticker: np.nan for ticker in results["Stock"]}
        
        results["Returns"] = results["Stock"].map(returns_dict)
        results["Rank"] = results["Stock"].map(ticker_to_rank)
        
        # Remove stocks with missing returns
        results_clean = results.dropna(subset=["Returns"]).copy()
        
        if len(results_clean) < 2:
            return None
        
        # Fit regression
        X = results_clean["Coreness"].values
        y = results_clean["Returns"].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
        r_squared = r_value ** 2
        
        diagnostics = {
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_squared,
            "p_value": p_value,
            "n_stocks": len(results_clean),
            "analysis_date": analysis_end_date,
        }
        
        return results_clean, diagnostics, analysis_end_date
        
    except Exception as e:
        logger.warning(f"Error analyzing window {year} interval {interval_num}: {e}")
        return None


def plot_window_scatter(
    results: pd.DataFrame,
    diagnostics: Dict[str, float],
    year: int,
    interval_num: int,
    output_path: Path,
) -> None:
    """Plot scatter for a single window."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colormap: green (rank 1) to purple (rank 500)
    cmap = mcolors.LinearSegmentedColormap.from_list('green_purple', ['green', 'purple'], N=500)
    
    ranks = results["Rank"].values
    norm = mcolors.Normalize(vmin=1, vmax=500)
    colors = cmap(norm(ranks))
    
    # Scatter with colors by rank
    ax.scatter(
        results["Coreness"],
        results["Returns"],
        alpha=0.6,
        s=40,
        c=colors,
    )
    
    # Regression line
    X = results["Coreness"].values
    y_pred = diagnostics["slope"] * X + diagnostics["intercept"]
    ax.plot(X, y_pred, "r-", linewidth=2, label="Regression line")
    
    ax.set_xlabel("Coreness Score", fontsize=10)
    ax.set_ylabel(f"Forward {FORWARD_DAYS}-Day Returns", fontsize=10)
    
    analysis_date = diagnostics["analysis_date"].strftime("%Y-%m-%d")
    ax.set_title(
        f"Year {year} - Interval {interval_num} ({analysis_date})\n"
        f"R² = {diagnostics['r_squared']:.4f}, "
        f"Slope = {diagnostics['slope']:.6f}, "
        f"P-value = {diagnostics['p_value']:.2e}, "
        f"N = {diagnostics['n_stocks']}",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="S&P 500 Rank")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()


def process_year_windows(
    year: int,
    n_stocks: int = 500,
) -> List[Tuple[int, pd.DataFrame, Dict, pd.Timestamp]]:
    """
    Process all windows for a single year.
    Returns list of (interval_num, results_df, diagnostics, analysis_date)
    """
    logger.info(f"\nProcessing year {year}...")
    
    # Get constituents and build rank mapping
    analysis_end_date = pd.Timestamp(year=year, month=12, day=31)
    if analysis_end_date > pd.Timestamp.today():
        analysis_end_date = pd.Timestamp.today()
    
    tickers_date = analysis_end_date.strftime("%Y-%m-%d")
    all_tickers = load_sp500_constituents(tickers_date)
    ticker_to_rank = {ticker: i + 1 for i, ticker in enumerate(all_tickers)}
    selected_tickers = list(all_tickers)[-n_stocks:]
    
    logger.info(f"  Loading price data for {len(selected_tickers)} tickers...")
    price_data_full = fetch_and_cache_stock_data(tuple(selected_tickers))
    
    # Get all 10-day intervals for this year
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year, month=12, day=31)
    if year_end > pd.Timestamp.today():
        year_end = pd.Timestamp.today()
    
    current_date = year_start + timedelta(days=LOOKBACK_DAYS)  # Need enough history
    intervals = []
    interval_num = 1
    
    while current_date <= year_end:
        intervals.append((current_date, interval_num))
        current_date += timedelta(days=INTERVAL_DAYS)
        interval_num += 1
    
    logger.info(f"  Analyzing {len(intervals)} intervals...")
    
    # Process each interval
    results_list = []
    for analysis_date, interval_num in intervals:
        result = analyze_window(
            analysis_date,
            LOOKBACK_DAYS,
            FORWARD_DAYS,
            selected_tickers,
            ticker_to_rank,
            price_data_full,
            year,
            interval_num,
        )
        
        if result is not None:
            results_df, diagnostics, analysis_end = result
            results_list.append((interval_num, results_df, diagnostics, analysis_end))
    
    logger.info(f"  Successfully analyzed {len(results_list)} intervals")
    return results_list, year


def create_video_for_year(
    year: int,
    window_results: List[Tuple[int, pd.DataFrame, Dict, pd.Timestamp]],
) -> None:
    """
    Create frames and video for a single year.
    """
    logger.info(f"\nCreating video for year {year}...")
    
    year_frames_dir = FRAMES_DIR / f"year_{year}"
    year_frames_dir.mkdir(exist_ok=True)
    
    frame_files = []
    
    for interval_num, results_df, diagnostics, analysis_date in window_results:
        frame_path = year_frames_dir / f"frame_{interval_num:03d}.png"
        plot_window_scatter(results_df, diagnostics, year, interval_num, frame_path)
        frame_files.append(str(frame_path))
    
    if not frame_files:
        logger.warning(f"No frames generated for year {year}")
        return
    
    # Create video using imageio
    logger.info(f"  Creating video from {len(frame_files)} frames...")
    video_path = VIDEOS_DIR / f"rossa_rolling_{year}.mp4"
    
    # Read frames and write to video (2 fps = 5 second video for 10 frames)
    with imageio.get_writer(str(video_path), fps=2, codec='libx264', quality=7) as writer:
        for frame_file in sorted(frame_files):
            image = imageio.imread(frame_file)
            writer.append_data(image)
    
    logger.info(f"  Video saved: {video_path}")


def plot_all_slopes_over_time(
    all_data: List[Tuple[int, int, float, float, float]],
) -> None:
    """
    Plot slopes across all years and intervals.
    all_data: List of (year, interval_num, slope, p_value, analysis_date_ordinal)
    """
    if not all_data:
        logger.warning("No data to plot")
        return
    
    years = [d[0] for d in all_data]
    intervals = [d[1] for d in all_data]
    slopes = [d[2] for d in all_data]
    p_values = [d[3] for d in all_data]
    dates = [datetime.fromordinal(int(d[4])) for d in all_data]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Slopes over time
    ax1.plot(range(len(slopes)), slopes, marker="o", linewidth=1.5, markersize=4, color="steelblue", alpha=0.7)
    ax1.axhline(y=0, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_ylabel("Slope (Forward Return per Unit Coreness)", fontsize=11)
    ax1.set_title(
        f"Slopes Over All Intervals (Years {START_YEAR}-{max(years)})\n"
        f"Lookback: {LOOKBACK_DAYS} days, Forward: {FORWARD_DAYS} days, Interval: {INTERVAL_DAYS} days",
        fontsize=12,
    )
    ax1.grid(True, alpha=0.3)
    
    # Add vertical lines for year boundaries
    interval_idx = 0
    for year in range(START_YEAR, max(years) + 1):
        year_intervals = [d for d in all_data if d[0] == year]
        if year_intervals:
            interval_idx += len(year_intervals)
            if interval_idx < len(slopes):
                ax1.axvline(x=interval_idx, color="gray", linestyle=":", alpha=0.3)
                ax1.text(interval_idx, ax1.get_ylim()[1], str(year), fontsize=8, alpha=0.5)
    
    # Plot 2: P-values (log scale)
    ax2.semilogy(range(len(p_values)), p_values, marker="s", linewidth=1.5, markersize=4, color="darkred", alpha=0.7)
    ax2.axhline(y=0.05, color="k", linestyle="--", linewidth=1, alpha=0.5, label="α=0.05")
    ax2.set_xlabel("Interval Number (across all years)", fontsize=11)
    ax2.set_ylabel("P-value (log scale)", fontsize=11)
    ax2.set_title("Statistical Significance of Slopes Over Time", fontsize=12)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend()
    
    # Add vertical lines for year boundaries in p-value plot too
    interval_idx = 0
    for year in range(START_YEAR, max(years) + 1):
        year_intervals = [d for d in all_data if d[0] == year]
        if year_intervals:
            interval_idx += len(year_intervals)
            if interval_idx < len(p_values):
                ax2.axvline(x=interval_idx, color="gray", linestyle=":", alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "all_slopes_over_time.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved master slope plot: {output_file}")


def main():
    """Main analysis loop."""
    logger.info(
        f"Starting rolling window Rossa analysis\n"
        f"  Lookback: {LOOKBACK_DAYS} days, Forward: {FORWARD_DAYS} days, Interval: {INTERVAL_DAYS} days\n"
        f"  Workers: {NUM_WORKERS}"
    )
    
    current_year = datetime.now().year
    end_year = END_YEAR or current_year
    
    # Collect all data for master plot
    all_slopes_data = []
    
    for year in range(START_YEAR, end_year + 1):
        try:
            window_results, processed_year = process_year_windows(year, n_stocks=N_STOCKS)
            
            if window_results:
                create_video_for_year(year, window_results)
                
                # Save summary stats
                slopes = [d["slope"] for _, _, d, _ in window_results]
                r_squared_list = [d["r_squared"] for _, _, d, _ in window_results]
                p_values = [d["p_value"] for _, _, d, _ in window_results]
                intervals = [interval_num for interval_num, _, _, _ in window_results]
                analysis_dates = [d["analysis_date"] for _, _, d, _ in window_results]
                
                logger.info(
                    f"Year {year} summary:\n"
                    f"  Avg Slope: {np.mean(slopes):.6f} (min: {np.min(slopes):.6f}, max: {np.max(slopes):.6f})\n"
                    f"  Avg R²: {np.mean(r_squared_list):.4f}\n"
                    f"  Avg P-value: {np.mean(p_values):.2e}\n"
                    f"  Significant (p<0.05): {sum(1 for p in p_values if p < 0.05)}/{len(p_values)}"
                )
                
                # Collect data for master plot
                for interval_num, slope, p_value, analysis_date in zip(intervals, slopes, p_values, analysis_dates):
                    all_slopes_data.append((year, interval_num, slope, p_value, analysis_date.toordinal()))
            else:
                logger.warning(f"No valid windows for year {year}")
                
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create master plot of all slopes
    if all_slopes_data:
        plot_all_slopes_over_time(all_slopes_data)
    
    logger.info(f"\nAll videos saved to: {VIDEOS_DIR}")


if __name__ == "__main__":
    main()
