import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from typing import Optional, List, Tuple


def fetch_and_cache_stock_data(
    ticker_list: List[str],
    cache_file: str = 'india_stocks_history.csv',
    period: str = '10y'
) -> pd.DataFrame:
    """
    Fetches historical stock data, manages cache intelligently.
    
    Verifies which tickers are in cache, fetches missing ones, and updates cache.
    Failed/delisted tickers are saved to cache with NaN values to avoid re-downloading.
    This ensures complete data without redundant downloads.
    
    Args:
        ticker_list: List of ticker symbols to fetch
        cache_file: Path to cache CSV file
        period: Historical period to fetch (e.g., '10y', '5y')
    
    Returns:
        DataFrame with Close prices for all requested tickers (excluding failed tickers)
    """
    cached_tickers = set()
    cached_data = None
    
    # Load existing cache if available (and non-empty)
    if os.path.exists(cache_file):
        print(f"Loading existing cache: {cache_file}...")
        cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Only use cache if it has data rows; empty cache is corrupted
        if len(cached_data) > 0:
            cached_tickers = set(cached_data.columns)
        else:
            print(f"  → Cache is empty, will re-download all tickers")
            cached_data = None
    
    # Identify missing tickers
    missing_tickers = [t for t in ticker_list if t not in cached_tickers]
    
    if missing_tickers:
        print(f"Downloading {len(missing_tickers)}/{len(ticker_list)} missing tickers...")
        downloaded = yf.download(missing_tickers, period=period, threads=False)
        
        # Extract 'Close' from MultiIndex columns (when multiple tickers)
        if isinstance(downloaded.columns, pd.MultiIndex):
            new_data = downloaded.xs('Close', level=0, axis=1)
        else:
            new_data = downloaded[['Close']]
        
        # CRITICAL FIX: Keep failed tickers with NaN values so they're not re-downloaded
        # Only drop columns that are completely empty (shouldn't happen), but keep all-NaN columns
        # from failed downloads to mark them as "attempted but failed"
        
        # Merge with cached data
        if cached_data is not None and len(cached_data) > 0:
            # Align indices and combine (fills with NaN for missing dates in new data)
            all_data = pd.concat([cached_data, new_data], axis=1)
        else:
            all_data = new_data
    else:
        if cached_data is None or len(cached_data) == 0:
            raise ValueError(f"No cache found and no tickers to download")
        all_data = cached_data
    
    # Keep only the requested tickers that were attempted (including failed ones with NaN)
    available_tickers = [t for t in ticker_list if t in all_data.columns]
    all_data = all_data[available_tickers]
    
    # IMPORTANT: Only save cache if we have actual data (not just NaN columns)
    if len(all_data) > 0:
        all_data.to_csv(cache_file)
        print(f"Cache updated: {cache_file}")
    else:
        print(f"WARNING: No data to save to cache (all downloads failed)")
    
    # Before returning, drop columns that are entirely NaN (failed downloads)
    # These stay in cache for next run, but we don't analyze them
    all_data = all_data.dropna(axis=1, how='all')
    
    return all_data


def prepare_price_data(
    price_data: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    missing_tolerance: float = 0.20
) -> Tuple[pd.DataFrame, int]:
    """
    Cleans and prepares price data for analysis.
    
    - Filters by date range
    - Drops stocks missing >missing_tolerance (default 20%) of data
    - Forward/backward fills NaN values
    - Removes remaining NaNs
    
    Args:
        price_data: DataFrame with close prices (index: date, columns: tickers)
        start_date: Start date (YYYY-MM-DD) or None for earliest
        end_date: End date (YYYY-MM-DD) or None for latest
        missing_tolerance: Max fraction of missing data allowed (default 0.20 = 20%)
    
    Returns:
        Tuple of (cleaned_data, log_returns)
        - cleaned_data: Price data after preprocessing
        - log_returns: Log returns for correlation analysis
    """
    # Filter by date range
    if start_date:
        price_data = price_data[price_data.index >= start_date]
    if end_date:
        price_data = price_data[price_data.index <= end_date]
    
    initial_tickers = price_data.shape[1]
    
    # Drop stocks missing more than missing_tolerance of data history
    threshold = int(len(price_data) * (1.0 - missing_tolerance))
    price_data = price_data.dropna(axis=1, thresh=threshold)
    
    dropped_tickers = initial_tickers - price_data.shape[1]
    if dropped_tickers > 0:
        print(f"  Dropped {dropped_tickers} stocks with >{missing_tolerance*100:.0f}% missing data")
    
    # Fill small gaps (trading halts, holidays) forward, then backward
    price_data = price_data.ffill().bfill()
    
    # Drop any remaining rows with NaNs
    price_data = price_data.dropna()
    
    if price_data.empty or price_data.shape[1] < 2:
        print(f"  ERROR: Only {price_data.shape[1]} stocks remain with overlapping data")
        raise ValueError("Not enough overlapping historical data to build a network")
    
    # Calculate log returns
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    
    print(f"  → Analyzing {log_returns.shape[1]} stocks over {log_returns.shape[0]} trading days")
    print(f"  → Date range: {price_data.index[0].date()} to {price_data.index[-1].date()}")
    
    return price_data, log_returns


def build_adjacency_matrix(log_returns: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
    """
    Builds weighted adjacency matrix from log returns using Pearson correlation.
    
    Weight transformation: A_ij = (1 + rho_ij) / 2, mapping [-1, 1] to [0, 1]
    
    Args:
        log_returns: DataFrame of log returns (rows: time, columns: tickers)
    
    Returns:
        Tuple of (adjacency_matrix, ticker_names)
    """
    rho = log_returns.corr()
    tickers = rho.columns
    
    # Adjacency weight transformation: (1 + rho_ij) / 2
    A = (1 + rho.values) / 2
    np.fill_diagonal(A, 0)  # Remove self-loops
    
    # Handle NaNs (e.g., zero variance stocks)
    A = np.nan_to_num(A, nan=0.0)
    
    return A, tickers


def rossa_core_periphery(
    A: np.ndarray,
    tickers: pd.Index
) -> pd.DataFrame:
    """
    Implements core-periphery profile algorithm by Rossa et al.
    
    Builds a core set S by iteratively adding vertices that maximize
    internal connectivity (phi score). Returns tickers ranked by coreness.
    
    Args:
        A: Adjacency matrix
        tickers: Stock ticker names
    
    Returns:
        DataFrame with columns ['Stock', 'Coreness'] sorted by coreness
    """
    N = A.shape[0]
    degrees = A.sum(axis=1)  # Weighted degree
    
    # Start with peripheral vertex (minimum degree)
    first_vertex = np.argmin(degrees)
    S = [first_vertex]
    unvisited = set(range(N))
    unvisited.remove(first_vertex)
    
    phi_list = [0.0]
    order = [first_vertex]
    
    current_internal_sum = 0.0
    current_degree_sum = degrees[first_vertex]
    
    while unvisited:
        min_phi = float('inf')
        best_v = None
        best_internal_sum = 0.0
        
        for j in unvisited:
            edges_to_S = 2 * A[j, S].sum()
            test_internal = current_internal_sum + edges_to_S
            test_degree = current_degree_sum + degrees[j]
            
            # Phi score: internal density of core set
            phi_S = test_internal / test_degree if test_degree > 0 else float('inf')
            
            if phi_S < min_phi:
                min_phi = phi_S
                best_v = j
                best_internal_sum = test_internal
        
        # Failsafe for disconnected graphs
        if best_v is None:
            best_v = next(iter(unvisited))
            min_phi = 1.0
        
        S.append(best_v)
        unvisited.remove(best_v)
        order.append(best_v)
        phi_list.append(min_phi)
        
        current_internal_sum = best_internal_sum
        current_degree_sum += degrees[best_v]
    
    return pd.DataFrame({
        'Stock': [tickers[i] for i in order],
        'Coreness': phi_list
    })


def plot_network(
    A: np.ndarray,
    ticker_names: pd.Index,
    results: pd.DataFrame,
    filename: str,
    corr_threshold: float = 0.3
) -> None:
    """
    Visualizes stock correlation network with core-periphery structure.
    
    - Core stocks positioned at center
    - Peripheral stocks positioned outside
    - Red edges: positive correlations
    - Blue edges: negative correlations
    - Thickness: correlation strength
    
    Args:
        A: Adjacency matrix
        ticker_names: Stock ticker names
        results: DataFrame with coreness scores
        filename: Output PNG filename
        corr_threshold: Minimum correlation magnitude to show edge
    """
    G = nx.Graph()
    
    # Add nodes
    for ticker in ticker_names:
        G.add_node(ticker)
    
    # Get coreness for each stock
    coreness_dict = dict(zip(results['Stock'], results['Coreness']))
    
    # Add edges based on correlation threshold
    for i in range(len(ticker_names)):
        for j in range(i + 1, len(ticker_names)):
            correlation = A[i, j] * 2 - 1  # Map [0, 1] back to [-1, 1]
            if abs(correlation) > corr_threshold:
                G.add_edge(ticker_names[i], ticker_names[j], weight=correlation)
    
    # Position nodes: map coreness to radius (core at center, periphery outside)
    coreness_values = results['Coreness'].values
    min_core = coreness_values.min()
    max_core = coreness_values.max()
    
    pos = {}
    n_stocks = len(ticker_names)
    for idx, (stock, coreness) in enumerate(coreness_dict.items()):
        # Invert: high coreness -> small radius (center)
        normalized = (coreness - min_core) / (max_core - min_core + 1e-10)
        radius = 1.0 + 3.0 * (1.0 - normalized)
        angle = 2 * np.pi * idx / n_stocks
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pos[stock] = (x, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Draw edges with color and thickness based on correlation
    if len(G.edges()) > 0:
        edge_weights = [abs(G[u][v]['weight']) for u, v in G.edges()]
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
        
        for edge in G.edges():
            u, v = edge
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            weight = G[u][v]['weight']
            
            # Normalize absolute weight
            abs_weight = abs(weight)
            norm_weight = (abs_weight - min_weight) / weight_range if weight_range > 0 else 0.5
            
            linewidth = 0.5 + 4.5 * norm_weight
            alpha = 0.2 + 0.7 * norm_weight
            
            # Color by sign
            if weight > 0:
                color = plt.cm.Reds(0.3 + 0.7 * norm_weight)
            else:
                color = plt.cm.Blues(0.3 + 0.7 * norm_weight)
            
            ax.plot(x, y, '-', color=color, alpha=alpha, linewidth=linewidth, zorder=1)
    
    # Draw nodes colored by coreness
    node_colors = [coreness_dict[node] for node in G.nodes()]
    node_sizes = [
        500 + 1500 * (coreness_dict[node] - min_core) / (max_core - min_core + 1e-10)
        for node in G.nodes()
    ]
    
    nodes = ax.scatter(
        [pos[node][0] for node in G.nodes()],
        [pos[node][1] for node in G.nodes()],
        c=node_colors,
        s=node_sizes,
        cmap='RdYlGn',
        alpha=0.8,
        edgecolors='black',
        linewidth=1.5,
        vmin=min_core,
        vmax=max_core,
        zorder=2
    )
    
    # Label nodes
    for node, (x, y) in pos.items():
        ax.text(x, y, node, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(nodes, ax=ax)
    cbar.set_label('Coreness Score', fontsize=12)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color=plt.cm.Reds(0.8), lw=3, label='Positive Correlation'),
        Line2D([0], [0], color=plt.cm.Blues(0.8), lw=3, label='Negative Correlation'),
        Line2D([0], [0], color='gray', lw=0.5, label='Weak (thin)'),
        Line2D([0], [0], color='gray', lw=4, label='Strong (thick)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)
    
    ax.set_title('Stock Correlation Network\nRed: Positive | Blue: Negative | Core in Center, Periphery Outside',
                 fontsize=16, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Visualization saved: {filename}")
    plt.close()


def analyze_core_periphery(
    ticker_list: List[str],
    price_history_start_date: Optional[str] = None,
    price_history_end_date: Optional[str] = None,
    visualize_filename: Optional[str] = None,
    cache_file: str = 'india_stocks_history.csv'
) -> pd.DataFrame:
    """
    Main analysis function: identifies core-periphery structure in stock correlations.
    
    This is the primary function for backtesting and analysis. It:
    1. Fetches/caches price data
    2. Prepares data for the specified date range
    3. Builds correlation-based adjacency matrix
    4. Runs core-periphery algorithm
    5. Optionally visualizes the network
    
    Args:
        ticker_list: List of stock tickers to analyze
        price_history_start_date: Start date (YYYY-MM-DD) or None for earliest
        price_history_end_date: End date (YYYY-MM-DD) or None for latest
        visualize_filename: PNG filename for visualization or None to skip
        cache_file: CSV file for caching price data
    
    Returns:
        DataFrame with columns ['Stock', 'Coreness'] sorted by coreness score.
        Higher coreness = more central/core. Lower coreness = more peripheral.
    """
    print("\n" + "="*60)
    print("CORE-PERIPHERY ANALYSIS")
    print("="*60)
    
    # Fetch and cache price data
    print(f"\nFetching data for {len(ticker_list)} tickers...")
    price_data = fetch_and_cache_stock_data(ticker_list, cache_file)
    
    # Prepare data for analysis period
    print("\nPreparing price data...")
    price_data, log_returns = prepare_price_data(
        price_data,
        start_date=price_history_start_date,
        end_date=price_history_end_date
    )
    
    # Build correlation matrix
    print("Building correlation network...")
    A, ticker_names = build_adjacency_matrix(log_returns)
    
    # Identify core-periphery structure
    print("Computing core-periphery decomposition...")
    results = rossa_core_periphery(A, ticker_names)
    
    # Print results
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    print("\nTOP 5 PERIPHERAL STOCKS (Coreness Ascending):")
    print(results.head(5).to_string(index=False))
    print("\nTOP 5 CORE STOCKS (Coreness Descending):")
    print(results.tail(5).to_string(index=False))
    
    # Optional visualization
    if visualize_filename:
        print(f"\nGenerating visualization...")
        plot_network(A, ticker_names, results, visualize_filename)
    
    print("\n" + "="*60 + "\n")
    return results


def main() -> None:
    """
    Main entry point. Reads tickers from file and runs analysis.
    """
    # Load tickers from file
    if not os.path.exists('tickers.txt'):
        print("Error: tickers.txt not found")
        sys.exit(1)
    
    with open('tickers.txt', 'r') as f:
        ticker_list = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(ticker_list)} tickers from tickers.txt")
    
    # Run analysis with visualization
    results = analyze_core_periphery(
        ticker_list=ticker_list,
        price_history_start_date=None,  # Use all available data
        price_history_end_date=None,
        visualize_filename='stock_network.png',
        cache_file='india_stocks_history.csv'
    )


if __name__ == "__main__":
    main()
