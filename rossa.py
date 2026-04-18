import logging
import os
import sys
from datetime import datetime
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from datamarshal import load_sp100_constituents, load_nasdaq100_constituents, load_sp500_constituents

from datamarshal import DataConfig
from data import fetch_and_cache_stock_data, clean_price_data

# LOGGING CONFIGURATION
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for verbose output
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def prepare_price_data(
    price_data: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    missing_tolerance: float = 0.20,
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
        price_data = price_data[price_data.index >= pd.Timestamp(start_date)]
    if end_date:
        price_data = price_data[price_data.index <= pd.Timestamp(end_date)]

    initial_tickers = price_data.shape[1]

    # Drop stocks missing more than missing_tolerance of data history
    threshold = int(len(price_data) * (1.0 - missing_tolerance))
    price_data = price_data.dropna(axis=1, thresh=threshold)

    dropped_tickers = initial_tickers - price_data.shape[1]
    if dropped_tickers > 0:
        logger.info(
            f"  Dropped {dropped_tickers} stocks with >{missing_tolerance * 100:.0f}% missing data"
        )

    # Fill small gaps (trading halts, holidays) forward, then backward
    price_data = price_data.ffill().bfill()

    # Drop any remaining rows with NaNs
    price_data = price_data.dropna()

    if price_data.empty or price_data.shape[1] < 2:
        logger.error(
            f"  ERROR: Only {price_data.shape[1]} stocks remain with overlapping data"
        )
        raise ValueError("Not enough overlapping historical data to build a network")

    # Calculate log returns
    log_returns = np.log(price_data / price_data.shift(1)).dropna()

    logger.info(
        f"  → Analyzing {log_returns.shape[1]} stocks over {log_returns.shape[0]} trading days"
    )
    logger.info(
        f"  → Date range: {price_data.index[0].date()} to {price_data.index[-1].date()}"
    )

    return price_data, log_returns


def load_analysis_price_data(
    ticker_list: List[str],
    price_history_start_date: Optional[str] = None,
    price_history_end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the cleaned price window used for coreness analysis.

    This applies date slicing before cleaning so future glitches cannot affect
    earlier rebalance windows.
    """
    logger.info(f"\nFetching data for {len(ticker_list)} tickers...")
    price_data = fetch_and_cache_stock_data(tuple(ticker_list))

    if price_history_start_date:
        price_data = price_data[price_data.index >= pd.Timestamp(price_history_start_date)]
    if price_history_end_date:
        price_data = price_data[price_data.index <= pd.Timestamp(price_history_end_date)]

    if price_data.empty:
        raise ValueError("No price data available in the requested analysis window")

    logger.info("\nCleaning price data (removing penny stocks and data glitches)...")
    price_data = clean_price_data(price_data, min_price=1.0, max_price=10000.0)

    logger.info("\nPreparing price data...")
    return prepare_price_data(price_data)


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

    # CRITICAL FIX: Treat NaN correlations (zero variance stocks) as 0 correlation (A = 0.5)
    # NOT as perfect negative correlation (A = 0.0), which was hunting down dead penny stocks
    A = np.nan_to_num(A, nan=0.5)

    # Remove self-loops AFTER handling NaNs
    np.fill_diagonal(A, 0)

    return A, tickers


# def rossa_core_periphery(A: np.ndarray, tickers: pd.Index) -> pd.DataFrame:
#     """
#     Implements core-periphery profile algorithm by Rossa et al.

#     Builds a core set S by iteratively adding vertices that maximize
#     internal connectivity (phi score). Returns tickers ranked by coreness.

#     Args:
#         A: Adjacency matrix
#         tickers: Stock ticker names

#     Returns:
#         DataFrame with columns ['Stock', 'Coreness'] sorted by coreness
#     """
#     N = A.shape[0]
#     degrees = A.sum(axis=1)  # Weighted degree

#     # Start with peripheral vertex (minimum degree)
#     first_vertex = np.argmin(degrees)
#     S = [first_vertex]
#     unvisited = set(range(N))
#     unvisited.remove(first_vertex)

#     phi_list = [0.0]
#     order = [first_vertex]

#     current_internal_sum = 0.0
#     current_degree_sum = degrees[first_vertex]

#     while unvisited:
#         min_phi = float("inf")
#         best_v = None
#         best_internal_sum = 0.0

#         for j in unvisited:
#             edges_to_S = 2 * A[j, S].sum()
#             test_internal = current_internal_sum + edges_to_S
#             test_degree = current_degree_sum + degrees[j]

#             # Phi score: internal density of core set
#             phi_S = test_internal / test_degree if test_degree > 0 else float("inf")

#             if phi_S < min_phi:
#                 min_phi = phi_S
#                 best_v = j
#                 best_internal_sum = test_internal

#         # Failsafe for disconnected graphs
#         if best_v is None:
#             best_v = next(iter(unvisited))
#             min_phi = 1.0

#         S.append(best_v)
#         unvisited.remove(best_v)
#         order.append(best_v)
#         phi_list.append(min_phi)

#         current_internal_sum = best_internal_sum
#         current_degree_sum += degrees[best_v]

#     return pd.DataFrame({"Stock": [tickers[i] for i in order], "Coreness": phi_list})

def rossa_core_periphery(A: np.ndarray, tickers: pd.Index) -> pd.DataFrame:
    """
    Optimized implementation of the core-periphery profile algorithm.
    Reduces time complexity from O(N^3) to O(N^2) via state maintenance and vectorization.
    """
    N = A.shape[0]
    degrees = A.sum(axis=1)  # Weighted degree

    # Start with peripheral vertex (minimum degree)
    first_vertex = np.argmin(degrees)
    
    # Pre-allocate output arrays for massive speedup
    order = np.empty(N, dtype=int)
    phi_list = np.empty(N, dtype=float)
    
    order[0] = first_vertex
    phi_list[0] = 0.0

    current_internal_sum = 0.0
    current_degree_sum = degrees[first_vertex]

    # STATE MAINTENANCE: Track edges from ALL nodes to the set S
    # This avoids recalculating A[j, S].sum() in a loop
    edges_to_S = 2 * A[:, first_vertex]
    
    # Boolean mask for O(1) tracking of unvisited nodes
    unvisited = np.ones(N, dtype=bool)
    unvisited[first_vertex] = False

    for i in range(1, N):
        # Vectorized calculation for all nodes simultaneously
        test_internal = current_internal_sum + edges_to_S
        test_degree = current_degree_sum + degrees

        # Safely calculate phi scores for all nodes at once
        with np.errstate(divide='ignore', invalid='ignore'):
            phi_scores = np.where(test_degree > 0, test_internal / test_degree, np.inf)
            
        # Mask out already visited nodes by setting their score to infinity
        phi_scores[~unvisited] = np.inf

        # Find the best vertex in one shot
        best_v = np.argmin(phi_scores)
        min_phi = phi_scores[best_v]

        # Original failsafe for disconnected graphs
        if np.isinf(min_phi):
            best_v = np.nonzero(unvisited)[0][0]
            min_phi = 1.0

        # Update order and tracking variables
        order[i] = best_v
        phi_list[i] = min_phi
        current_internal_sum = test_internal[best_v]
        current_degree_sum += degrees[best_v]

        # CRITICAL UPDATE: Add the new node's edges to our cache in O(N) time
        edges_to_S += 2 * A[:, best_v]
        unvisited[best_v] = False

    return pd.DataFrame({
        "Stock": tickers[order], 
        "Coreness": phi_list
    })

def plot_network(
    A: np.ndarray,
    ticker_names: pd.Index,
    results: pd.DataFrame,
    filename: str,
    corr_threshold: float = 0.3,
    node_size_map: Optional[dict] = None,
    node_alpha_map: Optional[dict] = None,
    edge_alpha_scale: float = 0.3,
) -> None:
    """
    Visualizes stock correlation network using a force-directed structure.

    - Highly correlated stocks form organic clusters
    - Red edges: positive correlations
    - Blue edges: negative correlations
    - Thickness: correlation strength

    Args:
        A: Adjacency matrix
        ticker_names: Stock ticker names
        results: DataFrame with coreness scores
        filename: Output PNG filename
        corr_threshold: Minimum correlation magnitude to show edge
        node_size_map: Optional per-ticker node sizes
        node_alpha_map: Optional per-ticker node alpha values
        edge_alpha_scale: Multiplier to reduce overall edge opacity
    """
    G = nx.Graph()

    # Add nodes
    for ticker in ticker_names:
        G.add_node(ticker)

    # Get coreness for each stock
    coreness_dict = dict(zip(results["Stock"], results["Coreness"]))

    # Add edges based on correlation threshold
    for i in range(len(ticker_names)):
        for j in range(i + 1, len(ticker_names)):
            correlation = A[i, j] * 2 - 1  # Map [0, 1] back to [-1, 1]
            if abs(correlation) > corr_threshold:
                G.add_edge(ticker_names[i], ticker_names[j], weight=correlation)

    # ====================================================================
    # FORCE-DIRECTED (SPRING) LAYOUT
    # ====================================================================
    # The physics engine works best when edges act as positive "springs".
    # We want positively correlated stocks to attract each other tightly.
    layout_weights = {}
    for u, v, data in G.edges(data=True):
        # Use max(0.01, weight) so positive correlations pull nodes together,
        # but negative correlations don't break the physics calculations.
        layout_weights[(u, v)] = max(0.01, data['weight'])
        
    nx.set_edge_attributes(G, layout_weights, 'layout_weight')

    # Generate the organic layout.
    # k: Cranks up the repulsion force to push nodes apart
    # scale: Spreads the entire graph over a larger coordinate space
    pos = nx.spring_layout(
        G, 
        weight='layout_weight', 
        k=7.8,               # Increased from 0.15 to force separation
        scale=1.0,           # Stretches the bounding box
        iterations=500,      # Gives the engine more time to settle
        seed=42,
    )

    # ====================================================================

    coreness_values = results["Coreness"].values
    min_core = coreness_values.min()
    max_core = coreness_values.max()

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 16))

    # Draw edges with color and thickness based on correlation
    if len(G.edges()) > 0:
        edge_weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        weight_range = max_weight - min_weight if max_weight > min_weight else 1.0

        for edge in G.edges():
            u, v = edge
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            weight = G[u][v]["weight"]

            # Normalize absolute weight
            abs_weight = abs(weight)
            norm_weight = (
                (abs_weight - min_weight) / weight_range if weight_range > 0 else 0.5
            )

            linewidth = 0.5 + 4.5 * norm_weight
            alpha = (0.2 + 0.7 * norm_weight) * edge_alpha_scale

            # Color by sign
            if weight > 0:
                color = plt.cm.Reds(0.3 + 0.7 * norm_weight)
            else:
                color = plt.cm.Blues(0.3 + 0.7 * norm_weight)

            ax.plot(x, y, "-", color=color, alpha=alpha, linewidth=linewidth, zorder=1)

    # Draw nodes colored by coreness
    node_order = list(G.nodes())
    node_colors = np.array([coreness_dict[node] for node in node_order], dtype=float)
    if node_size_map is None:
        node_sizes = np.array(
            [
                200
                + 600
                * (max_core - (coreness_dict[node] - min_core))
                / (max_core - min_core + 1e-10)
                for node in node_order
            ],
            dtype=float,
        )
    else:
        node_sizes = np.array(
            [float(node_size_map.get(node, 250.0)) for node in node_order], dtype=float
        )

    if node_alpha_map is None:
        node_alphas = np.full(len(node_order), 0.8, dtype=float)
    else:
        node_alphas = np.array(
            [float(node_alpha_map.get(node, 0.8)) for node in node_order], dtype=float
        )

    norm = Normalize(vmin=min_core, vmax=max_core)
    facecolors = plt.cm.RdYlGn(norm(node_colors))
    facecolors[:, 3] = node_alphas

    nodes = ax.scatter(
        [pos[node][0] for node in node_order],
        [pos[node][1] for node in node_order],
        s=node_sizes,
        facecolors=facecolors,
        edgecolors="black",
        linewidth=1.2,
        zorder=2,
    )

    # Label nodes
    for node, (x, y) in pos.items():
        ax.text(
            x,
            y,
            node,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            alpha=max(0.45, float(node_alpha_map.get(node, 0.8)) if node_alpha_map else 0.8),
        )

    # Add colorbar
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="RdYlGn"), ax=ax, fraction=0.046, pad=0.04
    )
    cbar.set_label("Coreness Score", fontsize=12)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color=plt.cm.Reds(0.8), lw=3, label="Positive Correlation"),
        Line2D([0], [0], color=plt.cm.Blues(0.8), lw=3, label="Negative Correlation"),
        Line2D([0], [0], color="gray", lw=0.5, label="Weak (thin)"),
        Line2D([0], [0], color="gray", lw=4, label="Strong (thick)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.95)

    ax.set_title(
        "Stock Correlation Network\nForce-Directed Clustering",
        fontsize=16,
        fontweight="bold",
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    logger.info(f"Visualization saved: {filename}")
    plt.close()


def analyze_core_periphery(
    ticker_list: List[str],
    price_history_start_date: Optional[str] = None,
    price_history_end_date: Optional[str] = None,
    visualize_filename: Optional[str] = None,
    corr_threshold: float = 0.55,
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
        corr_threshold: Minimum correlation magnitude to show edge (default 0.55). Higher values reduce clutter and prevent singularity.

    Returns:
        DataFrame with columns ['Stock', 'Coreness'] sorted by coreness score.
        Higher coreness = more central/core. Lower coreness = more peripheral.
    """
    logger.info("\n" + "=" * 60)
    logger.info("CORE-PERIPHERY ANALYSIS")
    logger.info("=" * 60)

    price_data, log_returns = load_analysis_price_data(
        ticker_list=ticker_list,
        price_history_start_date=price_history_start_date,
        price_history_end_date=price_history_end_date,
    )

    # Build correlation matrix
    logger.info("Building correlation network...")
    A, ticker_names = build_adjacency_matrix(log_returns)

    # Identify core-periphery structure
    logger.info("Computing core-periphery decomposition...")
    results = rossa_core_periphery(A, ticker_names)

    # Print results
    logger.info("\n" + "-" * 60)
    logger.info("RESULTS")
    logger.info("-" * 60)
    logger.info("\nTOP 5 PERIPHERAL STOCKS (Coreness Ascending):")
    logger.debug(results.head(5).to_string(index=False))
    logger.info("\nTOP 5 CORE STOCKS (Coreness Descending):")
    logger.debug(results.tail(5).to_string(index=False))

    # Optional visualization
    if visualize_filename:
        logger.info(f"\nGenerating visualization...")
        plot_network(A, ticker_names, results, visualize_filename, corr_threshold=corr_threshold)

    logger.info("\n" + "=" * 60 + "\n")
    return results


def main() -> None:
    """
    Main entry point. Reads tickers from file and runs analysis.
    """
    # Load tickers from file
    if not os.path.exists(DataConfig.TICKER_FILE    ):
        logger.error(f"Error: {DataConfig.TICKER_FILE} not found")
        sys.exit(1)

    with open(DataConfig.TICKER_FILE, "r") as f:
        ticker_list = [line.strip() for line in f if line.strip()]

    logger.info(f"Loaded {len(ticker_list)} tickers from {DataConfig.TICKER_FILE}")
    
    ticker_list = load_nasdaq100_constituents("2018-01-01")

    # Run analysis with visualization (0.55 threshold avoids the "gravity well" singularity)
    results = analyze_core_periphery(
        ticker_list=ticker_list,
        price_history_start_date="2018-01-01",
        price_history_end_date=None,
        visualize_filename=f"{DataConfig.OUTPUT_DIR}/core_periphery_network.png",
        corr_threshold=0.5,
    )


if __name__ == "__main__":
    main()
