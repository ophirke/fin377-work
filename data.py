"""
Data fetching and caching module for stock historical prices.

Provides cacheable functions to fetch stock data from multiple sources
(factorstoday.com API, yfinance) with intelligent caching to avoid redundant downloads.
"""

import logging
import multiprocessing
import os
import threading
from functools import cache, lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from datamarshal import DataConfig

# LOGGING CONFIGURATION
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for verbose output
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Global placeholder for cache lock (will be initialized by pool initializer)
_cache_lock = None
_all_tickers_seen = set()

def my_print_info(message: str):
    print(f"[PID {os.getpid()}] {message}")


def init_worker_lock(shared_lock):
    """Initializes the shared lock for each worker process in the Pool."""
    global _cache_lock
    _cache_lock = shared_lock


def _fetch_and_cache_stock_data_factorstoday(
    ticker_tuple: Tuple[str, ...],
    cache_file: Path = DataConfig.CACHE_FILE,
    api_base_url: str = "https://www.factorstoday.com/api",
) -> pd.DataFrame:
    """
    Fetches ALL historical stock data from factorstoday.com API, manages cache intelligently.

    Fetches maximum available data (no period limit). Verifies which tickers are in cache,
    fetches missing ones, and updates cache. Failed/delisted tickers are saved to cache
    with NaN values to avoid re-downloading. This ensures complete data without redundant downloads.

    Uses lru_cache for memoization - identical ticker sets return cached results.

    Args:
        ticker_tuple: Tuple of ticker symbols to fetch (hashable for caching)
        cache_file: Path to cache CSV file
        api_base_url: Base URL for the API (default: https://www.factorstoday.com/api)

    Returns:
        DataFrame with Close prices for all requested tickers (excluding failed tickers)
    """
    ticker_list = list(ticker_tuple)

    cached_tickers = set()
    cached_data = None

    # Load existing cache if available (and non-empty)
    if os.path.exists(cache_file):
        my_print_info(f"Loading existing cache: {cache_file}...")
        cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Only use cache if it has data rows; empty cache is corrupted
        if len(cached_data) > 0:
            cached_tickers = set(cached_data.columns)
        else:
            my_print_info("Cache is empty, will re-download all tickers")
            cached_data = None

    # Identify missing tickers
    missing_tickers = [t for t in ticker_list if t not in cached_tickers]

    if missing_tickers:
        my_print_info(
            f"Downloading {len(missing_tickers)}/{len(ticker_list)} missing tickers from factorstoday.com API..."
        )
        new_data_dict = {}

        for ticker_idx, ticker in enumerate(missing_tickers, 1):
            try:
                # Fetch ALL available data from factorstoday.com API by using a very large 'days' parameter
                url = f"{api_base_url}/stock-history/{ticker}?days=1000000000"
                my_print_info(f"Fetching {ticker}...")
                response = requests.get(url, timeout=10)
                response.raise_for_status()

                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                    df = df[["close"]].rename(columns={"close": ticker})
                    new_data_dict[ticker] = df[ticker]
                    date_range = f"{df.index[0].date()} to {df.index[-1].date()}"
                    my_print_info(f"✓ ({len(data)} records, {date_range})")
                else:
                    my_print_info(f"✗ (no data)")
                    new_data_dict[ticker] = pd.Series(dtype=float)
            except Exception as e:
                my_print_info(f"✗ (error: {e})")
                new_data_dict[ticker] = pd.Series(dtype=float)

            # Checkpoint every 100 tickers
            if ticker_idx % 100 == 0 or ticker_idx == len(missing_tickers):
                if new_data_dict:
                    new_data = pd.DataFrame(new_data_dict)
                    # Merge with cached data
                    if cached_data is not None and len(cached_data) > 0:
                        checkpoint_data = pd.concat([cached_data, new_data], axis=1)
                    else:
                        checkpoint_data = new_data
                    # Save checkpoint
                    checkpoint_data.to_csv(cache_file)
                    my_print_info(f"Checkpoint saved: {ticker_idx}/{len(missing_tickers)} tickers downloaded")
                    # Update cached_data for next checkpoint iteration
                    cached_data = checkpoint_data
                    new_data_dict = {}

        # After loop, combine all final data
        if new_data_dict:
            new_data = pd.DataFrame(new_data_dict)
            if cached_data is not None and len(cached_data) > 0:
                all_data = pd.concat([cached_data, new_data], axis=1)
            else:
                all_data = new_data
        else:
            all_data = cached_data if cached_data is not None else pd.DataFrame()
    else:
        if cached_data is None or len(cached_data) == 0:
            raise ValueError(f"No cache found and no tickers to download")
        all_data = cached_data

    # IMPORTANT: All data is already saved via checkpoints during the download loop
    # This final message confirms completion
    if missing_tickers:
        if len(all_data) > 0:
            my_print_info(
                f"Download complete. Date range: {all_data.index[0].date()} to {all_data.index[-1].date()}"
            )
        else:
            logger.warning("No data to save to cache (all downloads failed)")
    elif len(all_data) > 0:
        my_print_info(
            f"Using existing cache with {len(all_data.columns)} tickers, date range: {all_data.index[0].date()} to {all_data.index[-1].date()}"
        )

    # Keep only the requested tickers for return (but cache has all)
    available_tickers = [t for t in ticker_list if t in all_data.columns]
    return_data = all_data[available_tickers]

    # Drop columns that are entirely NaN (failed downloads) before returning
    return_data = return_data.dropna(axis=1, how="all")

    if len(return_data) > 0:
        my_print_info(
            f"Returning data for {len(return_data.columns)} tickers, date range: {return_data.index[0].date()} to {return_data.index[-1].date()}"
        )

    return return_data


def _fetch_and_cache_stock_data_yfinance(
    ticker_tuple: Tuple[str, ...], cache_file: Path = DataConfig.CACHE_FILE
) -> pd.DataFrame:
    """
    Fetches historical stock data using yfinance, manages cache intelligently.

    DEPRECATED: Use fetch_and_cache_stock_data instead, which prioritizes factorstoday.com API.
    This function is kept as a fallback data source.
    Downloads maximum available data (all history).

    Verifies which tickers are in cache, fetches missing ones, and updates cache.
    Failed/delisted tickers are saved to cache with NaN values to avoid re-downloading.
    This ensures complete data without redundant downloads.

    Uses lru_cache for memoization - identical ticker sets return cached results.

    Args:
        ticker_tuple: Tuple of ticker symbols to fetch (hashable for caching)
        cache_file: Path to cache CSV file

    Returns:
        DataFrame with Close prices for all requested tickers (excluding failed tickers)
    """
    ticker_list = list(ticker_tuple)

    cached_tickers = set()
    cached_data = None

    # Load existing cache if available (and non-empty)
    if os.path.exists(cache_file):
        my_print_info(f"Loading existing cache: {cache_file}...")
        cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Only use cache if it has data rows; empty cache is corrupted
        if len(cached_data) > 0:
            cached_tickers = set(cached_data.columns)
        else:
            my_print_info("Cache is empty, will re-download all tickers")
            cached_data = None

    # Identify missing tickers
    missing_tickers = [t for t in ticker_list if t not in cached_tickers]

    if missing_tickers:
        my_print_info(
            f"Downloading {len(missing_tickers)}/{len(ticker_list)} missing tickers using yfinance (all available history)..."
        )
        
        # Download in batches of 250 for checkpointing
        batch_size = 250
        all_new_data = {}
        
        for batch_start in range(0, len(missing_tickers), batch_size):
            batch_end = min(batch_start + batch_size, len(missing_tickers))
            batch_tickers = missing_tickers[batch_start:batch_end]
            
            my_print_info(f"Downloading batch {batch_start//batch_size + 1} ({batch_start+1}-{batch_end} of {len(missing_tickers)})...")
            downloaded = yf.download(batch_tickers, period="max", threads=False)
            
            # Extract 'Close' from MultiIndex columns (when multiple tickers)
            if isinstance(downloaded.columns, pd.MultiIndex):
                batch_data = downloaded.xs("Close", level=0, axis=1)
            else:
                batch_data = downloaded[["Close"]]
            
            all_new_data.update(batch_data.to_dict('series'))
            
            # Checkpoint every batch
            if len(all_new_data) > 0:
                new_data = pd.DataFrame(all_new_data)
                # Merge with cached data
                if cached_data is not None and len(cached_data) > 0:
                    checkpoint_data = pd.concat([cached_data, new_data], axis=1)
                else:
                    checkpoint_data = new_data
                # Save checkpoint
                checkpoint_data.to_csv(cache_file)
                my_print_info(f"Checkpoint saved: {batch_end}/{len(missing_tickers)} tickers downloaded")
                # Update cached_data for next batch
                cached_data = checkpoint_data

        # Final merged data
        if all_new_data:
            new_data = pd.DataFrame(all_new_data)
            if cached_data is not None and len(cached_data) > 0:
                all_data = pd.concat([cached_data, new_data], axis=1)
            else:
                all_data = new_data
        else:
            all_data = cached_data if cached_data is not None else pd.DataFrame()
    else:
        if cached_data is None or len(cached_data) == 0:
            raise ValueError(f"No cache found and no tickers to download")
        all_data = cached_data

    # IMPORTANT: Save full merged data to cache (keep all tickers, not just requested)
    # Only write if something changed (new data was downloaded)
    if missing_tickers and len(all_data) > 0:
        all_data.to_csv(cache_file)
        my_print_info(f"Cache updated: {cache_file}")
    elif missing_tickers and len(all_data) == 0:
        logger.warning("No data to save to cache (all downloads failed)")

    # Keep only the requested tickers for return (but cache has all)
    available_tickers = [t for t in ticker_list if t in all_data.columns]
    return_data = all_data[available_tickers]

    # Drop columns that are entirely NaN (failed downloads) before returning
    return_data = return_data.dropna(axis=1, how="all")

    if len(return_data) > 0:
        my_print_info(
            f"Returning data for {len(return_data.columns)} tickers, date range: {return_data.index[0].date()} to {return_data.index[-1].date()}"
        )

    return return_data


def _filter_fetched_data(return_data, ticker_list):
    """
    Filters the fetched data to include only the requested tickers, while keeping failed tickers with NaN values.

    This function ensures that we return data for all requested tickers, including those that failed to download (marked with NaN).
    It only drops columns that are completely empty (shouldn't happen), but keeps all-NaN columns from failed downloads to mark them as "attempted but failed".

    Args:
        return_data: DataFrame containing the merged cached and newly fetched data
        ticker_list: List of requested ticker symbols

    Returns:
        Filtered DataFrame with Close prices for all requested tickers (including failed tickers with NaN)
    """
    # Keep only the requested tickers for return (but cache has all)
    available_tickers = [t for t in ticker_list if t in return_data.columns]
    filtered_data = return_data[available_tickers]

    # Drop columns that are entirely NaN (failed downloads) before returning
    filtered_data = filtered_data.dropna(axis=1, how="all")

    if len(filtered_data) > 0:
        logger.info(
            f"Filtered data to {len(filtered_data.columns)} tickers, date range: {filtered_data.index[0].date()} to {filtered_data.index[-1].date()}"
        )

    return filtered_data


def clean_price_data(
    price_data: pd.DataFrame,
    min_price: float = 1.0,
    max_price: float = 10000.0,
) -> pd.DataFrame:
    """
    Filter a bounded analysis window by removing tickers with unrealistic prices.

    Removes any ticker whose observed price history ever drops below
    ``min_price`` or rises above ``max_price``. This catches penny stocks and
    data glitches from reverse splits or bad prints that can cause portfolio
    explosions in the backtest.

    This is intended for eligibility filtering inside a local analysis window
    such as a rebalance lookback period. It should not be used to globally
    delete tickers from realized PnL paths, because doing so can introduce
    lookahead bias.

    Args:
        price_data: DataFrame with dates as index and tickers as columns (price values)
        min_price: Minimum acceptable observed price (default $1.00)
        max_price: Maximum acceptable observed price (default $10,000)

    Returns:
        Cleaned DataFrame with unrealistic tickers removed
    """
    original_count = len(price_data.columns)

    # Filter tickers by their full observed price range, not the mean.
    min_prices = price_data.min(skipna=True)
    max_prices = price_data.max(skipna=True)

    below_min_mask = min_prices < min_price
    above_max_mask = max_prices > max_price
    missing_mask = min_prices.isna() | max_prices.isna()

    valid_mask = ~(below_min_mask | above_max_mask | missing_mask)
    valid_tickers = price_data.columns[valid_mask]

    # Extract cleaned data
    cleaned_data = price_data[valid_tickers]

    # Log results
    removed_count = original_count - len(valid_tickers)
    if removed_count > 0:
        penny_stocks = min_prices[below_min_mask]
        expensive_stocks = max_prices[above_max_mask]
        missing_tickers = price_data.columns[missing_mask]

        logger.info(f"Price data cleaning:")
        logger.info(f"  Original tickers: {original_count}")
        logger.info(
            f"  Removed tickers: {removed_count} ({100*removed_count/original_count:.1f}%)"
        )
        logger.info(f"    - Penny stocks (< ${min_price}): {len(penny_stocks)}")
        logger.info(f"    - Expensive stocks (> ${max_price}): {len(expensive_stocks)}")
        logger.info(f"    - Missing/invalid price history: {len(missing_tickers)}")
        logger.info(f"  Remaining tickers: {len(valid_tickers)}")

    return cleaned_data


def mask_invalid_price_cells(
    price_data: pd.DataFrame,
    min_price: float = 1.0,
    max_price: float = 10000.0,
) -> pd.DataFrame:
    """
    Replace invalid individual price observations with NaN.

    This preserves the ticker in the panel while removing bad prints from the
    realized valuation path. Downstream consumers can then forward-fill the last
    valid observation without retroactively dropping the security.

    Args:
        price_data: DataFrame with dates as index and tickers as columns
        min_price: Minimum acceptable observed price (default $1.00)
        max_price: Maximum acceptable observed price (default $10,000)

    Returns:
        DataFrame with invalid cells masked to NaN
    """
    cleaned = price_data.copy()
    numeric = cleaned.apply(pd.to_numeric, errors="coerce")
    invalid_mask = (~np.isfinite(numeric)) | (numeric < min_price) | (numeric > max_price)
    return numeric.mask(invalid_mask)


@lru_cache(maxsize=1)
def _fetch_and_cache_stock_data_inner(
    ticker_tuple: Tuple[str, ...],
) -> pd.DataFrame:
    """
    Fetches ALL historical stock data, prioritizing factorstoday.com API.

    Attempts to fetch from factorstoday.com API first (more reliable).
    If API is unavailable, falls back to yfinance.

    Downloads maximum available data for all tickers (no period limit).
    Manages cache intelligently: verifies which tickers are in cache,
    fetches missing ones, and updates cache. Failed/delisted tickers
    are saved with NaN values to avoid re-downloading.

    THREAD-SAFE: Wrapped with multiprocessing-managed lock via Pool initializer.

    Args:
        ticker_tuple: Tuple of ticker symbols to fetch (hashable for caching)
        cache_file: Path to cache CSV file

    Returns:
        DataFrame with Close prices for all requested tickers (excluding failed tickers)
    """
    global _cache_lock

    # Fallback to a threading lock if run outside a pool (single process mode)
    lock_to_use = _cache_lock if _cache_lock is not None else threading.Lock()
    print(f"[PID {os.getpid()}] using lock: {lock_to_use}")

    print(f"[PID {os.getpid()}] WAITING FOR LOCK...")
    with lock_to_use:
        print(f"[PID {os.getpid()}] LOCK ACQUIRED")
        try:
            logger.debug("Attempting to fetch from factorstoday.com API...")
            res = _fetch_and_cache_stock_data_factorstoday(ticker_tuple)
        except Exception as e:
            logger.debug(
                f"⚠ factorstoday.com API failed ({e}), falling back to yfinance..."
            )
            res = _fetch_and_cache_stock_data_yfinance(ticker_tuple)
        print(f"[PID {os.getpid()}] RELEASING LOCK")

    return res



@cache
def fetch_and_cache_stock_data(
    ticker_tuple: Tuple[str, ...],
) -> pd.DataFrame:
    """
    Fetches ALL historical stock data, prioritizing factorstoday.com API.

    Attempts to fetch from factorstoday.com API first (more reliable).
    If API is unavailable, falls back to yfinance.

    Downloads maximum available data for all tickers (no period limit).
    Manages cache intelligently: verifies which tickers are in cache,
    fetches missing ones, and updates cache. Failed/delisted tickers
    are saved with NaN values to avoid re-downloading.

    THREAD-SAFE: Wrapped with multiprocessing-managed lock via Pool initializer.

    Args:
        ticker_tuple: Tuple of ticker symbols to fetch (hashable for caching)
        cache_file: Path to cache CSV file

    Returns:
        DataFrame with Close prices for all requested tickers (excluding failed tickers)
    """
    global _all_tickers_seen
    
    for ticker in ticker_tuple:
        if ticker not in _all_tickers_seen:
            logger.info(f"New ticker requested: {ticker}")
            _all_tickers_seen.add(ticker)
    
    fetch_key = tuple(sorted(_all_tickers_seen))
    
    res = _fetch_and_cache_stock_data_inner(fetch_key)
    
    filtered_res = _filter_fetched_data(res, ticker_tuple)
    
    return filtered_res
