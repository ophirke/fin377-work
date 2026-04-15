"""
Data fetching and caching module for stock historical prices.

Provides cacheable functions to fetch stock data from multiple sources
(factorstoday.com API, yfinance) with intelligent caching to avoid redundant downloads.
"""

import pandas as pd
import os
import requests
import yfinance as yf
import logging
from functools import lru_cache
from typing import Tuple

# LOGGING CONFIGURATION
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for verbose output
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def fetch_and_cache_stock_data_factorstoday(
    ticker_tuple: Tuple[str, ...],
    cache_file: str = 'india_stocks_history.csv',
    api_base_url: str = 'https://www.factorstoday.com/api'
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
        logger.info(f"Loading existing cache: {cache_file}...")
        cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Only use cache if it has data rows; empty cache is corrupted
        if len(cached_data) > 0:
            cached_tickers = set(cached_data.columns)
        else:
            logger.info("Cache is empty, will re-download all tickers")
            cached_data = None
    
    # Identify missing tickers
    missing_tickers = [t for t in ticker_list if t not in cached_tickers]
    
    if missing_tickers:
        logger.info(f"Downloading {len(missing_tickers)}/{len(ticker_list)} missing tickers from factorstoday.com API...")
        new_data_dict = {}
        
        for ticker in missing_tickers:
            try:
                # Fetch ALL available data from factorstoday.com API by using a very large 'days' parameter
                url = f"{api_base_url}/stock-history/{ticker}?days=1000000000"
                logger.debug(f"Fetching {ticker}...", end=" ")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    df = df[['close']].rename(columns={'close': ticker})
                    new_data_dict[ticker] = df[ticker]
                    date_range = f"{df.index[0].date()} to {df.index[-1].date()}"
                    logger.debug(f"✓ ({len(data)} records, {date_range})")
                else:
                    logger.debug(f"✗ (no data)")
                    new_data_dict[ticker] = pd.Series(dtype=float)
            except Exception as e:
                logger.debug(f"✗ (error: {e})")
                new_data_dict[ticker] = pd.Series(dtype=float)
        
        # Combine all new data into single DataFrame
        if new_data_dict:
            new_data = pd.DataFrame(new_data_dict)
        else:
            new_data = pd.DataFrame()
        
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
        logger.info(f"Cache updated: {cache_file}")
        logger.info(f"Date range: {all_data.index[0].date()} to {all_data.index[-1].date()}")
    else:
        logger.warning("No data to save to cache (all downloads failed)")
    
    # Before returning, drop columns that are entirely NaN (failed downloads)
    # These stay in cache for next run, but we don't analyze them
    all_data = all_data.dropna(axis=1, how='all')
    
    if len(all_data) > 0:
        logger.info(f"Returning data for {len(all_data.columns)} tickers, date range: {all_data.index[0].date()} to {all_data.index[-1].date()}")
    
    return all_data


@lru_cache(maxsize=32)
def fetch_and_cache_stock_data_yfinance(
    ticker_tuple: Tuple[str, ...],
    cache_file: str = 'india_stocks_history.csv'
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
        logger.info(f"Loading existing cache: {cache_file}...")
        cached_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        # Only use cache if it has data rows; empty cache is corrupted
        if len(cached_data) > 0:
            cached_tickers = set(cached_data.columns)
        else:
            logger.info("Cache is empty, will re-download all tickers")
            cached_data = None
    
    # Identify missing tickers
    missing_tickers = [t for t in ticker_list if t not in cached_tickers]
    
    if missing_tickers:
        logger.info(f"Downloading {len(missing_tickers)}/{len(ticker_list)} missing tickers using yfinance (all available history)...")
        downloaded = yf.download(missing_tickers, period='max', threads=False)
        
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
        logger.info(f"Cache updated: {cache_file}")
    else:
        logger.warning("No data to save to cache (all downloads failed)")
    
    # Before returning, drop columns that are entirely NaN (failed downloads)
    # These stay in cache for next run, but we don't analyze them
    all_data = all_data.dropna(axis=1, how='all')
    
    if len(all_data) > 0:
        logger.info(f"Returning data for {len(all_data.columns)} tickers, date range: {all_data.index[0].date()} to {all_data.index[-1].date()}")
    
    return all_data


@lru_cache(maxsize=32)
def fetch_and_cache_stock_data(
    ticker_tuple: Tuple[str, ...],
    cache_file: str = 'india_stocks_history.csv'
) -> pd.DataFrame:
    """
    Fetches ALL historical stock data, prioritizing factorstoday.com API.
    
    Attempts to fetch from factorstoday.com API first (more reliable).
    If API is unavailable, falls back to yfinance.
    
    Downloads maximum available data for all tickers (no period limit).
    Manages cache intelligently: verifies which tickers are in cache, 
    fetches missing ones, and updates cache. Failed/delisted tickers 
    are saved with NaN values to avoid re-downloading.
    
    Uses lru_cache for memoization - identical ticker sets return cached results.
    
    Args:
        ticker_tuple: Tuple of ticker symbols to fetch (hashable for caching)
        cache_file: Path to cache CSV file
    
    Returns:
        DataFrame with Close prices for all requested tickers (excluding failed tickers)
    """
    try:
        logger.debug("Attempting to fetch from factorstoday.com API...")
        return fetch_and_cache_stock_data_factorstoday(ticker_tuple, cache_file)
    except Exception as e:
        logger.debug(f"⚠ factorstoday.com API failed ({e}), falling back to yfinance...")
        return fetch_and_cache_stock_data_yfinance(ticker_tuple, cache_file)
