"""
Diagnostic script to identify data quality issues in the backtest.
"""
import pandas as pd
import numpy as np
import os

# Note: Run this as: uv run -m tmp.diagnose_price_data from the project root
from data import fetch_and_cache_stock_data
from datamarshal import load_sp500_constituents

# Get a sample of tickers from a recent date
constituents_2023 = load_sp500_constituents("2023-01-15")
sample_tickers = constituents_2023[:100] if len(constituents_2023) > 100 else constituents_2023

print(f"Fetching {len(sample_tickers)} tickers...")
price_data = fetch_and_cache_stock_data(tuple(sample_tickers))

print("\nLoading full cached price data for analysis...")
cache_file = "data/stock_price_history.csv"
if os.path.exists(cache_file):
    print(f"Found cache file: {cache_file}")
    cached_prices = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    print(f"Cached data shape: {cached_prices.shape}")
    price_data = cached_prices
else:
    print(f"Cache file not found at {cache_file}, using fetched sample data")

print("\n" + "="*70)
print("DATA QUALITY DIAGNOSTICS")
print("="*70)

# 1. Check for duplicate columns
if price_data.columns.has_duplicates:
    print(f"\n❌ DUPLICATE COLUMNS FOUND: {price_data.columns[price_data.columns.duplicated()].unique().tolist()}")
else:
    print(f"\n✓ No duplicate columns")

# 2. Check for NaN patterns
print(f"\n--- NaN Analysis ---")
nan_counts = price_data.isna().sum()
print(f"Total NaN values: {nan_counts.sum()} out of {price_data.size}")
print(f"Columns with most NaNs: {nan_counts.nlargest(5).to_dict()}")

# 3. Check for zero-variance stocks
print(f"\n--- Zero Variance Analysis ---")
variances = price_data.var()
zero_var = variances[variances == 0].index.tolist()
print(f"Zero-variance stocks: {len(zero_var)}")
if zero_var:
    print(f"  Examples: {zero_var[:5]}")

# 4. Check for extremely low prices (penny stocks)
print(f"\n--- Penny Stock Analysis ---")
mean_prices = price_data.mean()
penny_stocks = mean_prices[mean_prices < 1.0].index.tolist()
print(f"Stocks with mean price < $1.00: {len(penny_stocks)}")
if penny_stocks:
    print(f"  Examples: {penny_stocks[:5]}")
    print(f"  Mean prices: {mean_prices[penny_stocks[:5]].to_dict()}")

# 5. Check for extreme price movements (potential reverse splits)
print(f"\n--- Extreme Price Movement Analysis ---")
pct_changes = price_data.pct_change().abs()
extreme_moves = (pct_changes > 0.5).sum()  # > 50% moves
print(f"Extreme moves (>50% in one day):")
for ticker in price_data.columns:
    moves = (pct_changes[ticker] > 0.5).sum()
    if moves > 0:
        print(f"  {ticker}: {moves} extreme moves")
        # Find the dates
        extreme_dates = pct_changes[pct_changes[ticker] > 0.5].index.tolist()[:3]
        for date in extreme_dates:
            print(f"    On {date}: {pct_changes.loc[date, ticker]:.1%}")

# 6. Check for NaN in returns (causes NaN correlation)
print(f"\n--- Log Returns Analysis ---")
log_returns = np.log(price_data / price_data.shift(1))
nan_in_returns = log_returns.isna().sum()
print(f"NaN in log returns: {nan_in_returns.sum()} out of {log_returns.size}")

# 7. Check correlation matrix for NaNs
print(f"\n--- Correlation Matrix Analysis ---")
rho = log_returns.corr()
nan_in_corr = rho.isna().sum()
print(f"NaN correlations: {nan_in_corr.sum()} out of {rho.size}")
if nan_in_corr.sum() > 0:
    print(f"Stocks with NaN correlations:")
    for ticker in rho.columns:
        nan_count = rho[ticker].isna().sum()
        if nan_count > 0:
            print(f"  {ticker}: {nan_count}")

# 8. Data summary
print(f"\n--- Data Summary ---")
print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
print(f"Number of tickers: {len(price_data.columns)}")
print(f"Number of dates: {len(price_data.index)}")
print(f"Data shape: {price_data.shape}")
print(f"Data type: {price_data.dtypes.unique()}")
print(f"\nPrice statistics:")
print(price_data.describe().to_string())

print("\n" + "="*70)
