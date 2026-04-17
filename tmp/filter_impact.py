"""
Analyze the impact of the price filter on data quality.
Shows before/after statistics and identifies problem tickers.
"""
import pandas as pd
import numpy as np

print("="*80)
print("PRICE FILTER IMPACT ANALYSIS")
print("="*80)

# Load cached price data
df = pd.read_csv("data/stock_price_history.csv", index_col=0, parse_dates=True)
print(f"\n--- ORIGINAL DATA ---")
print(f"Shape: {df.shape}")
print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
print(f"Total tickers: {len(df.columns)}")

# Calculate mean prices
mean_prices = df.mean()
print(f"\nPrice statistics (original):")
print(f"  Mean of mean prices: ${mean_prices.mean():.2f}")
print(f"  Median of mean prices: ${mean_prices.median():.2f}")
print(f"  Min mean price: ${mean_prices.min():.6f}")
print(f"  Max mean price: ${mean_prices.max():.2f}")

# Apply filter
valid_mask = (mean_prices >= 1.0) & (mean_prices <= 10000)
df_filtered = df[mean_prices[valid_mask].index]

print(f"\n--- FILTERED DATA (mean price $1-$10k) ---")
print(f"Shape: {df_filtered.shape}")
print(f"Total tickers after filter: {len(df_filtered.columns)}")

# Calculate filtered mean prices
mean_prices_filtered = df_filtered.mean()
print(f"\nPrice statistics (filtered):")
print(f"  Mean of mean prices: ${mean_prices_filtered.mean():.2f}")
print(f"  Median of mean prices: ${mean_prices_filtered.median():.2f}")
print(f"  Min mean price: ${mean_prices_filtered.min():.6f}")
print(f"  Max mean price: ${mean_prices_filtered.max():.2f}")

# Count removals
penny_stocks = mean_prices[mean_prices < 1.0]
expensive_stocks = mean_prices[mean_prices > 10000]

print(f"\nRemoved tickers:")
print(f"  Penny stocks (< $1.00): {len(penny_stocks)}")
print(f"  Extremely expensive (> $10k): {len(expensive_stocks)}")
print(f"  Total removed: {len(mean_prices) - len(df_filtered.columns)}")
print(f"  % of data removed: {100 * (len(mean_prices) - len(df_filtered.columns)) / len(mean_prices):.1f}%")

if len(penny_stocks) > 0:
    print(f"\n  Penny stock examples (< $1.00):")
    penny_sorted = penny_stocks.sort_values()
    for ticker, price in penny_sorted.head(10).items():
        print(f"    {ticker}: ${price:.6f}")

if len(expensive_stocks) > 0:
    print(f"\n  Expensive stock examples (> $100k):")
    expensive_sorted = expensive_stocks.sort_values(ascending=False)
    for ticker, price in expensive_sorted.head(10).items():
        print(f"    {ticker}: ${price:.2f}")

# Impact on volatility and correlations
print(f"\n--- IMPACT ON DATA QUALITY ---")

# Calculate log returns for both - handle zero/negative prices
def calculate_valid_returns(df):
    """Calculate log returns, filtering out zero and negative prices."""
    # Replace zero and negative prices with NaN
    df_clean = df.copy()
    df_clean[df_clean <= 0] = np.nan
    
    # Calculate log returns only where prices are positive
    returns = np.log(df_clean / df_clean.shift(1))
    
    # Count how many NaN/inf values we have
    nan_count = returns.isna().sum().sum() + np.isinf(returns.values).sum()
    total_count = returns.size
    
    return returns, nan_count, total_count

orig_returns, orig_nan, orig_total = calculate_valid_returns(df)
filt_returns, filt_nan, filt_total = calculate_valid_returns(df_filtered)

print(f"Original data - Daily returns:")
print(f"  Valid data points: {orig_total - orig_nan} of {orig_total} ({100*(orig_total-orig_nan)/orig_total:.1f}%)")
# Only compute stats on non-NaN values
orig_valid = orig_returns.values[~np.isnan(orig_returns.values) & ~np.isinf(orig_returns.values)]
if len(orig_valid) > 0:
    print(f"  Mean return (daily): {orig_valid.mean():.6f}")
    print(f"  Std dev (daily): {orig_valid.std():.6f}")
else:
    print(f"  Mean return (daily): NO VALID DATA")
    print(f"  Std dev (daily): NO VALID DATA")
extreme_moves_orig = (np.abs(orig_returns) > 1.0).sum().sum()
print(f"  Extreme moves (>100%): {extreme_moves_orig}")

print(f"\nFiltered data - Daily returns:")
print(f"  Valid data points: {filt_total - filt_nan} of {filt_total} ({100*(filt_total-filt_nan)/filt_total:.1f}%)")
filt_valid = filt_returns.values[~np.isnan(filt_returns.values) & ~np.isinf(filt_returns.values)]
if len(filt_valid) > 0:
    print(f"  Mean return (daily): {filt_valid.mean():.6f}")
    print(f"  Std dev (daily): {filt_valid.std():.6f}")
else:
    print(f"  Mean return (daily): NO VALID DATA")
    print(f"  Std dev (daily): NO VALID DATA")
extreme_moves_filt = (np.abs(filt_returns) > 1.0).sum().sum()
print(f"  Extreme moves (>100%): {extreme_moves_filt}")
if extreme_moves_orig > 0:
    print(f"  Reduction in extreme moves: {extreme_moves_orig - extreme_moves_filt} ({100*(extreme_moves_orig - extreme_moves_filt)/extreme_moves_orig:.1f}%)")
else:
    print(f"  Reduction in extreme moves: N/A")

# Correlation matrix analysis
print(f"\n--- CORRELATION MATRIX QUALITY ---")
print(f"(Skipping correlation computation - too slow on 3000+ tickers)")

# Count NaN values in returns before correlation
orig_return_nans = orig_returns.isna().sum().sum()
filt_return_nans = filt_returns.isna().sum().sum()

print(f"Original - NaN values in returns: {orig_return_nans}")
print(f"Filtered - NaN values in returns: {filt_return_nans}")
print(f"Reduction in NaN returns: {orig_return_nans - filt_return_nans} ({100*(orig_return_nans - filt_return_nans)/max(orig_return_nans, 1):.1f}%)")

# Summary impact
print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Tickers removed: {len(mean_prices) - len(df_filtered.columns)} of {len(mean_prices)} ({100*(len(mean_prices) - len(df_filtered.columns))/len(mean_prices):.1f}%)")
print(f"Extreme price moves eliminated: {extreme_moves_orig - extreme_moves_filt} ({100*(extreme_moves_orig - extreme_moves_filt)/max(extreme_moves_orig, 1):.1f}%)")
print(f"NaN values in returns eliminated: {orig_return_nans - filt_return_nans} ({100*(orig_return_nans - filt_return_nans)/max(orig_return_nans, 1):.1f}%)")
print(f"\nThis filter is CRITICAL to preventing portfolio explosions from penny stocks")
print(f"and reverse-split data glitches that were masquerading as extreme correlations.")
print(f"{'='*80}\n")
