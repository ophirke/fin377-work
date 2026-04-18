"""Quick check of the cached price data for data quality issues."""
import pandas as pd
import numpy as np

print("Loading cached price data...")
df = pd.read_csv("data/stock_price_history.csv", index_col=0, parse_dates=True)
print(f"Shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"Number of tickers: {len(df.columns)}")

# Check for duplicates
if df.columns.has_duplicates:
    dup_tickers = df.columns[df.columns.duplicated()].unique().tolist()
    print(f"\n❌ DUPLICATE COLUMNS: {len(dup_tickers)} unique duplicate tickers")
    print(f"   Examples: {dup_tickers[:10]}")
else:
    print(f"\n✓ No duplicate columns")

# Check for extreme values
print(f"\n--- Price Statistics ---")
print(f"Min price: ${df.min().min():.2f}")
print(f"Max price: ${df.max().max():.2f}")
print(f"Mean price: ${df.values[~np.isnan(df.values)].mean():.2f}")

# Find stocks with extreme prices
extreme_high = (df.max() > 1000000).sum()
extreme_low = (df.min() < 0.01).sum()
print(f"Stocks with price > $1M: {extreme_high}")
print(f"Stocks with price < $0.01: {extreme_low}")

if extreme_high > 0:
    extremely_high_stocks = df.max()[df.max() > 1000000].sort_values(ascending=False)
    print(f"   Top 5: {extremely_high_stocks.head().to_dict()}")

if extreme_low > 0:
    extremely_low_stocks = df.min()[df.min() < 0.01].sort_values()
    print(f"   Examples: {extremely_low_stocks.head().to_dict()}")

# Check for extreme daily moves
print(f"\n--- Extreme Daily Moves ---")
pct_change = df.pct_change()
extreme_moves = (pct_change.abs() > 1.0).sum()  # >100% moves
extreme_moves_sum = extreme_moves[extreme_moves > 0].sort_values(ascending=False)
print(f"Stocks with >100% daily move: {len(extreme_moves_sum)}")
if len(extreme_moves_sum) > 0:
    print(f"   Top 5: {extreme_moves_sum.head().to_dict()}")

# Check for NaN values
print(f"\n--- NaN Check ---")
nan_count = df.isna().sum()
nan_per_column = nan_count[nan_count > 0].sort_values(ascending=False)
print(f"Columns with NaN: {len(nan_per_column)}")
if len(nan_per_column) > 0:
    print(f"   Top columns with most NaN: {nan_per_column.head().to_dict()}")

# Check if columns are sorted/unique
print(f"\n--- Column Integrity ---")
print(f"All columns unique: {df.columns.is_unique}")
print(f"Sample columns: {list(df.columns[:10])}")

print("\nDone!")
