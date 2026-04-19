from pathlib import Path
from functools import lru_cache
import pandas as pd
from n100tickers.n100tickers import tickers_as_of

class DataConfig:
    OUTPUT_DIR = Path("outputs")
    
    DATA_DIR = Path("data")
    CACHE_FILE = DATA_DIR / Path("stock_price_history.csv")
    FACTOR_FILE = DATA_DIR / Path("factor_returns.xlsx")
    TICKER_FILE = DATA_DIR / Path("tickers.txt")
    SP_CONSTITUENTS_DIR = DATA_DIR / Path("sp_constituents_yearly_sep30")

    @classmethod
    def sp_constituents_file(cls, year: int):
        if year < 2000 or year > 2025:
            raise ValueError("Year must be between 2000 and 2025")
        return cls.SP_CONSTITUENTS_DIR / Path(f"{year}.csv")
    
    @classmethod
    def sp_constituents_file_for_date(cls, date: pd.Timestamp):
        year = date.year
        if date < pd.Timestamp(year=year, month=9, day=30):
            year -= 1
        return cls.sp_constituents_file(year)


@lru_cache(maxsize=128)
def _load_sp500_constituents_from_file(file_path_str: str) -> tuple:
    """
    Load and cache S&P 500 constituents from a specific file.
    Cached by file path to avoid repeated reads of the same file.
    
    Args:
        file_path_str: File path as string
    
    Returns:
        Tuple of tickers (hashable for use in caching)
    """
    df = pd.read_csv(file_path_str)
    # sort by weight
    df = df.sort_values(by="Weight", ascending=False)
    tickers = df.drop_duplicates(subset="Ticker", keep="first")["Ticker"].tolist()
    return tuple(tickers)


def load_sp500_constituents(date_str: str) -> tuple:
    """
    Load S&P 500 constituents for a given date.
    Uses cached file loading to avoid repeated reads.
    
    Args:
        date_str: Date as string in format "YYYY-MM-DD"
    
    Returns:
        Tuple of tickers
    """
    date = pd.Timestamp(date_str)
    file_path = DataConfig.sp_constituents_file_for_date(date)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Constituents file not found: {file_path}")
    
    return _load_sp500_constituents_from_file(str(file_path))

def load_sp100_constituents(date_str: str) -> tuple:
    """
    Load S&P 100 constituents for a given date.
    For simplicity, we can assume S&P 100 is a subset of S&P 500.
    
    Args:
        date_str: Date as string in format "YYYY-MM-DD"
    
    Returns:
        Tuple of tickers
    """
    sp500_tickers = load_sp500_constituents(date_str)
    print(sp500_tickers)
    sp100_tickers = sp500_tickers[:100]  # Assuming top 100 are the S&P 100
    return tuple(sp100_tickers)

def load_nasdaq100_constituents(date_str: str) -> tuple:
    """
    Load NASDAQ 100 constituents for a given date.
    
    Args:
        date_str: Date as string in format "YYYY-MM-DD"
    
    Returns:
        Tuple of tickers
    """
    date = pd.Timestamp(date_str)
    tickers = tickers_as_of(year=date.year, month=date.month, day=date.day)
    return tuple(tickers)