from pathlib import Path
from functools import lru_cache
import pandas as pd

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
    tickers = df["Ticker"].tolist()
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