import pandas as pd
import requests
from edgar import set_identity, get_filings
from tqdm import tqdm

# 1. SEC requires an identity (Name and Email) to prevent blocking
set_identity("Ojas Phirke ojasphirke@utexas.edu")

def fetch_historical_universe(start_year=2000, end_year=2024):
    """
    Builds a DataFrame of active companies per year based on 10-K filings.
    """
    print("Fetching SEC CIK-to-Ticker bulk mapping...")
    headers = {'User-Agent': 'Ojas Phirke ojasphirke@utexas.edu'}
    
    # We pull the bulk mapping once to avoid hitting the SEC API 10,000 times a year
    try:
        mapping_response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
        mapping_data = mapping_response.json()
        # Create a fast lookup dictionary: {'0000320193': 'AAPL'}
        cik_to_ticker = {str(v['cik_str']).zfill(10): v['ticker'] for v in mapping_data.values()}
    except Exception as e:
        print(f"Failed to fetch ticker mapping: {e}")
        cik_to_ticker = {}

    yearly_data = []

    print(f"Scraping EDGAR for active companies from {start_year} to {end_year}...")
    
    # tqdm provides a progress bar in your terminal
    for year in tqdm(range(start_year, end_year + 1), desc="Processing Years"):
        try:
            # Pull all 10-K filings for the specific year
            filings = get_filings(year=year, form="10-K")
            if not filings:
                continue
                
            # edgartools can convert the index directly to a Pandas DataFrame
            df = filings.to_pandas()
            
            # Format CIK to 10-digit strings to match the mapping dictionary
            df['cik_str'] = df['cik'].astype(str).str.zfill(10)
            
            # We only need one record per company per year
            unique_filers = df[['cik_str', 'company']].drop_duplicates().copy()
            
            # Map the Tickers
            unique_filers['ticker'] = unique_filers['cik_str'].map(cik_to_ticker)
            unique_filers['year'] = year
            
            yearly_data.append(unique_filers)
            
        except Exception as e:
            print(f"Error processing year {year}: {e}")

    # Combine all years into one master dataset
    master_universe = pd.concat(yearly_data, ignore_index=True)
    
    # Reorder columns for readability
    master_universe = master_universe[['year', 'ticker', 'cik_str', 'company']]
    
    return master_universe

if __name__ == "__main__":
    # Run the extraction
    start_year = 2000
    end_year = 2026
    df_universe = fetch_historical_universe(start_year, end_year)
    
    # Check the output
    print("\nSample Output:")
    print(df_universe.head())
    
    # Save to your local drive for caching
    output_file = f"historical_universe_{start_year}_{end_year}.csv"
    df_universe.to_csv(output_file, index=False)
    print(f"\nExtraction complete. Saved {len(df_universe)} records to {output_file}")