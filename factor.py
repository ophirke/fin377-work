"""
Factor analysis module for computing portfolio factor loadings.

Provides functions to:
- Load factor returns from Excel
- Align portfolio returns with factor data by date
- Compute factor loadings via regression
- Test statistical significance of factors
- Filter and return only significant factors
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS


def load_factor_data(xlsx_file: str, sheet_name: str = 0) -> pd.DataFrame:
    """
    Load factor returns from Excel file.
    
    Expects a DataFrame with:
    - A 'Date' column (or index that can be converted to datetime)
    - Factor return columns (one per factor)
    
    Args:
        xlsx_file: Path to Excel file with factor returns
        sheet_name: Sheet name or index to read (default: first sheet)
    
    Returns:
        DataFrame with datetime index and factor return columns
    """
    df = pd.read_excel(xlsx_file, sheet_name=sheet_name)
    
    # Try to find and use Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    else:
        # Try to convert index to datetime
        df.index = pd.to_datetime(df.index)
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date
    df = df.sort_index()
    
    print(f"Loaded factor data: {len(df)} dates, {len(df.columns)} factors")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Factors: {list(df.columns)}")
    
    return df


def get_available_factors(
    factor_returns: pd.DataFrame,
    min_coverage: float = 1.0
) -> List[str]:
    """
    Get factors that have data for the required coverage period.
    
    Args:
        factor_returns: DataFrame with factor returns indexed by date
        min_coverage: Minimum fraction of dates that must have data (0-1)
                     Default 1.0 = must have data for all dates
    
    Returns:
        List of factors with sufficient data coverage
    """
    total_dates = len(factor_returns)
    available_factors = []
    
    for col in factor_returns.columns:
        non_na_count = factor_returns[col].notna().sum()
        coverage = non_na_count / total_dates
        
        if coverage >= min_coverage:
            available_factors.append(col)
            print(f"  {col}: {coverage*100:.1f}% coverage")
        else:
            print(f"  {col}: {coverage*100:.1f}% coverage (EXCLUDED - insufficient data)")
    
    return available_factors


def align_returns_with_factors(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Align portfolio returns with factor returns by date.
    
    Ensures both DataFrames have the same date index and matching dates.
    Removes any dates with NaN values in either series.
    
    Args:
        portfolio_returns: Series of portfolio returns indexed by date
        factor_returns: DataFrame of factor returns indexed by date
    
    Returns:
        Tuple of (aligned_portfolio_returns, aligned_factor_returns)
    """
    # Ensure both are datetime indexed
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    if not isinstance(factor_returns.index, pd.DatetimeIndex):
        factor_returns.index = pd.to_datetime(factor_returns.index)
    
    # Find intersection of dates
    common_dates = portfolio_returns.index.intersection(factor_returns.index)
    
    print(f"\nAligning returns:")
    print(f"  Portfolio dates: {len(portfolio_returns)}")
    print(f"  Factor dates: {len(factor_returns)}")
    print(f"  Common dates: {len(common_dates)}")
    
    # Align to common dates
    portfolio_aligned = portfolio_returns.loc[common_dates].copy()
    factors_aligned = factor_returns.loc[common_dates].copy()
    
    # Drop rows with any NaN
    valid_mask = portfolio_aligned.notna() & factors_aligned.notna().all(axis=1)
    portfolio_aligned = portfolio_aligned[valid_mask]
    factors_aligned = factors_aligned[valid_mask]
    
    print(f"  Valid dates (no NaN): {len(portfolio_aligned)}")
    
    return portfolio_aligned, factors_aligned


def compute_factor_loadings(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    factors: Optional[List[str]] = None,
    market_factor: str = 'Market',
    confidence_threshold: float = 0.95,
    min_factor_coverage: float = 1.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute factor loadings via OLS regression with significance testing.
    
    Workflow:
    1. Load and align returns
    2. Filter factors by user selection + availability
    3. Ensure market factor is included
    4. Run initial regression with all factors
    5. Test statistical significance (default 95% confidence)
    6. Remove non-significant factors
    7. Re-run regression with significant factors only
    
    Args:
        portfolio_returns: Series of portfolio daily/monthly returns indexed by date
        factor_returns: DataFrame of factor returns indexed by date
        factors: Optional list of factors to use. If None, uses all available factors.
                 Market factor is always included if available.
        market_factor: Name of market factor column (default 'Market')
        confidence_threshold: Confidence level for significance (0-1, default 0.95 = 95%)
        min_factor_coverage: Minimum data coverage required for each factor (0-1, default 1.0)
    
    Returns:
        Tuple of:
        - DataFrame with columns: [Factor, Loading, Std_Error, t_stat, p_value, Significant]
        - Dict with regression diagnostics: {r_squared, adj_r_squared, n_obs, residual_std_error}
    """
    print("\n" + "=" * 70)
    print("FACTOR LOADINGS ANALYSIS")
    print("=" * 70)
    
    # Align returns and factors by date first (WITHOUT dropping NaN rows yet)
    # to preserve valid observations for factors that have data
    port_idx = portfolio_returns.index
    fact_idx = factor_returns.index
    
    if not isinstance(port_idx, pd.DatetimeIndex):
        port_idx = pd.to_datetime(port_idx)
    if not isinstance(fact_idx, pd.DatetimeIndex):
        fact_idx = pd.to_datetime(fact_idx)
    
    common_dates = port_idx.intersection(fact_idx)
    port_aligned = portfolio_returns.loc[common_dates].copy()
    fact_aligned = factor_returns.loc[common_dates].copy()
    
    print(f"\nAligning returns:")
    print(f"  Portfolio dates: {len(portfolio_returns)}")
    print(f"  Factor dates: {len(factor_returns)}")
    print(f"  Common dates: {len(common_dates)}")
    
    # Get available factors (on full aligned data, before dropping NaNs)
    print(f"\nAvailable factors (coverage >= {min_factor_coverage*100:.0f}%):")
    available_factors = get_available_factors(fact_aligned, min_coverage=min_factor_coverage)
    
    # Filter to user-specified factors
    if factors is not None:
        selected_factors = [f for f in factors if f in available_factors]
        print(f"\nUser selected: {selected_factors}")
    else:
        selected_factors = available_factors
        print(f"\nUsing all available factors: {selected_factors}")
    
    # Ensure market factor is included
    if market_factor not in selected_factors and market_factor in available_factors:
        selected_factors = [market_factor] + [f for f in selected_factors if f != market_factor]
        print(f"Added market factor ({market_factor}). Final selection: {selected_factors}")
    elif market_factor not in available_factors:
        print(f"WARNING: Market factor ({market_factor}) not available in data")
    
    if not selected_factors:
        raise ValueError("No factors available for analysis")
    
    # Prepare regression data by dropping NaN rows for SELECTED FACTORS ONLY
    # (not for all factors, which would discard valid data)
    X = fact_aligned[selected_factors].copy()
    valid_mask = port_aligned.notna() & X.notna().all(axis=1)
    X = X[valid_mask]
    y = port_aligned[valid_mask]
    
    print(f"\nRegression data: {len(y)} observations (after NaN drop), {len(X.columns)} factors")
    
    # Initial regression with all factors
    print("\n" + "-" * 70)
    print("STEP 1: Initial Regression (All Factors)")
    print("-" * 70)
    
    X_const = sm.add_constant(X, has_constant='add')
    model_initial = OLS(y, X_const).fit()
    
    # Extract regression results
    loadings_initial = model_initial.params.copy()
    std_errors_initial = model_initial.bse
    t_stats_initial = model_initial.tvalues
    p_values_initial = model_initial.pvalues
    
    # Calculate significance threshold
    # Note: statsmodels OLS.fit() already returns two-tailed p-values,
    # so we compare directly against alpha (not alpha/2)
    alpha = 1 - confidence_threshold
    significance_threshold = alpha
    
    print(f"Significance threshold (α): {significance_threshold:.4f} (two-tailed p-values from statsmodels)")
    print(f"\nInitial regression results:")
    print(model_initial.summary())
    
    # Determine which factors are significant
    results_initial = pd.DataFrame({
        'Factor': loadings_initial.index,
        'Loading': loadings_initial.values,
        'Std_Error': std_errors_initial.values,
        't_stat': t_stats_initial.values,
        'p_value': p_values_initial.values,
    })
    
    results_initial['Significant'] = results_initial['p_value'] < significance_threshold
    results_initial = results_initial[results_initial['Factor'] != 'const']
    
    print(f"\nSignificance test results:")
    for _, row in results_initial.iterrows():
        sig_marker = "✓ SIGNIFICANT" if row['Significant'] else "✗ NOT SIGNIFICANT"
        print(f"  {row['Factor']:15s}: t={row['t_stat']:7.3f}, p={row['p_value']:.4f} {sig_marker}")
    
    # Filter to significant factors (excluding constant)
    significant_factors = results_initial[results_initial['Significant']]['Factor'].tolist()
    
    # Always keep market factor if it was in the model
    if market_factor in selected_factors and market_factor not in significant_factors:
        significant_factors = [market_factor] + significant_factors
        print(f"\nForce-including market factor: {market_factor}")
    
    # Re-run regression with only significant factors
    if len(significant_factors) < len(selected_factors):
        print("\n" + "-" * 70)
        print(f"STEP 2: Refined Regression (Significant Factors Only)")
        print("-" * 70)
        
        X_refined = X[significant_factors].copy()
        X_refined_const = sm.add_constant(X_refined, has_constant='add')
        model_refined = OLS(y, X_refined_const).fit()
        
        print(f"\nRefined regression ({len(significant_factors)} factors):")
        print(model_refined.summary())
        
        # Extract refined results
        loadings_refined = model_refined.params.copy()
        std_errors_refined = model_refined.bse
        t_stats_refined = model_refined.tvalues
        p_values_refined = model_refined.pvalues
        
        results_final = pd.DataFrame({
            'Factor': loadings_refined.index,
            'Loading': loadings_refined.values,
            'Std_Error': std_errors_refined.values,
            't_stat': t_stats_refined.values,
            'p_value': p_values_refined.values,
        })
        
        results_final['Significant'] = results_final['p_value'] < significance_threshold
        results_final = results_final[results_final['Factor'] != 'const']
        
        model_final = model_refined
    else:
        print("\nAll factors are significant. Using initial regression results.")
        results_final = results_initial.copy()
        model_final = model_initial
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("REGRESSION DIAGNOSTICS")
    print("=" * 70)
    
    diagnostics = {
        'r_squared': model_final.rsquared,
        'adj_r_squared': model_final.rsquared_adj,
        'n_obs': len(y),
        'residual_std_error': np.sqrt(model_final.mse_resid),
        'f_statistic': model_final.fvalue,
        'f_pvalue': model_final.f_pvalue,
        'aic': model_final.aic,
        'bic': model_final.bic,
    }
    
    print(f"R-squared: {diagnostics['r_squared']:.4f}")
    print(f"Adjusted R-squared: {diagnostics['adj_r_squared']:.4f}")
    print(f"F-statistic: {diagnostics['f_statistic']:.4f} (p-value: {diagnostics['f_pvalue']:.4e})")
    print(f"Residual Std Error: {diagnostics['residual_std_error']:.6f}")
    print(f"N observations: {diagnostics['n_obs']}")
    print(f"AIC: {diagnostics['aic']:.2f}, BIC: {diagnostics['bic']:.2f}")
    
    # Sort by absolute loading
    results_final['Abs_Loading'] = results_final['Loading'].abs()
    results_final = results_final.sort_values('Abs_Loading', ascending=False)
    results_final = results_final.drop(columns=['Abs_Loading'])
    
    print("\n" + "=" * 70)
    print("FINAL FACTOR LOADINGS (Sorted by Magnitude)")
    print("=" * 70)
    print(results_final.to_string(index=False))
    
    return results_final, diagnostics


def factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    factor_loadings: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute factor attributions: how much each factor contributed to returns.
    
    Attribution = Factor_Loading × Factor_Return
    
    Args:
        portfolio_returns: Series of portfolio returns
        factor_returns: DataFrame of factor returns
        factor_loadings: DataFrame from compute_factor_loadings (with 'Factor' and 'Loading' columns)
    
    Returns:
        DataFrame with daily/period attributions by factor
    """
    # Align data
    port_ret, fact_ret = align_returns_with_factors(portfolio_returns, factor_returns)
    
    # Create attribution DataFrame
    attribution = pd.DataFrame(index=fact_ret.index)
    
    for _, row in factor_loadings.iterrows():
        factor_name = row['Factor']
        loading = row['Loading']
        
        if factor_name in fact_ret.columns:
            attribution[factor_name] = loading * fact_ret[factor_name]
    
    # Add total attributed
    attribution['Total_Attributed'] = attribution.sum(axis=1)
    # Preserve index-based alignment (not positional)
    attribution['Actual_Return'] = port_ret
    attribution['Unexplained (Alpha)'] = attribution['Actual_Return'] - attribution['Total_Attributed']
    
    return attribution
