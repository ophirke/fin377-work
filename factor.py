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
import logging
from typing import Optional, List, Dict, Tuple
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

# LOGGING CONFIGURATION
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for verbose output
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


GENERAL_FACTOR_NAMES = [
    "OilPrice",
    "GoldPrice",
    "InterestRate",
    "USDollar",
    "Buybacks",
    "CreditRisk",
    "SmallSize",
    "Momentum",
    "Value",
    "Quality",
    "LowVolatility",
    "Growth",
    "BetaFactor",
    "DividendYield",
    "Liquidity",
    "MarketBreadth",
]


def select_factor_columns(
    factor_returns: pd.DataFrame,
    include_market: bool = True,
    include_general: bool = True,
    include_sectors: bool = False,
    include_industries: bool = False,
) -> List[str]:
    """
    Select factor columns by named groups and prefixes.

    Returns only columns that are present in the factor data.
    """
    selected = []

    if include_market and "Market" in factor_returns.columns:
        selected.append("Market")

    if include_general:
        selected.extend(
            factor
            for factor in GENERAL_FACTOR_NAMES
            if factor in factor_returns.columns and factor not in selected
        )

    if include_sectors:
        selected.extend(
            col
            for col in factor_returns.columns
            if col.startswith("Sector:") and col not in selected
        )

    if include_industries:
        selected.extend(
            col
            for col in factor_returns.columns
            if col.startswith("Industry:") and col not in selected
        )

    return selected


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
    
    logger.info(f"Loaded factor data: {len(df)} dates, {len(df.columns)} factors")
    logger.info(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    logger.info(f"Factors: {list(df.columns)}")
    
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
            logger.info(f"  {col}: {coverage*100:.1f}% coverage")
        else:
            logger.debug(f"  {col}: {coverage*100:.1f}% coverage (EXCLUDED - insufficient data)")
    
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
    
    logger.debug("Aligning returns:")
    logger.debug(f"  Portfolio dates: {len(portfolio_returns)}")
    logger.debug(f"  Factor dates: {len(factor_returns)}")
    logger.debug(f"  Common dates: {len(common_dates)}")
    
    # Align to common dates
    portfolio_aligned = portfolio_returns.loc[common_dates].copy()
    factors_aligned = factor_returns.loc[common_dates].copy()
    
    # Drop rows with any NaN
    valid_mask = portfolio_aligned.notna() & factors_aligned.notna().all(axis=1)
    portfolio_aligned = portfolio_aligned[valid_mask]
    factors_aligned = factors_aligned[valid_mask]
    
    logger.debug(f"  Valid dates (no NaN): {len(portfolio_aligned)}")
    
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
    logger.info("=" * 70)
    logger.info("FACTOR LOADINGS ANALYSIS")
    logger.info("=" * 70)
    
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
    
    logger.debug("Aligning returns:")
    logger.debug(f"  Portfolio dates: {len(portfolio_returns)}")
    logger.debug(f"  Factor dates: {len(factor_returns)}")
    logger.debug(f"  Common dates: {len(common_dates)}")
    
    # Get available factors (on full aligned data, before dropping NaNs)
    logger.debug(f"Available factors (coverage >= {min_factor_coverage*100:.0f}%):")
    available_factors = get_available_factors(fact_aligned, min_coverage=min_factor_coverage)
    
    # Filter to user-specified factors
    if factors is not None:
        selected_factors = [f for f in factors if f in available_factors]
        logger.info(f"User selected: {selected_factors}")
    else:
        selected_factors = available_factors
        logger.info(f"Using all available factors: {selected_factors}")
    
    # Ensure market factor is included
    if market_factor not in selected_factors and market_factor in available_factors:
        selected_factors = [market_factor] + [f for f in selected_factors if f != market_factor]
        logger.info(f"Added market factor ({market_factor}). Final selection: {selected_factors}")
    elif market_factor not in available_factors:
        logger.warning(f"Market factor ({market_factor}) not available in data")
    
    if not selected_factors:
        raise ValueError("No factors available for analysis")
    
    # Prepare regression data by dropping NaN rows for SELECTED FACTORS ONLY
    # (not for all factors, which would discard valid data)
    X = fact_aligned[selected_factors].copy()
    valid_mask = port_aligned.notna() & X.notna().all(axis=1)
    X = X[valid_mask]
    y = port_aligned[valid_mask]
    
    # Initial regression with all factors
    logger.info("-" * 70)
    logger.info("STEP 1: Initial Regression (All Factors)")
    logger.info("-" * 70)
    
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
    
    logger.debug(f"Significance threshold (α): {significance_threshold:.4f} (two-tailed p-values from statsmodels)")
    logger.debug(f"\nInitial regression results:")
    logger.debug(model_initial.summary())
    
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
    
    logger.debug(f"Significance test results:")
    for _, row in results_initial.iterrows():
        sig_marker = "✓ SIGNIFICANT" if row['Significant'] else "✗ NOT SIGNIFICANT"
        logger.debug(f"  {row['Factor']:15s}: t={row['t_stat']:7.3f}, p={row['p_value']:.4f} {sig_marker}")
    
    # Filter to significant factors (excluding constant)
    significant_factors = results_initial[results_initial['Significant']]['Factor'].tolist()
    
    # Always keep market factor if it was in the model
    if market_factor in selected_factors and market_factor not in significant_factors:
        significant_factors = [market_factor] + significant_factors
        logger.info(f"\nForce-including market factor: {market_factor}")
    
    # Re-run regression with only significant factors
    if len(significant_factors) < len(selected_factors):
        logger.info("\n" + "-" * 70)
        logger.info(f"STEP 2: Refined Regression (Significant Factors Only)")
        logger.info("-" * 70)
        
        X_refined = X[significant_factors].copy()
        X_refined_const = sm.add_constant(X_refined, has_constant='add')
        model_refined = OLS(y, X_refined_const).fit()
        
        logger.debug(f"\nRefined regression ({len(significant_factors)} factors):")
        logger.debug(model_refined.summary())
        
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
        logger.info("All factors are significant. Using initial regression results.")
        results_final = results_initial.copy()
        model_final = model_initial
    
    # Summary statistics
    logger.info("\n" + "=" * 70)
    logger.info("REGRESSION DIAGNOSTICS")
    logger.info("=" * 70)
    
    diagnostics = {
        'r_squared': model_final.rsquared,
        'adj_r_squared': model_final.rsquared_adj,
        'n_obs': len(y),
        'residual_std_error': np.sqrt(model_final.mse_resid),
        'coefficient_pvalues': {
            ('Intercept' if factor == 'const' else factor): p_value
            for factor, p_value in model_final.pvalues.items()
        },
        'f_statistic': model_final.fvalue,
        'f_pvalue': model_final.f_pvalue,
        'aic': model_final.aic,
        'bic': model_final.bic,
    }
    
    logger.debug(f"R-squared: {diagnostics['r_squared']:.4f}")
    logger.debug(f"Adjusted R-squared: {diagnostics['adj_r_squared']:.4f}")
    logger.debug(f"F-statistic: {diagnostics['f_statistic']:.4f} (p-value: {diagnostics['f_pvalue']:.4e})")
    logger.debug(f"Residual Std Error: {diagnostics['residual_std_error']:.6f}")
    logger.debug(f"N observations: {diagnostics['n_obs']}")
    logger.debug(f"AIC: {diagnostics['aic']:.2f}, BIC: {diagnostics['bic']:.2f}")
    
    # Sort by absolute loading
    results_final['Abs_Loading'] = results_final['Loading'].abs()
    results_final = results_final.sort_values('Abs_Loading', ascending=False)
    results_final = results_final.drop(columns=['Abs_Loading'])
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL FACTOR LOADINGS (Sorted by Magnitude)")
    logger.info("=" * 70)
    logger.debug(results_final.to_string(index=False))
    
    return results_final, diagnostics

 
def compute_factor_loadings_iterative(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    factors: Optional[List[str]] = None,
    market_factor: str = "Market",
    confidence_threshold: float = 0.95,
    min_factor_coverage: float = 1.0,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute factor loadings via OLS regression with iterative significance pruning.

    Non-market factors are pruned repeatedly until every remaining non-market
    factor is significant at the requested threshold. The market factor may be
    retained even when it is not significant.
    """
    logger.info("=" * 70)
    logger.info("FACTOR LOADINGS ANALYSIS")
    logger.info("=" * 70)

    port_idx = portfolio_returns.index
    fact_idx = factor_returns.index

    if not isinstance(port_idx, pd.DatetimeIndex):
        port_idx = pd.to_datetime(port_idx)
    if not isinstance(fact_idx, pd.DatetimeIndex):
        fact_idx = pd.to_datetime(fact_idx)

    common_dates = port_idx.intersection(fact_idx)
    port_aligned = portfolio_returns.loc[common_dates].copy()
    fact_aligned = factor_returns.loc[common_dates].copy()

    logger.debug("Aligning returns:")
    logger.debug(f"  Portfolio dates: {len(portfolio_returns)}")
    logger.debug(f"  Factor dates: {len(factor_returns)}")
    logger.debug(f"  Common dates: {len(common_dates)}")

    logger.debug(f"Available factors (coverage >= {min_factor_coverage*100:.0f}%):")
    available_factors = get_available_factors(
        fact_aligned, min_coverage=min_factor_coverage
    )

    if factors is not None:
        selected_factors = [factor for factor in factors if factor in available_factors]
        logger.info(f"User selected: {selected_factors}")
    else:
        selected_factors = available_factors
        logger.info(f"Using all available factors: {selected_factors}")

    if market_factor not in selected_factors and market_factor in available_factors:
        selected_factors = [market_factor] + [
            factor for factor in selected_factors if factor != market_factor
        ]
        logger.info(
            f"Added market factor ({market_factor}). Final selection: {selected_factors}"
        )
    elif market_factor not in available_factors:
        logger.warning(f"Market factor ({market_factor}) not available in data")

    if not selected_factors:
        raise ValueError("No factors available for analysis")

    X = fact_aligned[selected_factors].copy()
    valid_mask = port_aligned.notna() & X.notna().all(axis=1)
    X = X[valid_mask]
    y = port_aligned[valid_mask]

    significance_threshold = 1 - confidence_threshold
    logger.debug(
        f"Significance threshold (alpha): {significance_threshold:.4f} (two-tailed p-values from statsmodels)"
    )

    current_factors = selected_factors.copy()
    iteration = 1

    while True:
        step_label = (
            "Initial Regression (All Factors)"
            if iteration == 1
            else f"Refined Regression {iteration}"
        )
        logger.info("-" * 70)
        logger.info(f"STEP {iteration}: {step_label}")
        logger.info("-" * 70)

        X_current = X[current_factors].copy()
        X_const = sm.add_constant(X_current, has_constant="add")
        model_current = OLS(y, X_const).fit()

        logger.debug(
            f"\nRegression iteration {iteration} ({len(current_factors)} factors):"
        )
        logger.debug(model_current.summary())

        results_current = pd.DataFrame(
            {
                "Factor": model_current.params.index,
                "Loading": model_current.params.values,
                "Std_Error": model_current.bse.values,
                "t_stat": model_current.tvalues.values,
                "p_value": model_current.pvalues.values,
            }
        )
        results_current["Significant"] = (
            results_current["p_value"] < significance_threshold
        )
        results_current = results_current[
            results_current["Factor"] != "const"
        ].reset_index(drop=True)

        logger.debug("Significance test results:")
        for _, row in results_current.iterrows():
            sig_marker = "SIGNIFICANT" if row["Significant"] else "NOT SIGNIFICANT"
            logger.debug(
                f"  {row['Factor']:15s}: t={row['t_stat']:7.3f}, p={row['p_value']:.4f} {sig_marker}"
            )

        insignificant_factors = results_current.loc[
            ~results_current["Significant"]
            & (results_current["Factor"] != market_factor),
            "Factor",
        ].tolist()

        if not insignificant_factors:
            if market_factor in current_factors:
                market_row = results_current[
                    results_current["Factor"] == market_factor
                ]
                if (
                    not market_row.empty
                    and not bool(market_row["Significant"].iloc[0])
                ):
                    logger.info(
                        f"Market factor retained despite p-value >= {significance_threshold:.2f}: {market_factor}"
                    )
            results_final = results_current
            model_final = model_current
            break

        logger.info(f"Pruning non-significant factors: {insignificant_factors}")
        current_factors = [
            factor for factor in current_factors if factor not in insignificant_factors
        ]
        if not current_factors:
            raise ValueError("No significant factors remain after iterative pruning")
        iteration += 1

    diagnostics = {
        "r_squared": model_final.rsquared,
        "adj_r_squared": model_final.rsquared_adj,
        "n_obs": len(y),
        "residual_std_error": np.sqrt(model_final.mse_resid),
        "coefficient_pvalues": {
            ("Intercept" if factor == "const" else factor): p_value
            for factor, p_value in model_final.pvalues.items()
        },
        "coefficient_loadings": {
            ("Intercept" if factor == "const" else factor): loading
            for factor, loading in model_final.params.items()
        },
        "f_statistic": model_final.fvalue,
        "f_pvalue": model_final.f_pvalue,
        "aic": model_final.aic,
        "bic": model_final.bic,
    }

    results_final["Abs_Loading"] = results_final["Loading"].abs()
    results_final = results_final.sort_values("Abs_Loading", ascending=False)
    results_final = results_final.drop(columns=["Abs_Loading"])

    logger.info("\n" + "=" * 70)
    logger.info("FINAL FACTOR LOADINGS (Sorted by Magnitude)")
    logger.info("=" * 70)
    logger.debug(results_final.to_string(index=False))

    return results_final, diagnostics


compute_factor_loadings = compute_factor_loadings_iterative


def compute_factor_exposures(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    factors: List[str],
    market_factor: str = "Market",
    min_factor_coverage: float = 0.7,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute factor exposures without dropping non-significant factors.

    This is useful when you want the full exposure surface for a portfolio
    snapshot rather than only statistically significant coefficients.
    """
    if not factors:
        raise ValueError("No factors supplied for exposure computation")

    available_factors = get_available_factors(factor_returns, min_coverage=min_factor_coverage)
    selected_factors = [factor for factor in factors if factor in available_factors]

    if market_factor in factors and market_factor not in selected_factors and market_factor in factor_returns.columns:
        selected_factors = [market_factor] + selected_factors

    if not selected_factors:
        raise ValueError("None of the requested factors are available with sufficient coverage")

    portfolio_aligned, factor_aligned = align_returns_with_factors(
        portfolio_returns, factor_returns[selected_factors]
    )
    if len(portfolio_aligned) < 10:
        raise ValueError(
            f"Insufficient aligned observations for exposure computation: {len(portfolio_aligned)}"
        )

    X_const = sm.add_constant(factor_aligned, has_constant="add")
    model = OLS(portfolio_aligned, X_const).fit()

    exposures = pd.DataFrame(
        {
            "Factor": model.params.index,
            "Loading": model.params.values,
            "Std_Error": model.bse.values,
            "t_stat": model.tvalues.values,
            "p_value": model.pvalues.values,
        }
    )
    exposures = exposures[exposures["Factor"] != "const"].reset_index(drop=True)

    diagnostics = {
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "n_obs": len(portfolio_aligned),
        "residual_std_error": np.sqrt(model.mse_resid),
        "sample_start": portfolio_aligned.index.min(),
        "sample_end": portfolio_aligned.index.max(),
        "coefficient_pvalues": {
            ("Intercept" if factor == "const" else factor): p_value
            for factor, p_value in model.pvalues.items()
        },
        "coefficient_loadings": {
            ("Intercept" if factor == "const" else factor): loading
            for factor, loading in model.params.items()
        },
        "f_statistic": model.fvalue,
        "f_pvalue": model.f_pvalue,
        "aic": model.aic,
        "bic": model.bic,
    }

    return exposures, diagnostics


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
