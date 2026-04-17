"""Readable run configuration for backtests and step-forward evaluation."""

from dataclasses import replace

import pandas as pd

from backtest import (
    BacktestConfig,
    SingleBacktestRun,
    StepForwardBacktestRun,
    load_tickers_from_file,
)
from datamarshal import DataConfig, load_nasdaq100_constituents, load_sp500_constituents


RUN_SINGLE_BACKTEST = True
RUN_STEP_FORWARD_EVAL = False


def create_backtests():
    """Create the backtest plans to execute."""
    lookback_days = 365
    rebalance_interval_days = 30
    factor_lookback_days = 365 * 3
    benchmark_tickers = ["SPY", "IWM"]

    # Pick the universe you want to use.
    ticker_source = load_sp500_constituents
    # ticker_source = load_nasdaq100_constituents
    # ticker_source = load_tickers_from_file(DataConfig.TICKER_FILE)

    base_config = BacktestConfig(
        ticker_list=ticker_source,
        start_backtest_date="2008-04-13",
        end_backtest_date="2026-04-13",
        lookback_days=lookback_days,
        rebalance_interval_days=rebalance_interval_days,
        output_excel=None,
        output_plots=False,
        benchmark_tickers=benchmark_tickers,
        factor_list=None,
        factor_lookback_days=None,
        factor_data_file=str(DataConfig.FACTOR_FILE),
        summary_file=None,
        parallel=True,
    )

    plans = []

    if RUN_SINGLE_BACKTEST:
        plans.append(
            SingleBacktestRun(
                name="sp500_full_backtest",
                config=replace(
                    base_config,
                    output_plots=True,
                    summary_file="backtest_summary.txt",
                    factor_lookback_days=factor_lookback_days,
                ),
            )
        )

    if RUN_STEP_FORWARD_EVAL:
        plans.append(
            StepForwardBacktestRun(
                name="sp500_step_forward",
                base_config=replace(
                    base_config,
                    output_plots=False,
                    output_excel=None,
                    summary_file=None,
                    factor_lookback_days=None,
                ),
                overall_start_date="2008-04-13",
                overall_end_date="2026-04-13",
                eval_lookback=pd.DateOffset(years=1),
                eval_interval=pd.DateOffset(months=1),
                summary_plot_filename="summary_over_time.png",
                parallel=True,
            )
        )

    return plans


if __name__ == "__main__":
    from backtest import execute_runs

    execute_runs(create_backtests())
