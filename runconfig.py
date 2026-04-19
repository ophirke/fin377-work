"""Readable run configuration for backtests and step-forward evaluation."""

import argparse
from dataclasses import asdict, is_dataclass
from dataclasses import replace
from pathlib import Path
from pprint import pformat

import pandas as pd

from backtest import (
    BacktestConfig,
    SingleBacktestRun,
    StrategyConfig,
    StepForwardBacktestRun,
    load_tickers_from_file,
)
from datamarshal import DataConfig, load_nasdaq100_constituents, load_sp100_constituents, load_sp500_constituents


def create_backtests(output_dir: str | None = None):
    """Create the backtest plans to execute."""
    lookback_days = 365 // 2
    signal_recalculation_interval_days = 60
    rebalance_interval_days = 5
    factor_lookback_days = 365 // 2
    # factor_lookback_days = None
    benchmark_tickers = ["SPY", "IWM", "QQQ"]
    
    strategy = StrategyConfig(
        target_net_exposure=1.0,
        short_amount=1.0,
        # enable_market_drawdown_stop=True,
        # market_drawdown_threshold=0.05,
        # market_drawdown_ticker="SPY",
        # market_drawdown_action="hold",
        # explicit_long_tickers=("SPY",),
        # explicit_long_total=1.0,
        # explicit_short_tickers=("IWM",),
        # explicit_short_total=1.0,
        cash_hold_ratio=0.00,
        network_filter="none",
        periphery_threshold_quantile=0.49,
        short_selection_quantile=0.49,
        long_periphery=True,
        # selection_mode="quantile",
        selection_mode="top_m",
        portfolio_size=50,
        rank_ties_by_sharpe=True,
        weighting_method="equal",
        # max_long_weight=1.0,
        # max_short_weight=0.90,
    )

    # Pick the universe you want to use.
    # ticker_source = load_sp100_constituents
    ticker_source = load_sp500_constituents
    # ticker_source = load_nasdaq100_constituents
    # ticker_source = load_tickers_from_file(DataConfig.TICKER_FILE)
    # ticker_source = load_tickers_from_file(DataConfig.DATA_DIR / "russell2000.txt")

    base_config = BacktestConfig(
        ticker_list=ticker_source,
        start_backtest_date="2008-01-01",
        end_backtest_date="2026-04-01",
        lookback_days=lookback_days,
        signal_recalculation_interval_days=signal_recalculation_interval_days,
        rebalance_interval_days=rebalance_interval_days,
        output_excel=None,
        output_plots=False,
        do_plot_network=False,
        benchmark_tickers=benchmark_tickers,
        factor_list=None,
        factor_lookback_days=None,
        snapshot_factor_lookback_days=lookback_days,
        factor_data_file=str(DataConfig.FACTOR_FILE),
        output_dir=output_dir or str(DataConfig.OUTPUT_DIR),
        summary_file=None,
        parallel=True,
        strategy=strategy,
    )

    plans = []

    plans.append(
        SingleBacktestRun(
            name="sp500_full_backtest",
            config=replace(
                base_config,
                output_excel="backtest_results.xlsx",
                output_plots=True,
                do_plot_network=True,
                summary_file="backtest_summary.txt",
                factor_lookback_days=factor_lookback_days,
            ),
        )
    )

    plans.append(
        StepForwardBacktestRun(
            name="sp500_step_forward",
            base_config=replace(
                base_config,
                output_plots=False,
                output_excel=None,
                summary_file=None,
                factor_lookback_days=None,
                parallel=False,
            ),
            overall_start_date="2008-01-01",
            overall_end_date="2026-04-01",
            eval_lookback=pd.DateOffset(years=1),
            eval_interval=pd.DateOffset(months=1),
            summary_plot_filename="summary_over_time.png",
            summary_excel_filename="summary_over_time.xlsx",
            parallel=True,
        )
    )

    return plans


def apply_experiment_output_dir(plans, experiment_name: str | None):
    """Redirect all generated artifacts to an experiment folder when requested."""
    if not experiment_name:
        return plans

    experiment_output_dir = Path("experiments") / experiment_name
    updated_plans = []
    for plan in plans:
        if hasattr(plan, "config") and not hasattr(plan, "base_config"):
            updated_plans.append(
                replace(plan, config=replace(plan.config, output_dir=str(experiment_output_dir)))
            )
        elif hasattr(plan, "base_config"):
            updated_plans.append(
                replace(
                    plan,
                    base_config=replace(plan.base_config, output_dir=str(experiment_output_dir)),
                )
            )
        else:
            updated_plans.append(plan)
    return updated_plans


def dump_experiment_config(plans, experiment_name: str | None):
    """Write a readable config snapshot into the experiment output folder."""
    if not experiment_name:
        return

    experiment_output_dir = Path("experiments") / experiment_name
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    serialized_plans = []
    for plan in plans:
        if is_dataclass(plan):
            serialized_plans.append(asdict(plan))
        else:
            serialized_plans.append(repr(plan))

    config_dump = {
        "experiment": experiment_name,
        "plans": serialized_plans,
    }

    (experiment_output_dir / "config_snapshot.txt").write_text(
        pformat(config_dump, sort_dicts=False),
        encoding="utf-8",
    )


def parse_args():
    """Parse CLI options for running configured experiments."""
    parser = argparse.ArgumentParser(description="Run configured backtests.")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="If provided, save outputs under experiments/<name>/ instead of the default output dir.",
    )
    return parser.parse_args()


def main():
    from backtest import execute_runs

    args = parse_args()
    plans = create_backtests()
    plans = apply_experiment_output_dir(plans, args.experiment)
    dump_experiment_config(plans, args.experiment)
    execute_runs(plans)


if __name__ == "__main__":
    main()
