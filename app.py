"""Basic Streamlit UI for running and viewing backtests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from backtest import (
    BacktestConfig,
    BacktestResult,
    ExcelExportConfig,
    SingleBacktestRun,
    StrategyConfig,
    load_tickers_from_file,
    run_backtest,
)
from datamarshal import DataConfig, load_nasdaq100_constituents, load_sp100_constituents, load_sp500_constituents
from runconfig import create_backtests


st.set_page_config(page_title="Backtest UI", layout="wide")


UNIVERSE_OPTIONS: Dict[str, Callable[[str], tuple] | List[str]] = {
    "S&P 500": load_sp500_constituents,
    "S&P 100": load_sp100_constituents,
    "NASDAQ 100": load_nasdaq100_constituents,
}


def _first_single_plan() -> Optional[SingleBacktestRun]:
    for plan in create_backtests():
        if hasattr(plan, "config") and not hasattr(plan, "base_config"):
            return plan
    return None


def _resolve_universe(label: str, custom_tickers: str) -> Callable[[str], tuple] | List[str]:
    if label == "Custom ticker list":
        tickers = [ticker.strip().upper() for ticker in custom_tickers.split(",") if ticker.strip()]
        return tickers
    if label == "Tickers from file":
        return load_tickers_from_file(DataConfig.TICKER_FILE)
    return UNIVERSE_OPTIONS[label]


def _build_config_from_sidebar(default_config: Optional[BacktestConfig]) -> BacktestConfig:
    default_strategy = default_config.strategy if default_config else StrategyConfig()
    default_excel = default_config.excel_export if default_config else ExcelExportConfig()

    st.sidebar.header("Run Settings")

    universe_label = st.sidebar.selectbox(
        "Universe",
        ["S&P 500", "S&P 100", "NASDAQ 100", "Tickers from file", "Custom ticker list"],
        index=0,
    )
    custom_tickers = st.sidebar.text_area(
        "Custom tickers",
        value="AAPL, MSFT, NVDA, AMZN, GOOGL",
        help="Used only when Universe is set to Custom ticker list.",
    )

    start_date = st.sidebar.date_input(
        "Start date",
        value=pd.to_datetime(default_config.start_backtest_date) if default_config else pd.Timestamp("2018-01-01"),
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=pd.to_datetime(default_config.end_backtest_date) if default_config else pd.Timestamp.today(),
    )

    lookback_days = st.sidebar.number_input(
        "Lookback days",
        min_value=30,
        max_value=5000,
        value=default_config.lookback_days if default_config else 365,
        step=5,
    )
    rebalance_interval_days = st.sidebar.number_input(
        "Rebalance interval (business days)",
        min_value=1,
        max_value=252,
        value=default_config.rebalance_interval_days if default_config else 30,
        step=1,
    )

    benchmark_tickers = st.sidebar.text_input(
        "Benchmarks",
        value=",".join(default_config.benchmark_tickers or ["SPY", "IWM"]) if default_config else "SPY,IWM",
        help="Comma-separated list",
    )

    st.sidebar.subheader("Strategy")
    target_net_exposure = st.sidebar.number_input(
        "Target net exposure",
        min_value=0.0,
        max_value=3.0,
        value=float(default_strategy.target_net_exposure),
        step=0.1,
    )
    short_amount = st.sidebar.number_input(
        "Short amount",
        min_value=0.0,
        max_value=3.0,
        value=float(default_strategy.short_amount),
        step=0.1,
    )
    periphery_threshold_quantile = st.sidebar.slider(
        "Periphery threshold quantile",
        min_value=0.01,
        max_value=0.50,
        value=float(default_strategy.periphery_threshold_quantile),
        step=0.01,
    )
    long_periphery = st.sidebar.checkbox(
        "Long periphery",
        value=default_strategy.long_periphery,
    )
    weighting_method = st.sidebar.selectbox(
        "Weighting method",
        ["equal", "markowitz_min_vol"],
        index=0 if default_strategy.weighting_method == "equal" else 1,
    )
    max_long_weight = st.sidebar.number_input(
        "Max long weight",
        min_value=0.0,
        max_value=5.0,
        value=float(default_strategy.resolved_max_long_weight),
        step=0.05,
    )
    max_short_weight = st.sidebar.number_input(
        "Max short weight",
        min_value=0.0,
        max_value=5.0,
        value=float(default_strategy.resolved_max_short_weight),
        step=0.05,
    )

    st.sidebar.subheader("Outputs")
    output_plots = st.sidebar.checkbox(
        "Generate plot files",
        value=default_config.output_plots if default_config else True,
    )
    output_excel_enabled = st.sidebar.checkbox(
        "Generate Excel file",
        value=bool(default_config.output_excel) if default_config else True,
    )
    output_excel = st.sidebar.text_input(
        "Excel filename",
        value=default_config.output_excel or "backtest_results.xlsx" if default_config else "backtest_results.xlsx",
        disabled=not output_excel_enabled,
    )
    summary_enabled = st.sidebar.checkbox(
        "Generate summary text file",
        value=bool(default_config.summary_file) if default_config else True,
    )
    summary_file = st.sidebar.text_input(
        "Summary filename",
        value=default_config.summary_file or "backtest_summary.txt" if default_config else "backtest_summary.txt",
        disabled=not summary_enabled,
    )
    parallel = st.sidebar.checkbox(
        "Parallel rebalance allocation",
        value=default_config.parallel if default_config else True,
    )
    constant_memory_excel = st.sidebar.checkbox(
        "Constant-memory Excel writer",
        value=default_excel.use_constant_memory,
    )

    strategy = StrategyConfig(
        target_net_exposure=target_net_exposure,
        short_amount=short_amount,
        periphery_threshold_quantile=periphery_threshold_quantile,
        long_periphery=long_periphery,
        weighting_method=weighting_method,
        max_long_weight=max_long_weight,
        max_short_weight=max_short_weight,
    )

    return BacktestConfig(
        ticker_list=_resolve_universe(universe_label, custom_tickers),
        start_backtest_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        end_backtest_date=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        lookback_days=int(lookback_days),
        rebalance_interval_days=int(rebalance_interval_days),
        output_excel=output_excel if output_excel_enabled else None,
        output_plots=output_plots,
        benchmark_tickers=[ticker.strip().upper() for ticker in benchmark_tickers.split(",") if ticker.strip()],
        factor_list=None,
        factor_lookback_days=None,
        factor_data_file=str(DataConfig.FACTOR_FILE),
        summary_file=summary_file if summary_enabled else None,
        parallel=parallel,
        strategy=strategy,
        excel_export=ExcelExportConfig(use_constant_memory=constant_memory_excel),
    )


def _render_summary(result: BacktestResult) -> None:
    st.subheader("Summary")
    cols = st.columns(4)
    cols[0].metric("Total Return", f"{result.summary_stats['Total_Return'] * 100:.2f}%")
    cols[1].metric("Annualized Return", f"{result.summary_stats['Annualized_Return'] * 100:.2f}%")
    cols[2].metric("Volatility", f"{result.summary_stats['Volatility'] * 100:.2f}%")
    cols[3].metric("Sharpe Ratio", f"{result.summary_stats['Sharpe_Ratio']:.3f}")

    cols = st.columns(3)
    cols[0].metric("Sortino Ratio", f"{result.summary_stats['Sortino_Ratio']:.3f}")
    cols[1].metric("Max Drawdown", f"{result.summary_stats['Max_Drawdown'] * 100:.2f}%")
    cols[2].metric("Final Portfolio Value", f"${result.final_portfolio_value:,.2f}")

    comparison = result.benchmark_comparison()
    if comparison is not None:
        st.dataframe(comparison, use_container_width=True)


def _render_inline_charts(result: BacktestResult) -> None:
    st.subheader("Charts")
    portfolio_summary = result.portfolio_summary.copy()
    portfolio_summary["Date"] = pd.to_datetime(portfolio_summary["Date"])

    chart_df = portfolio_summary[["Date", "Portfolio_Value"]].rename(columns={"Portfolio_Value": "Strategy"})
    if result.benchmark_data:
        for ticker, bench_df in result.benchmark_data.items():
            bench_series = bench_df[["Date", "Portfolio_Value"]].copy()
            bench_series["Date"] = pd.to_datetime(bench_series["Date"])
            chart_df = chart_df.merge(
                bench_series.rename(columns={"Portfolio_Value": ticker}),
                on="Date",
                how="left",
            )
    st.line_chart(chart_df.set_index("Date"), height=320)

    growth_df = pd.DataFrame({"Date": portfolio_summary["Date"], "Strategy": portfolio_summary["Portfolio_Value"] / portfolio_summary["Portfolio_Value"].iloc[0]})
    if result.benchmark_data:
        for ticker, bench_df in result.benchmark_data.items():
            bench_series = bench_df.copy()
            bench_series["Date"] = pd.to_datetime(bench_series["Date"])
            growth_df = growth_df.merge(
                pd.DataFrame(
                    {
                        "Date": bench_series["Date"],
                        ticker: bench_series["Portfolio_Value"] / bench_series["Portfolio_Value"].iloc[0],
                    }
                ),
                on="Date",
                how="left",
            )

    fig, ax = plt.subplots(figsize=(12, 4))
    for column in growth_df.columns:
        if column != "Date":
            ax.plot(growth_df["Date"], growth_df[column], label=column)
    ax.set_yscale("log")
    ax.set_title("Growth of $1 (Log Scale)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth Multiple")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    st.pyplot(fig, clear_figure=True)


def _render_tables(result: BacktestResult) -> None:
    st.subheader("Data")
    st.markdown("**Portfolio summary**")
    st.dataframe(result.portfolio_summary, use_container_width=True)

    if result.benchmark_data:
        for ticker, bench_df in result.benchmark_data.items():
            with st.expander(f"Benchmark data: {ticker}", expanded=False):
                st.dataframe(bench_df, use_container_width=True)


def _render_output_files() -> None:
    st.subheader("Generated Files")
    output_dir = Path(DataConfig.OUTPUT_DIR)
    if not output_dir.exists():
        st.info("No output files found yet.")
        return

    files = sorted(output_dir.iterdir(), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        st.info("No output files found yet.")
        return

    file_rows = [
        {
            "Name": path.name,
            "Size (KB)": round(path.stat().st_size / 1024, 1),
            "Modified": pd.Timestamp(path.stat().st_mtime, unit="s"),
        }
        for path in files
    ]
    st.dataframe(pd.DataFrame(file_rows), use_container_width=True)

    for image_name in ["backtest_plots.png", "backtest_plots_log.png"]:
        image_path = output_dir / image_name
        if image_path.exists():
            st.image(str(image_path), caption=image_name, use_container_width=True)

    summary_path = output_dir / "backtest_summary.txt"
    if summary_path.exists():
        with st.expander("Latest summary text", expanded=False):
            st.code(summary_path.read_text(encoding="utf-8"), language="text")


def main() -> None:
    st.title("Backtest UI")
    st.caption("Run backtests on demand and inspect metrics, charts, and output files.")

    preset = _first_single_plan()
    default_config = preset.config if preset else None
    config = _build_config_from_sidebar(default_config)

    if st.sidebar.button("Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Running backtest..."):
            st.session_state["last_result"] = run_backtest(config)
            st.session_state["last_config"] = config

    result = st.session_state.get("last_result")
    if result is None:
        st.info("Choose parameters in the sidebar and click Run Backtest.")
        return

    summary_tab, charts_tab, data_tab, outputs_tab = st.tabs(
        ["Summary", "Charts", "Data", "Outputs"]
    )

    with summary_tab:
        _render_summary(result)

    with charts_tab:
        _render_inline_charts(result)

    with data_tab:
        _render_tables(result)

    with outputs_tab:
        _render_output_files()


if __name__ == "__main__":
    main()
