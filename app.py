"""Basic Streamlit UI for running and viewing backtests."""

from __future__ import annotations

from datetime import date
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


def _default_universe_label(default_config: Optional[BacktestConfig]) -> str:
    """Infer the UI universe label from the configured ticker source."""
    if default_config is None:
        return "S&P 500"

    ticker_source = default_config.ticker_list
    if ticker_source == load_sp500_constituents:
        return "S&P 500"
    if ticker_source == load_sp100_constituents:
        return "S&P 100"
    if ticker_source == load_nasdaq100_constituents:
        return "NASDAQ 100"
    if callable(ticker_source):
        return "Tickers from file"
    if isinstance(ticker_source, list):
        return "Custom ticker list"
    return "S&P 500"


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
    universe_options = [
        "S&P 500",
        "S&P 100",
        "NASDAQ 100",
        "Tickers from file",
        "Custom ticker list",
    ]
    default_universe = _default_universe_label(default_config)

    universe_label = st.sidebar.selectbox(
        "Universe",
        universe_options,
        index=universe_options.index(default_universe),
    )
    custom_tickers = st.sidebar.text_area(
        "Custom tickers",
        value=", ".join(default_config.ticker_list)
        if default_config and isinstance(default_config.ticker_list, list)
        else "AAPL, MSFT, NVDA, AMZN, GOOGL",
        help="Used only when Universe is set to Custom ticker list.",
    )

    default_start_date = (
        pd.to_datetime(default_config.start_backtest_date).date()
        if default_config
        else date(2018, 1, 1)
    )
    default_end_date = (
        pd.to_datetime(default_config.end_backtest_date).date()
        if default_config
        else pd.Timestamp.today().date()
    )
    min_backtest_date = date(1970, 1, 1)
    max_backtest_date = max(default_end_date, pd.Timestamp.today().date())

    start_date = st.sidebar.date_input(
        "Start date",
        value=default_start_date,
        min_value=min_backtest_date,
        max_value=max_backtest_date,
    )
    end_date = st.sidebar.date_input(
        "End date",
        value=default_end_date,
        min_value=min_backtest_date,
        max_value=max_backtest_date,
    )

    lookback_days = st.sidebar.number_input(
        "Lookback days",
        min_value=30,
        max_value=5000,
        value=default_config.lookback_days if default_config else 365,
        step=5,
    )
    signal_recalculation_interval_days = st.sidebar.number_input(
        "Signal recalculation interval (business days)",
        min_value=1,
        max_value=252,
        value=(
            default_config.signal_recalculation_interval_days
            or default_config.rebalance_interval_days
        )
        if default_config
        else 125,
        step=1,
        help="How often to recompute the core-periphery rankings and target portfolio.",
    )
    rebalance_interval_days = st.sidebar.number_input(
        "Trade rebalance interval (business days)",
        min_value=1,
        max_value=252,
        value=default_config.rebalance_interval_days if default_config else 30,
        step=1,
        help="How often to rebalance holdings back to the latest target weights.",
    )
    snapshot_factor_lookback_days = st.sidebar.number_input(
        "Snapshot factor lookback days",
        min_value=30,
        max_value=5000,
        value=(
            default_config.snapshot_factor_lookback_days
            or default_config.factor_lookback_days
            or default_config.lookback_days
        )
        if default_config
        else 365,
        step=5,
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
    cash_hold_ratio = st.sidebar.slider(
        "Cash held ratio",
        min_value=0.0,
        max_value=1.0,
        value=float(default_strategy.cash_hold_ratio),
        step=0.01,
        help="Fraction of portfolio held in cash at each rebalance; only the remainder is invested.",
    )
    explicit_long_tickers = st.sidebar.text_input(
        "Explicit long tickers",
        value=",".join(default_strategy.normalized_explicit_long_tickers),
        help="Comma-separated tickers to long outside the Rossa-selected book, e.g. SPY.",
    )
    explicit_long_total = st.sidebar.number_input(
        "Explicit long total",
        min_value=0.0,
        max_value=3.0,
        value=float(default_strategy.explicit_long_total),
        step=0.1,
        help="Total exposure allocated equally across the explicit long tickers.",
    )
    explicit_short_tickers = st.sidebar.text_input(
        "Explicit short tickers",
        value=",".join(default_strategy.normalized_explicit_short_tickers),
        help="Comma-separated tickers to short outside the Rossa-selected book, e.g. IWM.",
    )
    explicit_short_total = st.sidebar.number_input(
        "Explicit short total",
        min_value=0.0,
        max_value=3.0,
        value=float(default_strategy.explicit_short_total),
        step=0.1,
        help="Total absolute exposure allocated equally across the explicit short tickers.",
    )
    periphery_threshold_quantile = st.sidebar.slider(
        "Periphery threshold quantile",
        min_value=0.0,
        max_value=1.0,
        value=float(default_strategy.periphery_threshold_quantile),
        step=0.01,
    )
    short_selection_quantile_raw = st.sidebar.slider(
        "Short selection quantile (0 = all remaining)",
        min_value=0.0,
        max_value=1.0,
        value=float(default_strategy.short_selection_quantile or 0.0),
        step=0.01,
    )
    long_periphery = st.sidebar.checkbox(
        "Long periphery",
        value=default_strategy.long_periphery,
    )
    selection_mode = st.sidebar.selectbox(
        "Selection mode",
        ["quantile", "top_m"],
        index=["quantile", "top_m"].index(default_strategy.selection_mode)
        if default_strategy.selection_mode in ["quantile", "top_m"]
        else 0,
        help="Use quantile tails or select a fixed number of most peripheral/core stocks.",
    )
    portfolio_size = st.sidebar.number_input(
        "Portfolio size (for top_m)",
        min_value=1,
        max_value=500,
        value=int(default_strategy.portfolio_size or 30),
        step=1,
        disabled=selection_mode != "top_m",
    )
    rank_ties_by_sharpe = st.sidebar.checkbox(
        "Break coreness ties by trailing Sharpe",
        value=default_strategy.rank_ties_by_sharpe,
        help="Closer to the paper's Type-1 description when using top_m selection.",
    )
    weighting_method = st.sidebar.selectbox(
        "Weighting method",
        ["equal", "coreness_proportional", "risk_parity", "markowitz_min_vol", "markowitz_max_sharpe"],
        index=["equal", "coreness_proportional", "risk_parity", "markowitz_min_vol", "markowitz_max_sharpe"].index(
            default_strategy.weighting_method
        )
        if default_strategy.weighting_method in ["equal", "coreness_proportional", "risk_parity", "markowitz_min_vol", "markowitz_max_sharpe"]
        else 0,
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
    do_plot_network = st.sidebar.checkbox(
        "Plot last snapshot network",
        value=default_config.do_plot_network if default_config else False,
    )
    network_filter = st.sidebar.selectbox(
        "Network filter",
        ["none", "mst", "pmfg"],
        index=["none", "mst", "pmfg"].index(default_strategy.network_filter)
        if default_strategy.network_filter in ["none", "mst", "pmfg"]
        else 0,
        help="Apply a graph filter before the core-periphery calculation. PMFG is closest to the paper; MST is the faster approximation.",
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
        short_selection_quantile=(
            float(short_selection_quantile_raw)
            if short_selection_quantile_raw > 0
            else None
        ),
        long_periphery=long_periphery,
        selection_mode=selection_mode,
        portfolio_size=int(portfolio_size) if selection_mode == "top_m" else None,
        rank_ties_by_sharpe=rank_ties_by_sharpe,
        cash_hold_ratio=float(cash_hold_ratio),
        explicit_long_tickers=tuple(
            ticker.strip().upper()
            for ticker in explicit_long_tickers.split(",")
            if ticker.strip()
        ),
        explicit_long_total=explicit_long_total,
        explicit_short_tickers=tuple(
            ticker.strip().upper()
            for ticker in explicit_short_tickers.split(",")
            if ticker.strip()
        ),
        explicit_short_total=explicit_short_total,
        network_filter=network_filter,
        weighting_method=weighting_method,
        max_long_weight=max_long_weight,
        max_short_weight=max_short_weight,
    )

    return BacktestConfig(
        ticker_list=_resolve_universe(universe_label, custom_tickers),
        start_backtest_date=pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        end_backtest_date=pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        lookback_days=int(lookback_days),
        signal_recalculation_interval_days=int(signal_recalculation_interval_days),
        rebalance_interval_days=int(rebalance_interval_days),
        output_excel=output_excel if output_excel_enabled else None,
        output_plots=output_plots,
        do_plot_network=do_plot_network,
        benchmark_tickers=[ticker.strip().upper() for ticker in benchmark_tickers.split(",") if ticker.strip()],
        factor_list=None,
        factor_lookback_days=None,
        snapshot_factor_lookback_days=int(snapshot_factor_lookback_days),
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


def _render_snapshots(result: BacktestResult) -> None:
    st.subheader("Portfolio Snapshots")
    snapshots = result.portfolio_snapshots
    if snapshots is None or snapshots.empty:
        st.info("No portfolio snapshots available.")
        return

    snapshot_dates = pd.to_datetime(snapshots["Snapshot_Date"]).drop_duplicates().sort_values()
    selected_snapshot = st.selectbox(
        "Snapshot date",
        options=snapshot_dates,
        format_func=lambda value: pd.Timestamp(value).strftime("%Y-%m-%d"),
    )

    snapshot_df = snapshots[pd.to_datetime(snapshots["Snapshot_Date"]) == pd.Timestamp(selected_snapshot)].copy()
    snapshot_df = snapshot_df.sort_values(["Action", "Coreness_Rank", "Ticker"])
    selected_signal_date = pd.Timestamp(snapshot_df["Signal_Date"].iloc[0])

    cols = st.columns(5)
    cols[0].metric("Effective date", pd.Timestamp(snapshot_df["Effective_Date"].iloc[0]).strftime("%Y-%m-%d"))
    cols[1].metric("Signal date", pd.Timestamp(snapshot_df["Signal_Date"].iloc[0]).strftime("%Y-%m-%d"))
    cols[2].metric("Positions", str(snapshot_df["Ticker"].nunique()))
    cols[3].metric("Gross long", f"{snapshot_df.loc[snapshot_df['Portfolio_Weight'] > 0, 'Portfolio_Weight'].sum():.3f}")
    cols[4].metric("Gross short", f"{-snapshot_df.loc[snapshot_df['Portfolio_Weight'] < 0, 'Portfolio_Weight'].sum():.3f}")

    st.dataframe(snapshot_df, use_container_width=True)

    exposure_tables = result.snapshot_factor_exposures or {}
    if not exposure_tables:
        st.info("No snapshot factor exposures available.")
        return

    exposure_labels = {
        "market_general": "Market + General",
        "market_general_sectors": "Market + General + Sectors",
        "market_general_sectors_industries": "Market + General + Sectors + Industries",
    }
    exposure_tabs = st.tabs(
        [exposure_labels.get(name, name) for name in exposure_tables.keys()]
    )

    for tab, (factor_set_name, exposure_df) in zip(exposure_tabs, exposure_tables.items()):
        with tab:
            selected_row = exposure_df[
                pd.to_datetime(exposure_df["Signal_Date"]) == selected_signal_date
            ]
            if selected_row.empty:
                st.info("No exposure data for this snapshot.")
                continue

            exposure_row = selected_row.iloc[0]
            cols = st.columns(5)
            cols[0].metric("Used positions", f"{int(exposure_row['Used_Num_Positions'])}")
            cols[1].metric("Dropped positions", f"{int(exposure_row['Dropped_Num_Positions'])}")
            cols[2].metric("R²", f"{exposure_row['R_Squared']:.4f}")
            cols[3].metric(
                "Actual start",
                pd.Timestamp(exposure_row["Actual_Lookback_Start"]).strftime("%Y-%m-%d"),
            )
            cols[4].metric(
                "Actual end",
                pd.Timestamp(exposure_row["Actual_Lookback_End"]).strftime("%Y-%m-%d"),
            )

            metadata_cols = [
                "Snapshot_Date",
                "Signal_Date",
                "Effective_Date",
                "Requested_Lookback_Start",
                "Requested_Lookback_End",
                "Actual_Lookback_Start",
                "Actual_Lookback_End",
                "Requested_Num_Positions",
                "Used_Num_Positions",
                "Dropped_Num_Positions",
                "Requested_Gross_Long_Weight",
                "Requested_Gross_Short_Weight",
                "Used_Gross_Long_Weight",
                "Used_Gross_Short_Weight",
                "R_Squared",
                "Adj_R_Squared",
                "N_Obs",
                "Residual_Std_Error",
            ]
            st.dataframe(selected_row[metadata_cols], use_container_width=True)

            factor_values = (
                selected_row.drop(columns=[col for col in metadata_cols if col in selected_row.columns])
                .transpose()
                .reset_index()
            )
            factor_values.columns = ["Raw_Factor", "Value"]
            pvalue_rows = factor_values[factor_values["Raw_Factor"].str.startswith("PValue:")].copy()
            pvalue_rows["Factor"] = pvalue_rows["Raw_Factor"].str.replace("PValue:", "", regex=False)
            pvalue_rows = pvalue_rows[["Factor", "Value"]].rename(columns={"Value": "P_Value"})

            loading_rows = factor_values[~factor_values["Raw_Factor"].str.startswith("PValue:")].copy()
            loading_rows = loading_rows.rename(columns={"Raw_Factor": "Factor", "Value": "Exposure"})

            factor_values = loading_rows.merge(pvalue_rows, on="Factor", how="left")
            factor_values = factor_values.dropna(subset=["Exposure"])
            st.dataframe(factor_values, use_container_width=True)


def _render_output_files(output_dir: Path) -> None:
    st.subheader("Generated Files")
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

    for image_name in ["backtest_plots.png", "backtest_plots_log.png", "last_snap_network.png"]:
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

    summary_tab, charts_tab, data_tab, snapshots_tab, outputs_tab = st.tabs(
        ["Summary", "Charts", "Data", "Snapshots", "Outputs"]
    )

    with summary_tab:
        _render_summary(result)

    with charts_tab:
        _render_inline_charts(result)

    with data_tab:
        _render_tables(result)

    with snapshots_tab:
        _render_snapshots(result)

    with outputs_tab:
        current_config = st.session_state.get("last_config", config)
        _render_output_files(Path(current_config.output_dir))


if __name__ == "__main__":
    main()
