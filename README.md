# fin377-work

## How to use

Install uv (https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1)
- macOs or linux: open a terminal and run `curl -LsSf https://astral.sh/uv/install.sh | sh`
- windows: open command prompt and run `winget install --id=astral-sh.uv  -e`

If you want more info about how to find/use the terminal: https://youtu.be/pcQuuGvsXn0

## Open the project in a code IDE (VS code) / terminal
- for a basic local UI, in terminal, run `uv run streamlit run app.py`, it should open automatically, or go to http://localhost:8501/ in your browser
- for no UI, in terminal, do `uv run runconfig.py`

It should generate accordingly.
Backtests and step-forward evaluations are now defined in `runconfig.py` inside `create_backtests()`.

## Paper-like settings

For a closer match to the 2025 Scientific Reports core-periphery paper, use:
- `network_filter="pmfg"`
- `selection_mode="top_m"`
- `portfolio_size=30` (or `5`, `10`, `20`)
- `rank_ties_by_sharpe=True`
- `short_amount=0.0`
- `weighting_method="equal"` or `"markowitz_max_sharpe"`

This matches the paper's long-only fixed-size portfolio construction more closely than the default quantile long/short backtest.
