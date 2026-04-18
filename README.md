# fin377-work

## How to use

Install uv (https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1)
- macOs or linux: open a terminal and run `curl -LsSf https://astral.sh/uv/install.sh | sh`
- windows: open command prompt and run `winget install --id=astral-sh.uv  -e`

If you want more info about how to find/use the terminal: https://youtu.be/pcQuuGvsXn0

## Open the project in a code IDE (VS code) / terminal
- in terminal, do `uv run runconfig.py`
- for a basic local UI, do `uv run streamlit run app.py`, it should open automatically, or go to http://localhost:8501/ in your browser

It should generate accordingly.
Backtests and step-forward evaluations are now defined in `runconfig.py` inside `create_backtests()`.
