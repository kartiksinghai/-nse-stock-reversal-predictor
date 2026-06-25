# Data

This project uses daily OHLCV market data for 43 NSE large-cap stocks and the NIFTY 50 index.

## Source

Data is downloaded using the `yfinance` Python library in:

[`notebooks/02_download_raw_data.ipynb`](../notebooks/02_download_raw_data.ipynb)

## Date Range

- Start: `2015-01-01`
- End: `2026-06-19`

## Directory Structure

| Directory | Contents |
|---|---|
| `raw/` | Individual stock-level and NIFTY OHLCV CSV files downloaded from yfinance |
| `interim/` | Combined, validated, and labelled datasets created during preprocessing |
| `processed/` | Feature-engineered modelling datasets and chronological train/validation/test splits |

The full datasets are intentionally not committed to the repository to keep it lightweight. Run the notebooks in numerical order to regenerate them.
