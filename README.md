# DK1 Electricity Price Forecasting

Forecasting methods for DK1 (West Denmark) day-ahead electricity prices.

## Data

| Source | Variables | Frequency |
|--------|-----------|-----------|
| ENTSO-E | DK1 day-ahead prices, load, wind/solar generation forecasts, cross-border physical flows | Hourly |
| Investing.com | Natural gas (TTF), EUA, coal, oil | Daily |
| FRED | VIX, USD/EUR, DK CPI, GDP | Daily / Monthly / Quarterly |
| ECA&D | DK1 wind speed, temperature, sunshine, precipitation | Daily |

## Structure

```
src/epf/          # core package
  data/           # data loaders
  eda/            # EDA helper functions
analyses/         # exploratory notebooks
data/             # raw data (not versioned)
tests/
```

## Setup

```bash
conda env create -f environment.yml
conda activate electricity
pip install -e ".[dev]"
```
