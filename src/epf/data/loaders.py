"""
Data loaders for the DK1 electricity price forecasting project.

All loaders return a pandas DataFrame or Series with a DatetimeIndex.
Pass ``data_dir`` as an absolute Path to the project ``data/`` folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_mtu_start(series: pd.Series, dayfirst: bool = True) -> pd.DatetimeIndex:
    """Extract the start timestamp from ENTSO-E MTU range strings.

    Examples handled:
      '31/12/2018 23:00:00 - 01/01/2019 00:00:00'  (prices, UTC)
      '01/01/2020 00:00 - 01/01/2020 01:00'         (loads / renewables, CET/CEST)
    """
    starts = series.astype(str).str.split(" - ").str[0].str.strip()
    return pd.to_datetime(starts, format="%d/%m/%Y %H:%M:%S", errors="coerce").fillna(
        pd.to_datetime(starts, format="%d/%m/%Y %H:%M", errors="coerce")
    )


def _mtu_col(raw: pd.DataFrame) -> pd.Series:
    """Return the MTU column regardless of whether a spurious index column was prepended."""
    for col in raw.columns:
        if "MTU" in str(col):
            return raw[col]
    raise KeyError(f"No MTU column found; columns are: {list(raw.columns)}")


def _concat_dedup(frames: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames)
    return df[~df.index.duplicated(keep="first")].sort_index()


# ---------------------------------------------------------------------------
# ENTSO-E loaders
# ---------------------------------------------------------------------------

def load_prices(data_dir: Path) -> pd.DataFrame:
    """DK1 hourly day-ahead prices (EUR/MWh).

    Index: UTC naive datetime (parsed from the MTU start timestamp).
    Columns: ``price``
    """
    paths = sorted(Path(data_dir, "ElectricityPrices/DK1/Prices").glob("*.csv"))
    frames = []
    for p in paths:
        raw = pd.read_csv(p, quotechar='"', encoding="utf-8-sig")
        ts = _parse_mtu_start(_mtu_col(raw))
        price = pd.to_numeric(raw["Day-ahead Price (EUR/MWh)"], errors="coerce").values
        frames.append(pd.DataFrame({"price": price}, index=ts))
    return _concat_dedup(frames)


def load_loads(data_dir: Path) -> pd.DataFrame:
    """DK1 hourly actual and day-ahead forecast load (MW).

    Index: CET/CEST naive datetime (parsed from the MTU start timestamp).
    Columns: ``actual_load``, ``forecast_load``
    """
    paths = sorted(Path(data_dir, "ElectricityPrices/DK1/Loads").glob("*.csv"))
    frames = []
    for p in paths:
        raw = pd.read_csv(p, quotechar='"', encoding="utf-8-sig")
        ts = _parse_mtu_start(_mtu_col(raw))
        df = pd.DataFrame(
            {
                "actual_load": pd.to_numeric(raw["Actual Total Load (MW)"], errors="coerce").values,
                "forecast_load": pd.to_numeric(
                    raw["Day-ahead Total Load Forecast (MW)"], errors="coerce"
                ).values,
            },
            index=ts,
        )
        frames.append(df)
    return _concat_dedup(frames)


def load_renewables(data_dir: Path, source: str = "onshore") -> pd.DataFrame:
    """DK1 renewable generation forecasts and actuals (MW).

    Parameters
    ----------
    source:
        One of ``'onshore'``, ``'offshore'``, ``'solar'``.

    Index: CET/CEST naive datetime.
    Columns: ``da_forecast``, ``actual``
    """
    tag = {"onshore": "ONSHORE", "offshore": "OFFSHORE", "solar": "SOLAR"}[source.lower()]
    paths = sorted(
        Path(data_dir, "ElectricityPrices/DK1/Renewables").glob(
            f"*GENERATION_FORECAST_{tag}_*.csv"
        )
    )
    frames = []
    for p in paths:
        raw = pd.read_csv(
            p, quotechar='"', encoding="utf-8-sig", na_values=["n/e", " ", ""]
        )
        ts = _parse_mtu_start(_mtu_col(raw))
        df = pd.DataFrame(
            {
                "da_forecast": pd.to_numeric(raw["Day-ahead (MW)"], errors="coerce").values,
                "actual": pd.to_numeric(raw["Actual (MW)"], errors="coerce").values,
            },
            index=ts,
        )
        frames.append(df)
    return _concat_dedup(frames)


def load_transmission(data_dir: Path) -> pd.DataFrame:
    """DK1 cross-border physical flows in long format.

    Index: CET/CEST naive datetime.
    Columns: ``out_area``, ``in_area``, ``flow_mw``

    Use :func:`net_dk1_flows` to pivot into per-partner net-flow columns.
    """
    paths = sorted(
        Path(data_dir, "ElectricityPrices/DK1/Transmission").glob(
            "DK1_CROSS_BORDER_PHYSICAL_FLOWS_*.csv"
        )
    )
    frames = []
    for p in paths:
        raw = pd.read_csv(
            p, quotechar='"', encoding="utf-8-sig", na_values=["n/e", " ", ""],
            low_memory=False,
        )
        ts = _parse_mtu_start(_mtu_col(raw))
        df = pd.DataFrame(
            {
                "out_area": raw["Out Area"].str.replace("BZN|", "", regex=False).str.strip().values,
                "in_area": raw["In Area"].str.replace("BZN|", "", regex=False).str.strip().values,
                "flow_mw": pd.to_numeric(raw["Physical Flow (MW)"], errors="coerce").values,
            },
            index=ts,
        )
        frames.append(df)
    # Multiple rows per timestamp (one per country pair) — do not dedup
    return pd.concat(frames).sort_index()


def net_dk1_flows(transmission_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format transmission data into net DK1 export by partner (MW).

    A positive value means DK1 is a net exporter to that partner at that hour.
    Partner zones are derived from rows where DK1 appears as Out or In Area.
    """
    df = transmission_long.copy()
    dk1_out = df[df["out_area"] == "DK1"].copy()
    dk1_out["partner"] = dk1_out["in_area"]
    dk1_out["signed"] = dk1_out["flow_mw"]

    dk1_in = df[df["in_area"] == "DK1"].copy()
    dk1_in["partner"] = dk1_in["out_area"]
    dk1_in["signed"] = -dk1_in["flow_mw"]

    combined = pd.concat([dk1_out[["partner", "signed"]], dk1_in[["partner", "signed"]]])
    net = (
        combined.groupby([combined.index, "partner"])["signed"]
        .sum()
        .unstack("partner")
    )
    net.index.name = "datetime"
    return net.sort_index()


# ---------------------------------------------------------------------------
# Energy commodity loaders  (Investing.com format)
# ---------------------------------------------------------------------------

def load_commodity(data_dir: Path, name: str) -> pd.Series:
    """Daily closing price for an energy commodity.

    Parameters
    ----------
    name:
        One of ``'Natural_Gas'``, ``'EUA'``, ``'Coal'``, ``'Oil'``.

    Returns a Series with a daily DatetimeIndex.
    """
    path = Path(data_dir, "EnergySeries", f"{name}.csv")
    raw = pd.read_csv(path, encoding="utf-8-sig", thousands=",")
    dates = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
    price = pd.to_numeric(raw["Price"], errors="coerce")
    return pd.Series(price.values, index=dates, name=name).sort_index()


# ---------------------------------------------------------------------------
# Macro / financial loaders
# ---------------------------------------------------------------------------

def load_macro_daily(data_dir: Path, name: str) -> pd.Series:
    """Daily macro or financial series.

    Parameters
    ----------
    name:
        ``'VIX'``, ``'USD_EUR_Exchange'``, ``'MSCI_EU'``, ``'EuroStoxx50'``

    Handles both FRED (``observation_date`` column) and Investing.com formats.
    """
    path = Path(data_dir, "MacroFinance/Daily", f"{name}.csv")
    raw = pd.read_csv(path, encoding="utf-8-sig", thousands=",")

    if "observation_date" in raw.columns:
        dates = pd.to_datetime(raw["observation_date"])
        val_col = [c for c in raw.columns if c != "observation_date"][0]
        vals = pd.to_numeric(raw[val_col], errors="coerce")
    else:
        dates = pd.to_datetime(raw["Date"], format="%m/%d/%Y")
        vals = pd.to_numeric(raw["Price"], errors="coerce")

    return pd.Series(vals.values, index=dates, name=name).sort_index()


def load_macro_lower_freq(data_dir: Path, freq: str, name: str) -> pd.Series:
    """Monthly or quarterly macro series (FRED format).

    Parameters
    ----------
    freq:
        ``'Monthly'`` or ``'Quarterly'``
    name:
        Filename stem, e.g. ``'DK_CPI'``, ``'GDP_Denmark'``
    """
    path = Path(data_dir, "MacroFinance", freq, f"{name}.csv")
    raw = pd.read_csv(path, encoding="utf-8-sig")
    dates = pd.to_datetime(raw["observation_date"])
    val_col = [c for c in raw.columns if c != "observation_date"][0]
    vals = pd.to_numeric(raw[val_col], errors="coerce")
    return pd.Series(vals.values, index=dates, name=name).sort_index()


# ---------------------------------------------------------------------------
# Climate loader  (ECA&D gzipped CSV)
# ---------------------------------------------------------------------------

def load_climate(data_dir: Path, variable: str) -> pd.Series:
    """Daily DK1 climate observations from Esbjerg Airport (ECA&D).

    Parameters
    ----------
    variable:
        ``'fg'`` (mean wind speed, 0.1 m/s),
        ``'tg'`` (mean temperature, 0.1 °C),
        ``'ss'`` (sunshine duration, 0.1 h),
        ``'rr'`` (precipitation, 0.1 mm).

    Returns a daily Series.  Missing / quality-flagged values → NaN.
    """
    path = Path(data_dir, "Climate Variables/DK1", f"{variable}_DK1_2018_2024.csv.gz")
    raw = pd.read_csv(path, compression="gzip", parse_dates=["DATE"])
    vals = pd.to_numeric(raw[variable], errors="coerce")
    # ECA&D uses -9999 as the missing value sentinel
    vals = vals.replace(-9999, np.nan)
    return pd.Series(vals.values, index=raw["DATE"], name=variable).sort_index()
