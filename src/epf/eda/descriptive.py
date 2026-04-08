"""
Descriptive statistics and distribution plots for DK1 electricity prices.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def summary_stats(prices: pd.Series) -> pd.DataFrame:
    """Return a descriptive statistics table augmented with EPF-specific metrics.

    Parameters
    ----------
    prices:
        Hourly price series (EUR/MWh).

    Returns
    -------
    DataFrame with one column per year plus an 'All' column.
    """
    p = prices.dropna()
    years = p.index.year.unique().sort_values()

    rows: dict[str, list] = {
        "N (obs)": [],
        "Mean": [],
        "Median": [],
        "Std": [],
        "Min": [],
        "Max": [],
        "% negative": [],
        "% > 200 EUR/MWh": [],
        "Skewness": [],
        "Kurtosis": [],
    }

    def _stats(s: pd.Series) -> None:
        rows["N (obs)"].append(len(s))
        rows["Mean"].append(round(s.mean(), 2))
        rows["Median"].append(round(s.median(), 2))
        rows["Std"].append(round(s.std(), 2))
        rows["Min"].append(round(s.min(), 2))
        rows["Max"].append(round(s.max(), 2))
        rows["% negative"].append(round(100 * (s < 0).mean(), 1))
        rows["% > 200 EUR/MWh"].append(round(100 * (s > 200).mean(), 1))
        rows["Skewness"].append(round(s.skew(), 2))
        rows["Kurtosis"].append(round(s.kurtosis(), 2))

    cols = []
    for yr in years:
        _stats(p[p.index.year == yr])
        cols.append(str(yr))
    _stats(p)
    cols.append("All")

    return pd.DataFrame(rows, index=cols).T


def plot_price_series(
    prices: pd.Series,
    resample: str = "D",
    ax: plt.Axes | None = None,
    crisis_shade: bool = True,
) -> plt.Axes:
    """Time series of DK1 day-ahead prices.

    Parameters
    ----------
    prices:
        Hourly price series.
    resample:
        Pandas offset alias for resampling before plotting (``'D'``, ``'W'``, ``'ME'``).
    crisis_shade:
        If True, shade the 2021-2022 energy crisis period.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4))

    daily = prices.resample(resample).mean()
    ax.plot(daily.index, daily.values, lw=0.8, color="steelblue")
    ax.axhline(0, color="k", lw=0.5, ls="--")

    if crisis_shade:
        ax.axvspan(
            pd.Timestamp("2021-07-01"),
            pd.Timestamp("2023-01-01"),
            alpha=0.08,
            color="orange",
            label="Energy crisis 2021–22",
        )
        ax.legend(fontsize=9)

    ax.set_xlabel("")
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"DK1 Day-Ahead Price ({resample}-averaged)")
    sns.despine(ax=ax)
    return ax


def plot_price_distribution(
    prices: pd.Series,
    bins: int = 100,
    ax: plt.Axes | None = None,
    clip: tuple[float, float] = (-100, 500),
) -> plt.Axes:
    """Histogram + KDE of the price distribution, clipped for readability."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    p = prices.clip(*clip).dropna()
    sns.histplot(p, bins=bins, kde=True, ax=ax, color="steelblue", alpha=0.6, stat="density")
    ax.axvline(0, color="crimson", lw=1.2, ls="--", label="Zero")
    ax.axvline(p.mean(), color="navy", lw=1.2, ls=":", label=f"Mean ({p.mean():.1f})")
    ax.set_xlabel("EUR/MWh")
    ax.set_title("Price Distribution (clipped)")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    return ax


def plot_annual_boxplots(prices: pd.Series, ax: plt.Axes | None = None) -> plt.Axes:
    """Box plots of hourly prices by calendar year, showing the energy-crisis shift."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    df = prices.dropna().rename("price").to_frame()
    df["year"] = df.index.year
    sns.boxplot(
        data=df,
        x="year",
        y="price",
        ax=ax,
        showfliers=False,
        palette="Blues",
        whis=[5, 95],
    )
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Annual price distribution (whiskers = 5th–95th pct)")
    sns.despine(ax=ax)
    return ax


def plot_negative_prices(prices: pd.Series, ax: plt.Axes | None = None) -> plt.Axes:
    """Annual share of negative-price hours — a key DK1 / high-wind market feature."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))

    annual = (
        prices.dropna()
        .to_frame("price")
        .assign(year=lambda d: d.index.year, neg=lambda d: d["price"] < 0)
        .groupby("year")["neg"]
        .mean()
        .mul(100)
    )
    annual.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_ylabel("% of hours")
    ax.set_xlabel("")
    ax.set_title("Annual share of negative-price hours (%)")
    ax.tick_params(axis="x", rotation=0)
    sns.despine(ax=ax)
    return ax
