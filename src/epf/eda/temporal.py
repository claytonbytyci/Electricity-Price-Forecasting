"""
Temporal patterns: seasonality profiles and autocorrelation structure.

The autocorrelation at lags 1 h, 24 h, and 168 h (1 week) is central to the
EPF literature — these are the canonical lags used in LEAR and LASSO models.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ---------------------------------------------------------------------------
# Seasonality profiles
# ---------------------------------------------------------------------------

def plot_hourly_profile(prices: pd.Series, ax: plt.Axes | None = None) -> plt.Axes:
    """Median price by hour-of-day with IQR band."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    df = prices.dropna().rename("price").to_frame()
    df["hour"] = df.index.hour
    grp = df.groupby("hour")["price"]
    med = grp.median()
    q25 = grp.quantile(0.25)
    q75 = grp.quantile(0.75)

    ax.fill_between(med.index, q25, q75, alpha=0.25, color="steelblue", label="IQR")
    ax.plot(med.index, med.values, color="steelblue", lw=1.8, label="Median")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (CET)")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Intra-day price profile")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    return ax


def plot_dow_profile(prices: pd.Series, ax: plt.Axes | None = None) -> plt.Axes:
    """Median price by day of week (Monday=0)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    df = prices.dropna().rename("price").to_frame()
    df["dow"] = df.index.dayofweek
    grp = df.groupby("dow")["price"]
    med = grp.median()
    q25 = grp.quantile(0.25)
    q75 = grp.quantile(0.75)

    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.fill_between(range(7), q25, q75, alpha=0.25, color="steelblue")
    ax.plot(range(7), med.values, color="steelblue", lw=1.8, marker="o", ms=4)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xticks(range(7))
    ax.set_xticklabels(labels)
    ax.set_ylabel("EUR/MWh  (median)")
    ax.set_title("Day-of-week price profile")
    sns.despine(ax=ax)
    return ax


def plot_monthly_profile(prices: pd.Series, ax: plt.Axes | None = None) -> plt.Axes:
    """Median price and IQR by calendar month."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    df = prices.dropna().rename("price").to_frame()
    df["month"] = df.index.month
    grp = df.groupby("month")["price"]
    med = grp.median()
    q25 = grp.quantile(0.25)
    q75 = grp.quantile(0.75)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.fill_between(range(1, 13), q25, q75, alpha=0.25, color="steelblue")
    ax.plot(range(1, 13), med.values, color="steelblue", lw=1.8, marker="o", ms=4)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(months)
    ax.set_ylabel("EUR/MWh  (median)")
    ax.set_title("Monthly price profile")
    sns.despine(ax=ax)
    return ax


def plot_heatmap_hour_month(prices: pd.Series, ax: plt.Axes | None = None) -> plt.Axes:
    """Median price heatmap: hours (rows) × months (columns)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    df = prices.dropna().rename("price").to_frame()
    pivot = df.assign(hour=df.index.hour, month=df.index.month).pivot_table(
        values="price", index="hour", columns="month", aggfunc="median"
    )
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn_r",
        center=0,
        fmt=".0f",
        annot=True,
        annot_kws={"size": 7},
        linewidths=0.3,
        cbar_kws={"label": "EUR/MWh"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("Hour of day")
    ax.set_title("Median price by hour × month (EUR/MWh)")
    return ax


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def plot_acf_key_lags(prices: pd.Series, max_lag: int = 200) -> plt.Figure:
    """ACF and PACF up to max_lag, with key EPF lags annotated.

    Lags 1, 24, and 168 are the canonical LEAR regressors (Lago et al. 2021).
    """
    p = prices.dropna()
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    plot_acf(p, lags=max_lag, ax=axes[0], alpha=0.05, zero=False, color="steelblue")
    plot_pacf(p, lags=max_lag, ax=axes[1], alpha=0.05, zero=False, color="steelblue",
              method="ywm")

    for ax, label in zip(axes, ["ACF", "PACF"]):
        for lag, name in [(1, "1h"), (24, "24h"), (168, "168h")]:
            if lag <= max_lag:
                ax.axvline(lag, color="crimson", lw=1.0, ls="--", alpha=0.7)
                ax.text(lag + 1, ax.get_ylim()[1] * 0.85, name, color="crimson", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f"{label} of DK1 day-ahead price")
        sns.despine(ax=ax)

    axes[1].set_xlabel("Lag (hours)")
    fig.tight_layout()
    return fig


def plot_acf_manual(prices: pd.Series, lags: list[int] | None = None) -> plt.Figure:
    """Bar chart of autocorrelation at selected lags (default: 1–48 + 168)."""
    p = prices.dropna()
    if lags is None:
        lags = list(range(1, 49)) + [168]

    acf_vals = [p.autocorr(lag=lag) for lag in lags]
    labels = [str(l) if l != 168 else "168\n(1wk)" for l in lags]

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["crimson" if l in (1, 24, 168) else "steelblue" for l in lags]
    ax.bar(range(len(lags)), acf_vals, color=colors, width=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(lags)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Price autocorrelation at selected lags  (red = canonical LEAR lags)")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig
