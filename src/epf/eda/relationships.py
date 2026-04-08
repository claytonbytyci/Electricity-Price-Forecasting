"""
Cross-variable relationships: wind-price, cross-border flows, correlations.

Key literature findings encoded here:
- Nonlinear, threshold-dependent wind-price relationship (Jónsson et al. 2013)
- DK1↔Germany flow dominates cross-border effects (Liu et al. 2023)
- ~800 MW threshold on DK1-Germany interconnector (Liu et al. 2023)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Wind–price relationship
# ---------------------------------------------------------------------------

def plot_wind_price_scatter(
    prices: pd.Series,
    wind: pd.Series,
    resample: str = "H",
    ax: plt.Axes | None = None,
    hexbin: bool = True,
) -> plt.Axes:
    """Scatter / hexbin of wind generation vs price.

    Uses hexbin (density) by default to avoid overplotting on ~80k hourly obs.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    df = pd.concat([prices.rename("price"), wind.rename("wind")], axis=1).dropna()

    if hexbin:
        hb = ax.hexbin(df["wind"], df["price"], gridsize=50, cmap="Blues", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count")
    else:
        ax.scatter(df["wind"], df["price"], alpha=0.05, s=2, color="steelblue")

    # LOWESS-style moving median
    df_sorted = df.sort_values("wind")
    bins = pd.cut(df_sorted["wind"], bins=40)
    med = df_sorted.groupby(bins, observed=True)["price"].median()
    bin_mid = [interval.mid for interval in med.index]
    ax.plot(bin_mid, med.values, color="crimson", lw=2, label="Bin median")

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Wind generation (MW)")
    ax.set_ylabel("Day-ahead price (EUR/MWh)")
    ax.set_title("Wind generation vs. DK1 day-ahead price")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    return ax


def plot_wind_share_series(
    wind_onshore: pd.Series,
    wind_offshore: pd.Series,
    load: pd.Series,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Monthly average wind penetration (wind / load) over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    wind_total = wind_onshore.add(wind_offshore, fill_value=0)
    share = (wind_total / load).replace([np.inf, -np.inf], np.nan).dropna()
    monthly = share.resample("ME").mean().mul(100)

    ax.fill_between(monthly.index, monthly.values, alpha=0.4, color="steelblue")
    ax.plot(monthly.index, monthly.values, color="steelblue", lw=1.0)
    ax.axhline(49, color="crimson", lw=1, ls="--", label="49% (2014–15 avg, Becker & Blatt 2020)")
    ax.set_ylabel("Wind / Load  (%)")
    ax.set_title("Monthly average wind penetration in DK1")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    return ax


# ---------------------------------------------------------------------------
# Cross-border flows
# ---------------------------------------------------------------------------

def plot_net_flows(
    net_flows: pd.DataFrame,
    partners: list[str] | None = None,
    resample: str = "ME",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Monthly average net DK1 export by partner zone.

    A positive value means DK1 is a net exporter to that zone.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))

    priority = ["DE-LU", "DK2", "NO2", "SE3", "SE4", "GB"]
    cols = [c for c in priority if c in net_flows.columns]
    if partners:
        cols = [c for c in partners if c in net_flows.columns]
    if not cols:
        cols = net_flows.columns.tolist()

    monthly = net_flows[cols].resample(resample).mean()
    monthly.plot(ax=ax, lw=1.2)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_ylabel("Net export (MW, monthly avg)")
    ax.set_title("DK1 net cross-border flows by partner  (+ = DK1 exports)")
    ax.legend(title="Partner", fontsize=9, ncol=3)
    sns.despine(ax=ax)
    return ax


def plot_de_flow_price_scatter(
    prices: pd.Series,
    net_flows: pd.DataFrame,
    partner_col: str = "DE-LU",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """DK1↔Germany net flow vs. price, with the ~800 MW threshold annotated.

    Liu et al. (2023) report a threshold around 800 MW on this interconnector:
    below it, higher export raises DK1 prices; above it, the relationship reverses.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    if partner_col not in net_flows.columns:
        ax.set_title(f"Column '{partner_col}' not found in net_flows")
        return ax

    df = pd.concat(
        [prices.rename("price"), net_flows[partner_col].rename("flow")], axis=1
    ).dropna()

    hb = ax.hexbin(df["flow"], df["price"], gridsize=50, cmap="Blues", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count")

    # Bin median
    df_s = df.sort_values("flow")
    bins = pd.cut(df_s["flow"], bins=40)
    med = df_s.groupby(bins, observed=True)["price"].median()
    bin_mid = [i.mid for i in med.index]
    ax.plot(bin_mid, med.values, color="crimson", lw=2, label="Bin median")
    ax.axvline(800, color="orange", lw=1.5, ls="--", label="~800 MW threshold (Liu 2023)")

    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("Net DK1→Germany flow (MW)")
    ax.set_ylabel("Day-ahead price (EUR/MWh)")
    ax.set_title("DK1↔Germany flow vs. price")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    return ax


# ---------------------------------------------------------------------------
# Commodity prices
# ---------------------------------------------------------------------------

def plot_commodities(
    series_dict: dict[str, pd.Series],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Normalised commodity price time series on a single axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(13, 4))

    palette = sns.color_palette("tab10", len(series_dict))
    for (name, s), col in zip(series_dict.items(), palette):
        s_norm = s.dropna()
        s_norm = (s_norm - s_norm.mean()) / s_norm.std()
        ax.plot(s_norm.index, s_norm.values, lw=1.0, label=name, color=col)

    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.set_ylabel("Z-score")
    ax.set_title("Commodity prices (z-score normalised)")
    ax.legend(fontsize=9, ncol=4)
    sns.despine(ax=ax)
    return ax


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def plot_correlation_matrix(
    df: pd.DataFrame,
    method: str = "spearman",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Spearman (default) correlation heat-map of all columns in ``df``."""
    if ax is None:
        n = len(df.columns)
        size = max(6, n * 0.7)
        _, ax = plt.subplots(figsize=(size, size * 0.85))

    corr = df.corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.4,
        cbar_kws={"shrink": 0.8, "label": f"{method.title()} ρ"},
        square=True,
    )
    ax.set_title(f"{method.title()} correlation matrix")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    return ax


def build_daily_feature_matrix(
    prices: pd.Series,
    wind_onshore: pd.Series,
    wind_offshore: pd.Series,
    solar: pd.Series,
    load: pd.Series,
    net_flows: pd.DataFrame,
    gas: pd.Series,
    eua: pd.Series,
    wind_speed: pd.Series | None = None,
) -> pd.DataFrame:
    """Assemble a daily-averaged feature matrix for correlation analysis.

    All hourly series are averaged to daily frequency before joining.
    """
    partners = ["DE-LU", "DK2", "NO2", "SE3"]
    flow_cols = [c for c in partners if c in net_flows.columns]

    hourly = pd.concat(
        [
            prices.rename("price"),
            wind_onshore.rename("wind_onshore"),
            wind_offshore.rename("wind_offshore"),
            solar.rename("solar"),
            load.rename("load"),
            net_flows[flow_cols].rename(columns=lambda c: f"flow_{c}"),
        ],
        axis=1,
    )
    daily_elec = hourly.resample("D").mean()

    daily_comm = pd.concat(
        [gas.rename("gas_price"), eua.rename("eua_price")], axis=1
    ).resample("D").last()

    df = daily_elec.join(daily_comm, how="left")

    if wind_speed is not None:
        df = df.join(wind_speed.rename("wind_speed_obs").resample("D").mean(), how="left")

    # Derived: total wind
    if "wind_onshore" in df.columns and "wind_offshore" in df.columns:
        df["wind_total"] = df["wind_onshore"] + df["wind_offshore"]

    return df.dropna(how="all")
