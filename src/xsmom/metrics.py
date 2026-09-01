"""Performance metrics and factor-evaluation statistics.

Two layers, deliberately separate:

- Strategy metrics (Sharpe, CAGR, drawdown) evaluate a *portfolio*.
- Factor metrics (IC, quantile returns) evaluate a *signal*, before any
  portfolio construction choices contaminate the read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------- strategy

def cagr(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    years = len(equity) / periods_per_year
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    sd = returns.std()
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(returns.mean() / sd * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1.0
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def annual_turnover(turnover: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(turnover.mean() * periods_per_year)


def summary(
    equity: pd.Series,
    returns: pd.Series,
    turnover: pd.Series | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> dict[str, float]:
    out = {
        "cagr": cagr(equity, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "volatility": float(returns.std() * np.sqrt(periods_per_year)),
    }
    if turnover is not None:
        out["annual_turnover"] = annual_turnover(turnover, periods_per_year)
    return out


# ------------------------------------------------------------------ factor

def forward_returns(monthly_prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Return over the `periods` months AFTER each row's date.

    This is the one place in the repo where future data is used on
    purpose: it exists to be correlated with the signal, never to be
    traded on. Keep it out of anything the engine sees.
    """
    return monthly_prices.pct_change(periods, fill_method=None).shift(-periods)


def information_coefficient(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    min_names: int = 20,
) -> pd.Series:
    """Per-date Spearman rank correlation between signal and forward return."""
    fwd = fwd.reindex_like(signal)
    ics = {}
    for dt in signal.index:
        s, f = signal.loc[dt], fwd.loc[dt]
        mask = s.notna() & f.notna()
        if mask.sum() >= min_names:
            # Spearman = Pearson on average ranks; avoids the scipy dep.
            ics[dt] = s[mask].rank().corr(f[mask].rank())
    return pd.Series(ics, dtype=float)


def quantile_returns(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    n_quantiles: int = 10,
) -> pd.Series:
    """Mean forward return per signal quantile, pooled across dates.

    A signal that works produces a monotone staircase from quantile 1
    (losers) to quantile `n_quantiles` (winners). A signal that only
    "works" at the extremes, or is flat in the middle, shows up here
    before it shows up in an equity curve.
    """
    fwd = fwd.reindex_like(signal)
    rows = []
    for dt in signal.index:
        s, f = signal.loc[dt], fwd.loc[dt]
        mask = s.notna() & f.notna()
        if mask.sum() < 2 * n_quantiles:
            continue
        buckets = pd.qcut(s[mask].rank(method="first"), n_quantiles, labels=False)
        rows.append(f[mask].groupby(buckets).mean())
    if not rows:
        return pd.Series(dtype=float)
    pooled = pd.concat(rows, axis=1).mean(axis=1)
    pooled.index = pooled.index + 1  # quantiles as 1..n
    pooled.index.name = "quantile"
    return pooled
