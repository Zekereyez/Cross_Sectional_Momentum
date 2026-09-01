"""Momentum signal construction on month-end prices."""
from __future__ import annotations

import pandas as pd


def month_end_prices(daily: pd.DataFrame) -> pd.DataFrame:
    """Last observed price in each calendar month, indexed by the actual
    trading date so the result aligns with the daily index downstream."""
    last_days = daily.index.to_series().resample("ME").last().dropna()
    return daily.loc[last_days.values]


def momentum(
    monthly_prices: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
) -> pd.DataFrame:
    """Classic momentum: total return from t-lookback to t-skip, in months.

    The default (12, 1) is Jegadeesh & Titman's 12-1 signal: twelve-month
    return skipping the most recent month, which is dominated by
    short-term reversal.

    The signal at row t uses only prices at or before t. The engine owns
    the trading shift; do not shift here.
    """
    if lookback <= skip:
        raise ValueError(f"lookback ({lookback}) must exceed skip ({skip})")
    return monthly_prices.shift(skip) / monthly_prices.shift(lookback) - 1.0
