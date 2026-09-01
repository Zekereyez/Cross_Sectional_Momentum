"""Rank a signal into quantile portfolio weights."""
from __future__ import annotations

import pandas as pd


def quantile_weights(
    signal: pd.DataFrame,
    n_quantiles: int = 10,
    long_short: bool = True,
    membership: pd.DataFrame | None = None,
    min_names: int | None = None,
) -> pd.DataFrame:
    """Equal-weight the top quantile long and (optionally) the bottom
    quantile short.

    Long book sums to +1, short book to -1, so the long/short portfolio is
    dollar neutral with gross exposure 2. It is NOT beta neutral; momentum
    long/short carries large time-varying beta, which is one of the things
    this project exists to measure.

    Parameters
    ----------
    signal : pd.DataFrame
        Rebalance dates x tickers. NaN means "not rankable on this date".
    membership : pd.DataFrame, optional
        Boolean mask, same shape semantics as `signal`. Names outside the
        index on the formation date are excluded before ranking.
    min_names : int, optional
        Minimum count of rankable names required to form a portfolio;
        below it the date gets zero weights. Defaults to `2 * n_quantiles`.
    """
    if min_names is None:
        min_names = 2 * n_quantiles

    if membership is not None:
        mask = membership.reindex_like(signal).fillna(False).astype(bool)
        signal = signal.where(mask)

    def one_date(row: pd.Series) -> pd.Series:
        w = pd.Series(0.0, index=row.index)
        valid = row.dropna()
        if len(valid) < min_names:
            return w
        buckets = pd.qcut(valid.rank(method="first"), n_quantiles, labels=False)
        top = valid.index[buckets == n_quantiles - 1]
        w[top] = 1.0 / len(top)
        if long_short:
            bottom = valid.index[buckets == 0]
            w[bottom] = -1.0 / len(bottom)
        return w

    return signal.apply(one_date, axis=1)
