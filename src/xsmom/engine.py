"""Vectorized multi-asset backtest engine.

Invariant: position[t] equals weights[t-1]. The engine owns the shift.
Portfolio construction returns target weights dated at the rebalance bar;
those weights take effect on the next bar. Do not shift inside a strategy
or a weighting function.

Between rebalance dates the engine holds weights constant, which is
equivalent to rebalancing back to target every day. The implicit daily
turnover from ignoring weight drift is not charged. For monthly-rebalanced
decile portfolios this is a second-order effect; it matters more at higher
leverage or lower rebalance frequency.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame  # position actually held, after the shift
    turnover: pd.Series
    costs: pd.Series
    long_exposure: pd.Series
    short_exposure: pd.Series


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
    borrow_bps_annual: float = 0.0,
    initial_capital: float = 1.0,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run a portfolio backtest from target weights.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted close, dates x tickers. May contain NaN before listing or
        after delisting.
    weights : pd.DataFrame
        Target weights at rebalance dates (a subset of `prices.index`),
        dates x tickers. Held constant until the next rebalance date.
    cost_bps, slippage_bps : float
        Charged on per-bar turnover, `sum(abs(dw)) * bps / 10_000`.
    borrow_bps_annual : float
        Charged per bar on gross short exposure.

    Notes
    -----
    A NaN return on a bar where the book holds the name is treated as a
    zero return. That silently absorbs delisting gaps; `data.py` reports
    how much of the book this affects so the number can be quoted.
    """
    weights = weights.reindex(index=prices.index, columns=prices.columns)
    weights = weights.ffill().fillna(0.0)

    returns = prices.pct_change(fill_method=None)

    position = weights.shift(1).fillna(0.0)

    gross_pnl = (position * returns.fillna(0.0)).sum(axis=1)

    turnover = position.diff().abs().sum(axis=1)
    turnover.iloc[0] = position.iloc[0].abs().sum()

    trade_costs = turnover * (cost_bps + slippage_bps) / 10_000.0

    long_exposure = position.clip(lower=0.0).sum(axis=1)
    short_exposure = (-position).clip(lower=0.0).sum(axis=1)
    borrow_cost = short_exposure * (borrow_bps_annual / 10_000.0) / periods_per_year

    net_pnl = gross_pnl - trade_costs - borrow_cost
    equity = (1.0 + net_pnl).cumprod() * initial_capital

    return BacktestResult(
        equity=equity,
        returns=net_pnl,
        weights=position,
        turnover=turnover,
        costs=trade_costs + borrow_cost,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
    )
