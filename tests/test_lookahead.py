"""Pin the no-lookahead invariant across the whole pipeline.

Perturb every price strictly after a cutoff date and rerun the full
signal -> weights -> engine chain. Equity up to the cutoff must be
bit-identical. If this test fails, the pipeline reads the future; do not
trust any result the repo produces until it passes.
"""
import numpy as np
import pandas as pd

from xsmom.engine import run_backtest
from xsmom.portfolio import quantile_weights
from xsmom.signals import momentum, month_end_prices

N_TICKERS = 40


def random_prices(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", "2022-12-31")
    rets = rng.normal(0.0003, 0.02, size=(len(dates), N_TICKERS))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=dates, columns=[f"T{i}" for i in range(N_TICKERS)])


def run_pipeline(prices: pd.DataFrame) -> pd.Series:
    monthly = month_end_prices(prices)
    signal = momentum(monthly, lookback=12, skip=1)
    weights = quantile_weights(signal, n_quantiles=5, long_short=True)
    return run_backtest(prices, weights, cost_bps=5.0, borrow_bps_annual=50.0).equity


def test_future_prices_cannot_change_the_past():
    prices = random_prices()
    cutoff = pd.Timestamp("2021-06-30")

    base = run_pipeline(prices)

    rng = np.random.default_rng(99)
    perturbed = prices.copy()
    future = perturbed.index > cutoff
    perturbed.loc[future] *= rng.uniform(0.5, 2.0, size=(future.sum(), N_TICKERS))

    changed = run_pipeline(perturbed)

    pd.testing.assert_series_equal(base.loc[:cutoff], changed.loc[:cutoff])
    # Sanity: the perturbation actually changed the future.
    assert not base.loc[cutoff:].equals(changed.loc[cutoff:])
