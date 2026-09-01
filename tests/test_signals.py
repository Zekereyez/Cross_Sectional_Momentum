import numpy as np
import pandas as pd
import pytest

from xsmom.signals import momentum, month_end_prices


def test_momentum_known_value():
    # Price doubles every month: P_t = 2^t.
    idx = pd.date_range("2020-01-31", periods=15, freq="ME")
    prices = pd.DataFrame({"A": 2.0 ** np.arange(15)}, index=idx)

    mom = momentum(prices, lookback=12, skip=1)

    # P[t-1] / P[t-12] - 1 = 2^11 - 1, once 12 months of history exist.
    assert mom["A"].iloc[12] == pytest.approx(2.0**11 - 1.0)
    assert mom["A"].iloc[:12].isna().all()


def test_skip_month_is_excluded():
    idx = pd.date_range("2020-01-31", periods=14, freq="ME")
    prices = pd.DataFrame({"A": np.ones(14)}, index=idx)
    # A huge move in the most recent month must not touch the 12-1 signal.
    prices.iloc[-1] = 100.0

    mom = momentum(prices, lookback=12, skip=1)
    assert mom["A"].iloc[-1] == pytest.approx(0.0)


def test_lookback_must_exceed_skip():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    prices = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=idx)
    with pytest.raises(ValueError):
        momentum(prices, lookback=1, skip=1)


def test_month_end_uses_actual_trading_days():
    daily = pd.DataFrame(
        {"A": np.arange(60, dtype=float)},
        index=pd.bdate_range("2020-01-01", periods=60),
    )
    monthly = month_end_prices(daily)

    assert all(d in daily.index for d in monthly.index)
    # Last business day of Jan 2020 is Fri Jan 31.
    assert monthly.index[0] == pd.Timestamp("2020-01-31")
