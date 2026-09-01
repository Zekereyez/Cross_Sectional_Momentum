"""Hand-computed checks on the portfolio engine."""
import pandas as pd
import pytest

from xsmom.engine import run_backtest


def make_case():
    dates = pd.bdate_range("2020-01-01", periods=3)
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [100.0, 90.0, 81.0]}, index=dates
    )
    weights = pd.DataFrame(
        {"A": [1.0], "B": [-1.0]}, index=[dates[0]]
    )
    return prices, weights


def test_shift_and_gross_pnl():
    prices, weights = make_case()
    result = run_backtest(prices, weights)

    # Weights set on day 0 take effect day 1: long A +10%, short B -(-10%).
    assert result.returns.iloc[0] == 0.0
    assert result.returns.iloc[1] == pytest.approx(0.20)
    assert result.returns.iloc[2] == pytest.approx(0.20)
    assert result.equity.iloc[2] == pytest.approx(1.2 * 1.2)


def test_costs_charged_on_turnover():
    prices, weights = make_case()
    result = run_backtest(prices, weights, cost_bps=10.0)

    # Day 1 turnover is 2 (0 -> +1 and 0 -> -1); 10 bps on it is 0.002.
    assert result.turnover.iloc[1] == pytest.approx(2.0)
    assert result.returns.iloc[1] == pytest.approx(0.20 - 2.0 * 10.0 / 10_000.0)
    # No trades after day 1, so no further trade costs.
    assert result.costs.iloc[2] == pytest.approx(0.0)


def test_borrow_on_short_side_only():
    prices, weights = make_case()
    result = run_backtest(prices, weights, borrow_bps_annual=252.0 * 100.0)

    # 25200 bps annual / 252 periods = 1% per bar on short exposure of 1.
    assert result.short_exposure.iloc[1] == pytest.approx(1.0)
    assert result.returns.iloc[1] == pytest.approx(0.20 - 0.01)


def test_nan_returns_treated_as_zero():
    prices, weights = make_case()
    prices.loc[prices.index[2], "B"] = float("nan")
    result = run_backtest(prices, weights)

    # B's day-2 return is unknown; the engine books zero for it.
    assert result.returns.iloc[2] == pytest.approx(0.10)
