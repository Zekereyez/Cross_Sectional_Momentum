"""End-to-end 12-1 momentum run on the S&P 500.

Downloads take a while on the first run; everything is cached under
data/cache/ afterwards. Writes results/tearsheet.png and prints the
summary table plus the price-coverage numbers that must accompany any
quoted result.
"""
from pathlib import Path

import pandas as pd

from xsmom.data import coverage_report, load_prices
from xsmom.engine import run_backtest
from xsmom.metrics import forward_returns, information_coefficient, quantile_returns, summary
from xsmom.portfolio import quantile_weights
from xsmom.signals import momentum, month_end_prices
from xsmom.tearsheet import make_tearsheet
from xsmom.universe import membership_mask

START, END = "2005-01-01", "2024-12-31"
COST_BPS, SLIPPAGE_BPS, BORROW_BPS = 5.0, 5.0, 50.0

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# Rebalance dates: last trading day of each month, taken from SPY's calendar.
spy = load_prices(["SPY"], START, END)["SPY"]
rebalance_dates = month_end_prices(spy.to_frame())["SPY"].index

print("Reconstructing point-in-time membership...")
membership = membership_mask(rebalance_dates)
tickers = list(membership.columns)
print(f"{len(tickers)} tickers were index members at some point in the window")

print("Loading prices (first run downloads; later runs hit the cache)...")
prices = load_prices(tickers, START, END)

coverage = coverage_report(prices, membership)
print(
    f"Price coverage of the point-in-time universe: "
    f"mean {coverage['pct_priced'].mean():.1%}, min {coverage['pct_priced'].min():.1%}"
)
coverage.to_csv(results_dir / "coverage.csv")

monthly = month_end_prices(prices)
signal = momentum(monthly, lookback=12, skip=1)
weights = quantile_weights(signal, n_quantiles=10, long_short=True, membership=membership)

result = run_backtest(
    prices, weights,
    cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS, borrow_bps_annual=BORROW_BPS,
)

fwd = forward_returns(monthly, periods=1)
ic = information_coefficient(signal, fwd)
q_rets = quantile_returns(signal, fwd, n_quantiles=10)

benchmark_equity = (1.0 + spy.pct_change().fillna(0.0)).cumprod()

make_tearsheet(
    result, ic, q_rets,
    benchmark_equity=benchmark_equity,
    out_path=str(results_dir / "tearsheet.png"),
    title="S&P 500 12-1 momentum, decile long/short",
)

stats = summary(result.equity, result.returns, result.turnover)
print(pd.Series(stats).round(4).to_string())
print(f"Mean monthly IC: {ic.mean():.4f}  (t-stat {ic.mean() / ic.std() * len(ic) ** 0.5:.2f})")
print(f"Tearsheet written to {results_dir / 'tearsheet.png'}")
