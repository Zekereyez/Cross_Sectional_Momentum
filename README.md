# Cross_Sectional_Momentum

Cross-sectional 12-1 momentum on the S&P 500 with a point-in-time universe, realistic frictions, and factor-level evaluation. A replication of Jegadeesh & Titman (1993) on modern data, built to measure how much of the published anomaly survives costs and honest universe construction.

Companion project to [vector-backtester](https://github.com/Zekereyez/vector-backtester), which handles one asset at a time. This engine handles a book.

## Status

Infrastructure is built and tested. The research — the actual point of the repo — is not done yet. See "Work to be done" below. Numbers will appear here only after the coverage caveats can be quoted next to them.

## The strategy

At each month-end, rank index members by their total return from 12 months ago to 1 month ago (the skip-month removes short-term reversal). Go long the top decile equal-weighted, short the bottom decile equal-weighted. Dollar neutral, gross exposure 2. Rebalance monthly.

## The universe problem

Backtesting on today's S&P 500 members inflates momentum results: the losers got deleted. This repo reconstructs approximate point-in-time membership by walking Wikipedia's dated add/remove log backwards from the current constituent list (`universe.py`).

That fixes **membership** bias but not **price** bias: yfinance has no data for most delisted tickers, so names that left the index by dying often cannot be traded in the backtest even though the universe knows they were members. Delisted names are disproportionately losers, so the short leg's returns are understated by construction. `data.coverage_report` measures the gap per rebalance date; every result quoted in this README must carry that number next to it.

## Install

```
git clone https://github.com/Zekereyez/Cross_Sectional_Momentum
cd Cross_Sectional_Momentum
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```
python examples/run_momentum.py
```

First run downloads ~20 years of prices for every ticker that was ever a member in the window (slow); everything caches under `data/cache/`. Writes `results/tearsheet.png` and `results/coverage.csv`, prints the summary stats, the mean monthly IC with its t-stat, and the price-coverage numbers.

## Run the tests

```
pytest
```

`tests/test_lookahead.py` pins the no-lookahead invariant across the entire pipeline: perturbing every price after a cutoff date must leave equity before the cutoff bit-identical. Do not trust any result the repo produces until it passes.

## Layout

```
src/xsmom/
  universe.py    # point-in-time S&P 500 membership from Wikipedia's change log
  data.py        # multi-ticker yfinance loader with parquet cache + coverage report
  signals.py     # month-end resampling, 12-1 momentum
  portfolio.py   # rank -> decile -> equal-weight long/short weights
  engine.py      # multi-asset vectorized backtest, owns the one-bar shift
  metrics.py     # strategy metrics (Sharpe, CAGR, DD) and factor metrics (IC, quantile returns)
  tearsheet.py   # 6-panel matplotlib report
tests/
  test_engine.py     # hand-computed PnL, cost, and borrow checks
  test_lookahead.py  # the invariant that makes everything else trustworthy
  test_signals.py    # known-value momentum, skip-month exclusion
  test_portfolio.py  # dollar neutrality, book sums, membership masking
  test_universe.py   # membership reconstruction, no network
examples/
  run_momentum.py    # end-to-end run, writes results/
```

## Design notes

**The shift invariant.** Weights dated at rebalance bar `t` take effect on bar `t+1`. The engine applies exactly one bar of shift; nothing else in the pipeline shifts. Same rule as vector-backtester, now enforced across a whole book.

**Two evaluation layers.** `metrics.py` deliberately separates strategy metrics (evaluate a portfolio) from factor metrics (evaluate a signal). IC and quantile returns diagnose whether the *ranking* predicts returns before portfolio construction choices — decile width, weighting scheme, costs — contaminate the read. `forward_returns` is the one function in the repo that touches future data on purpose; it exists to be correlated with the signal, never traded on.

**Rebalance drift.** Between month-ends the engine holds weights constant, which implicitly rebalances to target daily without charging the drift turnover. Second-order at this leverage and frequency; documented rather than modeled.

**Missing returns.** A NaN return on a held name books as zero. This silently absorbs delisting gaps, which is exactly where momentum shorts make their money — another reason the short leg is understated and the coverage report is mandatory context.

## Work to be done

Roughly in order:

1. **Baseline run and writeup.** Run the pipeline 2005–2024, report long/short and long-only against SPY, with coverage numbers attached. Compare the decile spread to Jegadeesh & Titman's published magnitudes.
2. **Quantify the survivorship gap.** Rerun on the static current-members universe and difference the results. That difference *is* the bias; publishing it is more interesting than pretending it away.
3. **Beta profile and hedging.** Measure rolling beta of the long/short book (it is not zero). Add a beta hedge via an index overlay and see what it does to Sharpe and to the 2009-style momentum crash.
4. **Momentum crashes.** Reproduce Daniel & Moskowitz (2016) on this data: what happens to the short leg in the March 2009 and April 2020 rebounds.
5. **Turnover/alpha frontier.** Sweep rebalance frequency and decile width; find where costs eat the signal.
6. **IC decay.** Compute IC against 1-, 3-, 6-, 12-month forward returns to measure how fast the signal fades, which bounds how slow trading can be.
7. **Statistical honesty.** Deflated Sharpe ratio (Bailey & López de Prado) on the final configuration, accounting for every configuration tried along the way.

## Prior art

- Jegadeesh, N. & Titman, S. (1993), *Returns to Buying Winners and Selling Losers* — the 12-1 signal this replicates.
- Daniel, K. & Moskowitz, T. (2016), *Momentum Crashes* — why the short leg blows up in rebounds.
- Bailey, D. & López de Prado, M. (2014), *The Deflated Sharpe Ratio* — the multiple-testing correction step 7 applies.

## Known limits

- Membership before ~2000 from Wikipedia's change log is unreliable; the study window starts 2005.
- Price data is yfinance; point-in-time correctness of adjusted closes is not guaranteed, and delisted tickers are mostly absent (see "The universe problem").
- Costs are linear in notional. Fine for the ETF-scale question; wrong for institutional size.
- Ticker symbol changes over time are not mapped; a name that changed symbols looks like a delisting followed by a new listing.
