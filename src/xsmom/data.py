"""Multi-ticker price loading with a parquet cache, plus the coverage
report that quantifies price survivorship bias."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_CACHE = Path("data/cache")


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Adjusted close for `tickers`, dates x tickers.

    Tickers yfinance knows nothing about come back as all-NaN columns and
    are kept, not dropped: the coverage report needs to count them, and
    dropping them silently would hide exactly the bias this project is
    supposed to measure.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(
        ("|".join(sorted(tickers)) + start + end).encode()
    ).hexdigest()[:16]
    cache_path = cache_dir / f"prices_{key}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = yf.download(
        sorted(set(tickers)),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.reindex(columns=sorted(set(tickers)))
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    prices.to_parquet(cache_path)
    return prices


def coverage_report(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Per rebalance date: how much of the point-in-time universe do we
    actually have prices for?

    The shortfall is the price survivorship bias left in the study after
    fixing membership. It concentrates in names that left the index by
    delisting - disproportionately losers - so the momentum short leg is
    understated by construction. Quote `pct_priced` in the README next to
    every result.
    """
    rows = []
    for dt in membership.index:
        members = membership.columns[membership.loc[dt]]
        if dt not in prices.index or len(members) == 0:
            continue
        have = prices.loc[dt, prices.columns.intersection(members)].notna().sum()
        rows.append({"date": dt, "members": len(members), "priced": int(have)})
    out = pd.DataFrame(rows).set_index("date")
    out["pct_priced"] = out["priced"] / out["members"]
    return out
