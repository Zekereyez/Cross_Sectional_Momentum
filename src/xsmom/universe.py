"""Point-in-time S&P 500 membership reconstructed from Wikipedia.

Wikipedia's constituents page carried two tables: the current members and
a dated change log of additions/removals going back to the 1990s. The
change log was removed from the live page after mid-2025, so this module
reads a pinned historical revision through the MediaWiki API. Both tables
come from the same revision, so "current" means "members on ANCHOR_DATE"
and `members_asof` is only valid for dates at or before it. Extending the
study window past ANCHOR_DATE requires a newer membership source.

Walking the change log backwards from the anchor reconstructs approximate
historical membership. This removes *membership* survivorship bias, but
not *price* survivorship bias: yfinance has no data for most delisted
tickers, so names that left the index by dying often can't be traded in
the backtest even though the universe knows about them. `data.py`
measures that gap; the README must quote it.

Known dirt in the source: tickers change symbols over time, some change
rows are missing one side, and pre-2000 coverage thins out. Treat
membership before ~2000 as unreliable.
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

_API_URL = "https://en.wikipedia.org/w/api.php"
# Last-known-good revision of List_of_S&P_500_companies that still
# carries the change-log table.
_REVISION_ID = 1292523673
ANCHOR_DATE = pd.Timestamp("2025-05-27")
_UA = {"User-Agent": "xsmom-research/0.1 (github.com/Zekereyez/Cross_Sectional_Momentum)"}

DEFAULT_CACHE = Path("data/cache")


def normalize_ticker(ticker: str) -> str:
    """Wikipedia uses BRK.B; yfinance wants BRK-B."""
    return ticker.strip().upper().replace(".", "-")


def fetch_sp500_tables(cache_dir: Path = DEFAULT_CACHE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (current_members, changes), cached to disk after first fetch."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    current_path = cache_dir / "sp500_current.parquet"
    changes_path = cache_dir / "sp500_changes.parquet"
    if current_path.exists() and changes_path.exists():
        return pd.read_parquet(current_path), pd.read_parquet(changes_path)

    resp = requests.get(
        _API_URL,
        params={"action": "parse", "oldid": _REVISION_ID, "prop": "text", "format": "json"},
        headers=_UA,
        timeout=60,
    )
    resp.raise_for_status()
    html = resp.json()["parse"]["text"]["*"]
    tables = pd.read_html(io.StringIO(html))

    current = tables[0].rename(columns=str.lower)[["symbol", "security"]]
    current["symbol"] = current["symbol"].map(normalize_ticker)

    changes = _flatten_changes(tables[1])

    current.to_parquet(current_path)
    changes.to_parquet(changes_path)
    return current, changes


def _flatten_changes(raw: pd.DataFrame) -> pd.DataFrame:
    """Flatten the MultiIndex change-log table to date/added/removed."""
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[("Date", "Date")], format="mixed"),
            "added": raw[("Added", "Ticker")],
            "removed": raw[("Removed", "Ticker")],
        }
    )
    for col in ("added", "removed"):
        out[col] = out[col].where(out[col].notna()).map(
            lambda t: normalize_ticker(t) if isinstance(t, str) else None
        )
    return out.sort_values("date").reset_index(drop=True)


def members_asof(
    asof: pd.Timestamp,
    current: set[str],
    changes: pd.DataFrame,
) -> set[str]:
    """Reconstruct membership at `asof` by undoing changes after it.

    Pure function so the reconstruction logic is testable without the
    network; `membership_mask` wires in the live tables.
    """
    members = set(current)
    later = changes[changes["date"] > asof].sort_values("date", ascending=False)
    for row in later.itertuples():
        # NaN is truthy; only strings are real tickers.
        if isinstance(row.added, str):
            members.discard(row.added)
        if isinstance(row.removed, str):
            members.add(row.removed)
    return members


def membership_mask(
    dates: pd.DatetimeIndex,
    cache_dir: Path = DEFAULT_CACHE,
) -> pd.DataFrame:
    """Boolean dates x tickers mask: was the name in the index on that date?

    Columns cover every ticker that was a member on any of the dates.
    """
    if dates.max() > ANCHOR_DATE:
        raise ValueError(
            f"membership source is anchored at {ANCHOR_DATE.date()}; "
            f"cannot reconstruct membership for {dates.max().date()}"
        )
    current_df, changes = fetch_sp500_tables(cache_dir)
    current = set(current_df["symbol"])

    per_date = {dt: members_asof(dt, current, changes) for dt in dates}
    all_tickers = sorted(set().union(*per_date.values()))
    mask = pd.DataFrame(False, index=dates, columns=all_tickers)
    for dt, members in per_date.items():
        mask.loc[dt, list(members)] = True
    return mask
