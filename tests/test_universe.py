"""Membership reconstruction logic, no network required."""
import pandas as pd

from xsmom.universe import members_asof, normalize_ticker


def test_normalize_ticker():
    assert normalize_ticker("BRK.B") == "BRK-B"
    assert normalize_ticker(" aapl ") == "AAPL"


def test_members_asof_reverses_changes():
    current = {"AAA", "BBB", "CCC"}
    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-06-01", "2021-06-01", "2022-06-01"]),
            "added": ["BBB", "CCC", None],
            "removed": ["XXX", None, "YYY"],
        }
    )

    # Before any change: undo all three. CCC and BBB leave; XXX and YYY return.
    early = members_asof(pd.Timestamp("2020-01-01"), current, changes)
    assert early == {"AAA", "XXX", "YYY"}

    # Between the first and second change.
    mid = members_asof(pd.Timestamp("2020-12-31"), current, changes)
    assert mid == {"AAA", "BBB", "YYY"}

    # After every change: membership is just the current set.
    late = members_asof(pd.Timestamp("2023-01-01"), current, changes)
    assert late == current
