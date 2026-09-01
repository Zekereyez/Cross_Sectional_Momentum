import numpy as np
import pandas as pd
import pytest

from xsmom.portfolio import quantile_weights


def signal_frame(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2020-01-31", periods=1, freq="ME")
    return pd.DataFrame([np.arange(1.0, n + 1.0)], index=idx,
                        columns=[f"T{i}" for i in range(n)])


def test_dollar_neutral_decile_books():
    w = quantile_weights(signal_frame(20), n_quantiles=10, long_short=True)
    row = w.iloc[0]

    # 20 names, 10 deciles: top 2 long at 0.5, bottom 2 short at -0.5.
    assert row[["T18", "T19"]].tolist() == [0.5, 0.5]
    assert row[["T0", "T1"]].tolist() == [-0.5, -0.5]
    assert row.clip(lower=0).sum() == pytest.approx(1.0)
    assert row.clip(upper=0).sum() == pytest.approx(-1.0)
    assert row.sum() == pytest.approx(0.0)


def test_long_only_has_no_shorts():
    w = quantile_weights(signal_frame(20), n_quantiles=10, long_short=False)
    row = w.iloc[0]
    assert (row >= 0).all()
    assert row.sum() == pytest.approx(1.0)


def test_too_few_names_returns_zero_weights():
    w = quantile_weights(signal_frame(5), n_quantiles=10, long_short=True)
    assert (w.iloc[0] == 0.0).all()


def test_membership_mask_excludes_non_members():
    sig = signal_frame(25)
    membership = pd.DataFrame(True, index=sig.index, columns=sig.columns)
    membership["T24"] = False  # top-ranked name is not in the index

    w = quantile_weights(sig, n_quantiles=10, long_short=True, membership=membership)
    row = w.iloc[0]
    assert row["T24"] == 0.0
    # Deciles re-form over the remaining 24 names; the exact split depends
    # on qcut, so just require the books stay balanced.
    assert row.clip(lower=0).sum() == pytest.approx(1.0)
    assert row.sum() == pytest.approx(0.0)
