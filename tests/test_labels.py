"""Leakage and correctness tests for the label construction.

The point of this file is one property: **the label for day t may depend on the
price at t+1 and on nothing after it.** If that ever breaks, every metric in the
project becomes meaningless while still looking excellent, so it is pinned by a
behavioural test rather than by a code comment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketgraph.features.labels import (
    MIN_STOCKS_PER_DAY,
    build_labels,
    compute_forward_returns,
    compute_log_returns,
)


@pytest.fixture
def toy_prices() -> pd.DataFrame:
    """A deterministic (days x tickers) price frame, wide enough for build_labels."""
    dates = pd.bdate_range("2020-01-01", periods=12)
    n = MIN_STOCKS_PER_DAY + 10
    rng = np.random.default_rng(0)
    data = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(len(dates), n)), axis=0))
    return pd.DataFrame(data, index=dates, columns=[f"T{i:02d}" for i in range(n)])


def test_forward_return_points_forward(toy_prices):
    """fwd_ret on row t must equal the return realised from t to t+1."""
    fwd = compute_forward_returns(toy_prices)
    p = toy_prices["T00"]
    for t in range(len(p) - 1):
        expected = p.iloc[t + 1] / p.iloc[t] - 1
        assert fwd["T00"].iloc[t] == pytest.approx(expected)


def test_last_day_has_no_forward_return(toy_prices):
    """The final row cannot have a target; it must be NaN, never filled."""
    fwd = compute_forward_returns(toy_prices)
    assert fwd.iloc[-1].isna().all()


def test_log_returns_point_backward(toy_prices):
    """Feature returns must look back, so row t uses t-1 and t only."""
    rets = compute_log_returns(toy_prices)
    p = toy_prices["T00"]
    assert np.isnan(rets["T00"].iloc[0])
    assert rets["T00"].iloc[3] == pytest.approx(np.log(p.iloc[3] / p.iloc[2]))


def test_label_matches_hand_computation(toy_prices):
    """label == 1 exactly when the stock's next-day return beats that day's median."""
    labels = build_labels(toy_prices)
    day = labels.date.unique()[2]
    slice_ = labels[labels.date == day]
    median = slice_.fwd_ret.median()
    expected = (slice_.fwd_ret > median).astype(int)
    assert (slice_.label.to_numpy() == expected.to_numpy()).all()


def test_label_depends_on_next_day_price(toy_prices):
    """Perturbing the price at t+1 must change the label at t -- the signal is real."""
    base = build_labels(toy_prices)
    target_row = 4

    bumped = toy_prices.copy()
    bumped.iloc[target_row + 1, 0] *= 1.5  # make T00 spike tomorrow

    after = build_labels(bumped)
    day = toy_prices.index[target_row]

    before_label = base[(base.date == day) & (base.ticker == "T00")].label.iloc[0]
    after_label = after[(after.date == day) & (after.ticker == "T00")].label.iloc[0]
    assert before_label == 0 and after_label == 1


def test_label_ignores_prices_beyond_next_day(toy_prices):
    """THE leakage test: nothing at t+2 or later may influence the label at t."""
    base = build_labels(toy_prices)
    target_row = 4

    tampered = toy_prices.copy()
    # Violently rewrite the entire future from t+2 onward.
    tampered.iloc[target_row + 2 :, :] *= 3.0

    after = build_labels(tampered)
    day = toy_prices.index[target_row]

    before = base[base.date == day].sort_values("ticker").label.to_numpy()
    now = after[after.date == day].sort_values("ticker").label.to_numpy()
    assert (before == now).all(), "labels changed when only the far future changed"


def test_no_missing_or_duplicate_rows(toy_prices):
    labels = build_labels(toy_prices)
    assert labels[["fwd_ret", "cs_median", "label"]].notna().all().all()
    assert not labels.duplicated(["date", "ticker"]).any()
    assert set(labels.label.unique()) <= {0, 1}


def test_real_data_class_balance():
    """On the real panel the classes must be near-balanced by construction."""
    labels = build_labels()
    assert 0.48 < labels.label.mean() < 0.52
    # Every day must have a forward return except the very last session.
    assert labels.groupby("date").size().min() >= MIN_STOCKS_PER_DAY
