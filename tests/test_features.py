"""Causality and normalisation tests for node features.

The label tests pin that ``label[t]`` looks exactly one day forward. These pin the
opposite property for the inputs: **no feature at day t may depend on any data after
t.** Together they close both halves of the leakage surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketgraph.data.universe import MARKET_PROXY
from marketgraph.features.node_features import (
    FEATURE_NAMES,
    build_features,
    cross_sectional_normalize,
)

N_TICKERS = 40
N_DAYS = 300


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    """A deterministic long-format OHLCV panel, including the market proxy."""
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    tickers = [f"T{i:02d}" for i in range(N_TICKERS)] + [MARKET_PROXY]
    rng = np.random.default_rng(7)

    rows = []
    for t in tickers:
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, N_DAYS)))
        vol = rng.lognormal(15, 0.4, N_DAYS)
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": t,
                    "open": px,
                    "high": px * 1.01,
                    "low": px * 0.99,
                    "close": px,
                    "volume": vol,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_features_ignore_the_future(toy_panel):
    """THE causality test: rewrite everything after t; features at t must not move."""
    base = build_features(toy_panel)
    cutoff = base.date.unique()[50]

    tampered = toy_panel.copy()
    future = tampered.date > cutoff
    tampered.loc[future, ["open", "high", "low", "close"]] *= 4.0
    tampered.loc[future, "volume"] *= 10.0

    after = build_features(tampered)

    before_slice = base[base.date <= cutoff].sort_values(["date", "ticker"])
    after_slice = after[after.date <= cutoff].sort_values(["date", "ticker"])

    assert len(before_slice) == len(after_slice)
    np.testing.assert_allclose(
        before_slice[FEATURE_NAMES].to_numpy(),
        after_slice[FEATURE_NAMES].to_numpy(),
        rtol=1e-10,
        err_msg="a feature at time t changed when only future data changed",
    )


def test_features_do_depend_on_the_past(toy_panel):
    """Sanity counterpart: changing the recent past *must* move features, otherwise
    the causality test above would pass trivially on constant output.

    Note the perturbation is a *single day*, not a uniform rescaling of the series.
    Every feature here is built from ratios -- log-return differences, price over
    moving average, beta -- so multiplying a stock's whole price history by a
    constant cancels out exactly. That scale invariance is desirable (a 2:1 split
    must not change a stock's features), but it makes a uniform scaling useless as
    a probe.
    """
    base = build_features(toy_panel)
    cutoff = base.date.unique()[50]

    tampered = toy_panel.copy()
    that_day = (tampered.date == cutoff) & (tampered.ticker == "T00")
    tampered.loc[that_day, "close"] *= 1.3

    after = build_features(tampered)
    b = base[(base.date == cutoff) & (base.ticker == "T00")][FEATURE_NAMES].to_numpy()
    a = after[(after.date == cutoff) & (after.ticker == "T00")][FEATURE_NAMES].to_numpy()
    assert not np.allclose(a, b)


def test_normalisation_is_across_stocks_not_time(toy_panel):
    """Each day, across stocks, features must be ~mean 0 / sd 1.

    This is what catches an accidental ``axis=0``: time-series standardisation would
    leave the per-day cross-section with arbitrary mean and spread.
    """
    feats = build_features(toy_panel)
    per_day = feats.groupby("date")[FEATURE_NAMES].agg(["mean", "std"])

    for name in FEATURE_NAMES:
        assert per_day[(name, "mean")].abs().max() < 1e-6
        assert per_day[(name, "std")].between(0.7, 1.3).all()


def test_winsorising_caps_a_lone_outlier():
    """One extreme stock must not flatten the rest of that day's cross-section."""
    frame = pd.DataFrame(
        [[0.01] * 49 + [50.0]],
        index=pd.to_datetime(["2020-01-02"]),
        columns=[f"T{i:02d}" for i in range(50)],
    )
    z = cross_sectional_normalize(frame)
    assert z.iloc[0].abs().max() < 10, "outlier survived winsorisation"


def test_no_missing_values(toy_panel):
    feats = build_features(toy_panel)
    assert feats[FEATURE_NAMES].notna().all().all()
    assert np.isfinite(feats[FEATURE_NAMES].to_numpy()).all()
    assert not feats.duplicated(["date", "ticker"]).any()


def test_warmup_period_is_dropped(toy_panel):
    """No features before the longest lookback has enough history to fill."""
    feats = build_features(toy_panel)
    assert feats.date.min() > toy_panel.date.min() + pd.Timedelta(days=63)
