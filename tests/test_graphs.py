"""Tests for graph construction.

The property that matters: **a correlation snapshot applied to day t may only use
returns from strictly before t.** A graph built over the full sample would encode
which stocks moved together in 2024 into the graph used to predict 2016 -- the most
damaging leak available in this project, and an easy one to introduce by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from marketgraph.graphs.build import (
    correlation_snapshots,
    normalize_adjacency,
    sector_adjacency,
    sparsify_top_k,
)

N_TICKERS = 30


@pytest.fixture
def returns() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-01", periods=600)
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        rng.normal(0, 0.01, size=(len(dates), N_TICKERS)),
        index=dates,
        columns=[f"T{i:02d}" for i in range(N_TICKERS)],
    )


@pytest.fixture
def graph_dates(returns) -> pd.DatetimeIndex:
    """Apply graphs only to the back half, leaving history for the first window."""
    return returns.index[300:]


def test_snapshots_ignore_future_returns(returns, graph_dates):
    """THE leakage test: rewrite all returns after a date; earlier graphs must not move."""
    base, assign = correlation_snapshots(returns, graph_dates, lookback=126)

    cut = graph_dates[100]
    tampered = returns.copy()
    tampered.loc[tampered.index >= cut] *= 50.0

    after, assign_after = correlation_snapshots(tampered, graph_dates, lookback=126)

    # Every snapshot that takes effect at or before `cut` was built from data before
    # `cut`, so it must be untouched.
    unaffected = np.unique(assign[graph_dates <= cut])
    np.testing.assert_allclose(
        base[unaffected], after[unaffected], rtol=1e-10,
        err_msg="a graph snapshot changed when only future returns changed",
    )
    np.testing.assert_array_equal(assign, assign_after)


def test_snapshots_do_depend_on_past_returns(returns, graph_dates):
    """Counterpart: altering the trailing window must change the graph."""
    base, _ = correlation_snapshots(returns, graph_dates, lookback=126)

    tampered = returns.copy()
    early = tampered.index < graph_dates[0]
    # Make two stocks move together in the past; the first snapshot should notice.
    tampered.loc[early, "T00"] = tampered.loc[early, "T01"]

    after, _ = correlation_snapshots(tampered, graph_dates, lookback=126)
    assert not np.allclose(base[0], after[0])


def test_sparsify_limits_neighbour_count(returns):
    """Top-k keeps each node's degree near k, not the full 30."""
    corr = np.clip(returns.corr().to_numpy(), 0.0, None)
    sparse = sparsify_top_k(corr, k=5)

    degree = (sparse > 0).sum(1)
    # Symmetrising with max can push a hub slightly above k, never below.
    assert degree.min() >= 1
    assert degree.mean() < 12
    assert np.allclose(sparse, sparse.T), "adjacency must be symmetric"
    assert np.allclose(np.diag(sparse), 0.0), "no self-loops before normalisation"


def test_normalisation_adds_self_loops_and_bounds_scale(returns):
    """D^-1/2 (A+I) D^-1/2 must have positive diagonal and rows that do not explode."""
    adj = sparsify_top_k(np.clip(returns.corr().to_numpy(), 0.0, None), k=5)
    norm = normalize_adjacency(adj)

    assert (np.diag(norm) > 0).all(), "self-loops missing"
    assert np.allclose(norm, norm.T)
    assert norm.sum(1).max() < 3.0, "row sums should stay O(1) after normalisation"


def test_sector_adjacency_matches_membership():
    tickers = ["A", "B", "C", "D"]
    sectors = {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"}
    adj = sector_adjacency(tickers, sectors)

    assert adj[0, 1] == 1 and adj[2, 3] == 1
    assert adj[0, 2] == 0 and adj[1, 3] == 0
    assert np.allclose(np.diag(adj), 0.0)


def test_partial_correlation_removes_the_market_factor():
    """Residualising must collapse correlation that comes only from a shared factor.

    Returns are built as one market factor plus independent noise, so the *true*
    pairwise relationship is zero and everything they share is the factor. Plain
    correlation should be large; partial correlation should be near zero.

    Compared on the raw correlation matrices, not the normalised adjacencies --
    normalisation forces row sums to ~1 either way, so it cannot show this.
    """
    from marketgraph.graphs.build import _residualise

    rng = np.random.default_rng(11)
    n_days, n_stocks = 400, 20
    market = rng.normal(0, 0.01, n_days)
    data = market[:, None] + rng.normal(0, 0.005, size=(n_days, n_stocks))

    dates = pd.bdate_range("2018-01-01", periods=n_days)
    window = pd.DataFrame(
        data, index=dates, columns=[f"T{i:02d}" for i in range(n_stocks)]
    )

    def mean_offdiag_corr(frame: pd.DataFrame) -> float:
        corr = frame.corr().to_numpy()
        off = ~np.eye(len(corr), dtype=bool)
        return float(corr[off].mean())

    plain = mean_offdiag_corr(window)
    partial = mean_offdiag_corr(_residualise(window))

    assert plain > 0.6, "the synthetic market factor should dominate plain correlation"
    assert abs(partial) < 0.1, "residualising should leave near-zero correlation"
    assert partial < plain
