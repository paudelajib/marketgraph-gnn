"""Graph construction: who is connected to whom, and how strongly.

This is the research core of the project. The GNN layer is two matrix multiplies; the
interesting question is what matrix you multiply by.

Three graphs, deliberately different in character:

* **sector** -- an edge between stocks sharing a GICS sector. Static, hand-specified,
  and knows nothing about prices. Our data validation measured its premise directly:
  same-sector pairs correlate 0.34 versus 0.18 for unrelated pairs.

* **correlation** -- an edge between stocks whose returns have moved together
  recently, rebuilt every month from a trailing window. Adaptive, but see the market
  factor problem below.

* **partial correlation** -- correlation after projecting out the market's own
  return. This is the interesting one. Two random large caps correlate ~0.18 purely
  because both follow the index; a plain correlation graph therefore connects
  everything to everything and mostly encodes "is a stock". Removing the common
  factor first leaves the *residual* co-movement, which is the genuinely relational
  part.

Causality
---------
A correlation graph computed over the full sample would be catastrophic leakage: the
2016 graph would encode which stocks moved together in 2024. Every snapshot here is
built from a trailing window ending strictly **before** the first day it is applied
to, and ``tests/test_graphs.py`` asserts exactly that.

Sparsification
--------------
A dense 115x115 correlation matrix is not a useful graph -- everything connects to
everything and message passing degenerates into a global average. Keeping each node's
top-k strongest neighbours forces the model to work with a real neighbourhood
structure. k=8 sits near the average GICS sector size here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK = 126
REBUILD_EVERY = 21
TOP_K = 8


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """Symmetric normalisation with self-loops: ``D^-1/2 (A + I) D^-1/2``.

    Two jobs. The self-loop keeps a node's own features in the mix, so a stock is not
    described purely by its neighbours. The degree scaling stops well-connected nodes
    from dominating: without it, activations grow with degree and deep stacks blow up.
    """
    n = adj.shape[-1]
    a = adj + np.eye(n, dtype=adj.dtype)
    deg = a.sum(-1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return a * d_inv_sqrt[..., None] * d_inv_sqrt[..., None, :]


def sparsify_top_k(matrix: np.ndarray, k: int = TOP_K) -> np.ndarray:
    """Keep each row's k largest off-diagonal entries, then symmetrise.

    Symmetrising with a max (rather than requiring mutual selection) means an edge
    survives if *either* endpoint considers the other a top neighbour -- otherwise a
    hub like a mega-cap would be dropped by all its smaller peers.
    """
    m = matrix.copy()
    np.fill_diagonal(m, -np.inf)

    keep = np.zeros_like(m, dtype=bool)
    top = np.argpartition(-m, kth=k, axis=1)[:, :k]
    keep[np.arange(m.shape[0])[:, None], top] = True

    out = np.where(keep, matrix, 0.0)
    out = np.maximum(out, out.T)
    np.fill_diagonal(out, 0.0)
    return np.clip(out, 0.0, None)


def sector_adjacency(tickers: list[str], sector_map: dict[str, str]) -> np.ndarray:
    """Binary adjacency: 1 where two stocks share a sector."""
    sectors = np.array([sector_map.get(t, "UNKNOWN") for t in tickers])
    adj = (sectors[:, None] == sectors[None, :]).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    return adj


def _residualise(window: pd.DataFrame) -> pd.DataFrame:
    """Regress out the cross-sectional mean return (the market factor) day by day.

    The equal-weighted mean across stocks is a serviceable market proxy, and
    subtracting each stock's fitted exposure to it leaves idiosyncratic residuals.
    """
    market = window.mean(axis=1)
    var = market.var()
    if not np.isfinite(var) or var <= 0:
        return window
    betas = window.apply(lambda col: col.cov(market) / var)
    return window - np.outer(market, betas)


def _snapshot(window: pd.DataFrame, partial: bool, k: int) -> np.ndarray:
    """One adjacency matrix from one trailing window of returns."""
    if partial:
        window = _residualise(window)
    corr = window.corr().to_numpy()
    corr = np.nan_to_num(corr, nan=0.0)  # stocks not yet listed
    # Only positive co-movement becomes an edge. A strongly negative correlation is
    # informative, but averaging a node with its anti-correlated peers cancels signal.
    return sparsify_top_k(np.clip(corr, 0.0, None), k=k)


def correlation_snapshots(
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lookback: int = LOOKBACK,
    rebuild_every: int = REBUILD_EVERY,
    top_k: int = TOP_K,
    partial: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build rolling correlation graphs.

    ``returns`` must be a (date x ticker) frame whose history starts **before**
    ``dates[0]``, so that even the first snapshot has a full trailing window of
    genuinely prior data.

    Returns ``(snapshots, assignment)`` where ``snapshots`` is (S, N, N) normalised
    adjacencies and ``assignment[i]`` gives the snapshot index for ``dates[i]``.
    """
    rebuild_at = list(range(0, len(dates), rebuild_every))
    snapshots = []
    assignment = np.zeros(len(dates), dtype=np.int64)

    for s, start in enumerate(rebuild_at):
        as_of = dates[start]
        # Strictly before `as_of`: the window may not contain the day it is used on.
        history = returns.loc[returns.index < as_of]
        window = history.iloc[-lookback:]

        if len(window) < 20:
            snapshots.append(np.zeros((returns.shape[1], returns.shape[1]), dtype=np.float32))
        else:
            snapshots.append(_snapshot(window, partial=partial, k=top_k).astype(np.float32))

        end = rebuild_at[s + 1] if s + 1 < len(rebuild_at) else len(dates)
        assignment[start:end] = s

    stacked = np.stack(snapshots)
    return normalize_adjacency(stacked).astype(np.float32), assignment


def build_all_graphs(
    tickers: list[str], dates: pd.DatetimeIndex
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Build every graph variant, each as ``(snapshots, assignment)``.

    The sector graph is static, so it is returned as a single snapshot that every day
    points at -- which keeps the interface uniform for the training loop.
    """
    from marketgraph.data.ingest import load_prices, to_wide
    from marketgraph.data.universe import get_sector_map

    prices = to_wide(load_prices(), "close")
    returns = np.log(prices).diff()
    returns = returns.reindex(columns=tickers)

    sector = normalize_adjacency(sector_adjacency(tickers, get_sector_map()))
    static_assignment = np.zeros(len(dates), dtype=np.int64)

    return {
        "sector": (sector[None].astype(np.float32), static_assignment),
        "correlation": correlation_snapshots(returns, dates, partial=False),
        "partial_corr": correlation_snapshots(returns, dates, partial=True),
    }


if __name__ == "__main__":
    from marketgraph.features.tensors import build_tensors

    t = build_tensors()
    graphs = build_all_graphs(t.tickers, t.dates)

    print(f"{t.n_nodes} nodes, {t.n_days} days\n")
    for name, (snaps, assign) in graphs.items():
        raw = snaps > 0
        np.einsum("sii->si", raw)[:] = False  # ignore self-loops when counting degree
        density = raw.mean() * 100
        degree = raw.sum(-1).mean()
        print(f"{name:14s} {len(snaps):>4d} snapshots | mean degree {degree:5.1f} | "
              f"density {density:4.1f}%")

    # How much do the price-based graphs actually agree with sector membership?
    from marketgraph.data.universe import get_sector_map

    same_sector = sector_adjacency(t.tickers, get_sector_map()) > 0
    for name in ("correlation", "partial_corr"):
        snaps, _ = graphs[name]
        edges = snaps > 0
        np.einsum("sii->si", edges)[:] = False
        overlap = (edges & same_sector).sum() / max(edges.sum(), 1) * 100
        print(f"{name:14s} {overlap:.1f}% of its edges are within-sector")
