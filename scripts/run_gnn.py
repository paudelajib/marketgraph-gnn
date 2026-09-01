#!/usr/bin/env python3
"""Walk-forward evaluation of the graph models against the non-graph control.

Every configuration is trained from scratch on each fold, across several seeds, and
scored on that fold's untouched test window. Predictions are pooled into one
continuous ~10-year out-of-sample series.

The comparison that matters is ``mlp_no_graph`` versus ``gcn_*``: identical capacity,
optimiser, loss and training loop, differing only in whether information crosses
between stocks.

    python scripts/run_gnn.py                # 3 seeds, all configs
    python scripts/run_gnn.py --seeds 1      # quick pass
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from marketgraph.config import ARTIFACTS_DIR, ensure_dirs
from marketgraph.eval.metrics import evaluate, results_table
from marketgraph.features.dataset import load_dataset
from marketgraph.features.splits import WalkForwardSplitter
from marketgraph.features.tensors import build_tensors
from marketgraph.graphs.build import build_all_graphs
from marketgraph.models.gnn import GAT, GCN, MLP, LearnedGCN
from marketgraph.train.loop import predict, train_model

# (config name, model factory, which graph to feed it)
CONFIGS = [
    ("mlp_no_graph", lambda f, n: MLP(f), "sector"),
    ("gcn_sector", lambda f, n: GCN(f), "sector"),
    ("gcn_correlation", lambda f, n: GCN(f), "correlation"),
    ("gcn_partial_corr", lambda f, n: GCN(f), "partial_corr"),
    ("gat_partial_corr", lambda f, n: GAT(f), "partial_corr"),
    ("gcn_learned_graph", lambda f, n: LearnedGCN(f, n), "sector"),
]


def run(n_seeds: int) -> pd.DataFrame:
    data = load_dataset()
    tensors = build_tensors(data)
    graphs = build_all_graphs(tensors.tickers, tensors.dates)

    splitter = WalkForwardSplitter()
    folds = splitter.split(data.date)

    print(f"{tensors.n_days} days | {tensors.n_nodes} nodes | "
          f"{tensors.n_features} features | {len(folds)} folds | {n_seeds} seed(s)\n")

    predictions: dict[str, list[pd.DataFrame]] = {}

    for fold in folds:
        train_idx = tensors.index_for(fold.train_start, fold.train_end)
        val_idx = tensors.index_for(fold.val_start, fold.val_end)
        test_idx = tensors.index_for(fold.test_start, fold.test_end)

        print(f"fold {fold.index} (test {fold.test_start.date()}..{fold.test_end.date()})")

        for name, factory, graph_name in CONFIGS:
            adj, assign = graphs[graph_name]
            t0 = time.time()

            # Average scores across seeds before scoring. A single seed on data this
            # noisy is close to a coin flip about which model looks best.
            seed_scores = []
            val_ics = []
            for seed in range(n_seeds):
                model = factory(tensors.n_features, tensors.n_nodes)
                model, val_ic = train_model(
                    model, tensors, adj, assign, train_idx, val_idx,
                    seed=seed + 100 * fold.index,
                )
                seed_scores.append(predict(model, tensors, adj, assign, test_idx))
                val_ics.append(val_ic)

            scores = np.mean(seed_scores, axis=0)
            predictions.setdefault(name, []).append(tensors.to_long(test_idx, scores))
            print(f"    {name:20s} val_ic {np.mean(val_ics):+.4f}  "
                  f"({time.time() - t0:5.1f}s)")

    ensure_dirs()

    rows = []
    pooled = []
    for name, frames in predictions.items():
        preds = pd.concat(frames, ignore_index=True)
        rows.append(evaluate(preds, name))
        pooled.append(preds.assign(model=name))

    # Persist the out-of-sample predictions so the backtest -- and any later analysis
    # -- can run without retraining 180 models.
    pd.concat(pooled, ignore_index=True).to_parquet(
        ARTIFACTS_DIR / "preds_gnn.parquet", index=False
    )

    table = results_table(rows).sort_values("rank_ic", ascending=False)
    out = ARTIFACTS_DIR / "gnn_results.csv"
    table.to_csv(out)

    print("\n" + "=" * 100)
    print(f"WALK-FORWARD OUT-OF-SAMPLE RESULTS ({n_seeds} seeds averaged)")
    print("=" * 100)
    print(table.to_string())
    print(f"\nSaved -> {out}")
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    pd.set_option("display.width", 160)
    run(args.seeds)
