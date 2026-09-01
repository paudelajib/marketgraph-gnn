#!/usr/bin/env python3
"""Walk-forward evaluation of the non-graph baselines.

Each model is refit from scratch on every fold's training window and scored on that
fold's test window. Predictions from all folds are concatenated into one continuous
out-of-sample series covering ~10 years, and metrics are computed on that.

    python scripts/run_baselines.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from marketgraph.config import ARTIFACTS_DIR, ensure_dirs
from marketgraph.eval.metrics import evaluate, results_table
from marketgraph.features.dataset import load_dataset
from marketgraph.features.splits import WalkForwardSplitter
from marketgraph.models.baselines import all_baselines


def run() -> pd.DataFrame:
    data = load_dataset()
    splitter = WalkForwardSplitter()
    folds = splitter.split(data.date)

    print(f"{len(folds)} folds | {len(data):,} rows | "
          f"{data.date.min().date()}..{data.date.max().date()}\n")

    predictions: dict[str, list[pd.DataFrame]] = {}

    for fold in folds:
        train, _val, test = splitter.apply(data, fold)
        print(f"fold {fold.index}: train {len(train):,} -> "
              f"test {fold.test_start.date()}..{fold.test_end.date()}", end="", flush=True)
        t0 = time.time()

        for model in all_baselines(seed=fold.index):
            model.fit(train)
            out = test[["date", "ticker", "label", "fwd_ret"]].copy()
            out["score"] = model.predict(test)
            predictions.setdefault(model.name, []).append(out)

        print(f"  ({time.time() - t0:.1f}s)")

    ensure_dirs()

    rows = []
    pooled = []
    for name, frames in predictions.items():
        preds = pd.concat(frames, ignore_index=True)
        rows.append(evaluate(preds, name))
        pooled.append(preds.assign(model=name))

    pd.concat(pooled, ignore_index=True).to_parquet(
        ARTIFACTS_DIR / "preds_baselines.parquet", index=False
    )

    table = results_table(rows).sort_values("rank_ic")
    out_path = ARTIFACTS_DIR / "baseline_results.csv"
    table.to_csv(out_path)

    print("\n" + "=" * 100)
    print("WALK-FORWARD OUT-OF-SAMPLE RESULTS (no graph)")
    print("=" * 100)
    print(table.to_string())
    print(f"\nSaved -> {out_path}")
    return table


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    run()
