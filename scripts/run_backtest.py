#!/usr/bin/env python3
"""Backtest every model's saved out-of-sample predictions, and plot the result.

Consumes the prediction files written by ``run_baselines.py`` and ``run_gnn.py``, so
it runs in seconds and never retrains anything.

    python scripts/run_backtest.py
    python scripts/run_backtest.py --cost-bps 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from marketgraph.config import ARTIFACTS_DIR, DOCS_DIR, ensure_dirs
from marketgraph.eval.backtest import COST_BPS, run_backtest, summarize, summary_table

PRED_FILES = ["preds_baselines.parquet", "preds_gnn.parquet"]


def load_predictions() -> pd.DataFrame:
    frames = []
    for fname in PRED_FILES:
        path = ARTIFACTS_DIR / fname
        if path.exists():
            frames.append(pd.read_parquet(path))
        else:
            print(f"  (missing {fname} -- run the matching script first)")
    if not frames:
        raise FileNotFoundError("No predictions found. Run run_baselines.py first.")
    return pd.concat(frames, ignore_index=True)


def plot_equity(curves: dict[str, pd.DataFrame], path: Path, cost_bps: float) -> None:
    """Two panels: gross on the left, net of costs on the right, same y-scale.

    Plotting them side by side with a shared scale is the point of the figure -- the
    collapse from one panel to the other is the finding.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for name, bt in curves.items():
        axes[0].plot(bt.index, bt.gross_equity, label=name, lw=1.3)
        axes[1].plot(bt.index, bt.net_equity, label=name, lw=1.3)

    axes[0].set_title("Gross of costs")
    axes[1].set_title(f"Net of costs ({cost_bps:.0f} bps per unit traded)")
    for ax in axes:
        ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel("date")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("growth of $1")
    axes[1].legend(fontsize=8, loc="best")

    fig.suptitle(
        "Long-short decile portfolio, walk-forward out-of-sample (2016-2026)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"Saved -> {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cost-bps", type=float, default=COST_BPS)
    args = p.parse_args()

    ensure_dirs()
    preds = load_predictions()
    print(f"{preds.model.nunique()} models | {len(preds):,} predictions\n")

    rows, curves = [], {}
    for name, group in preds.groupby("model"):
        bt = run_backtest(group, cost_bps=args.cost_bps)
        curves[name] = bt
        rows.append(summarize(bt, name))

    table = summary_table(rows).sort_values("net_sharpe", ascending=False)
    table.to_csv(ARTIFACTS_DIR / "backtest_results.csv")
    # Also under docs/, which is tracked, so the README's numbers ship with the repo.
    table.to_csv(DOCS_DIR / "backtest_results.csv")

    pd.set_option("display.width", 170)
    print("=" * 110)
    print(f"LONG-SHORT BACKTEST, DAILY REBALANCE, {args.cost_bps:.0f} BPS COSTS")
    print("=" * 110)
    print(table.to_string())

    plot_equity(curves, DOCS_DIR / "equity_curves.png", args.cost_bps)
    print(f"Saved -> {ARTIFACTS_DIR / 'backtest_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
