#!/usr/bin/env python3
"""How much does the conclusion depend on the assumed transaction cost?

Two parts:

1. **What a one-cent spread actually costs, per stock.** A bid-ask spread quoted in
   cents converts to basis points as ``spread / price``, so the same penny is trivial
   on a $200 stock and expensive on a $12 one. This estimates the universe-wide cost
   floor from real prices instead of assuming a single number.

2. **Net performance across a range of costs**, so the reader can see exactly where
   each strategy crosses from profitable to unprofitable.

    python scripts/cost_sensitivity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from marketgraph.config import ARTIFACTS_DIR
from marketgraph.data.ingest import load_prices
from marketgraph.data.universe import MARKET_PROXY
from marketgraph.eval.backtest import run_backtest, summarize

COST_GRID = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
HEADLINE = ["reversal_1d", "gbdt", "gcn_sector", "logistic", "random"]


def spread_floor() -> None:
    """Convert a one-cent quoted spread into basis points, per stock, at latest prices.

    One cent is the minimum tick for almost all US equities above $1, so this is a
    *lower bound* on the spread -- and you pay roughly half of it when you cross.
    """
    prices = load_prices()
    latest = (
        prices[prices.ticker != MARKET_PROXY]
        .sort_values("date")
        .groupby("ticker")
        .close.last()
    )

    half_spread_bps = (0.005 / latest) * 1e4  # half of one cent, in bps
    summary = half_spread_bps.describe(percentiles=[0.1, 0.5, 0.9])

    print("Cost floor from a 1-cent spread (you pay ~half when crossing)")
    print(f"  cheapest decile   {summary['10%']:5.2f} bps")
    print(f"  median stock      {summary['50%']:5.2f} bps")
    print(f"  priciest decile   {summary['90%']:5.2f} bps")
    print(f"  universe mean     {half_spread_bps.mean():5.2f} bps\n")

    worst = half_spread_bps.nlargest(5)
    print("  Most expensive to trade (lowest share prices):")
    for ticker, bps in worst.items():
        print(f"    {ticker:6s} ${latest[ticker]:7.2f} -> {bps:5.2f} bps")
    print()


def sensitivity() -> pd.DataFrame:
    """Net Sharpe for each model across the cost grid."""
    frames = []
    for fname in ("preds_baselines.parquet", "preds_gnn.parquet"):
        path = ARTIFACTS_DIR / fname
        if path.exists():
            frames.append(pd.read_parquet(path))
    preds = pd.concat(frames, ignore_index=True)

    rows = []
    for name, group in preds.groupby("model"):
        if name not in HEADLINE:
            continue
        row: dict[str, float | str] = {"model": name}
        for cost in COST_GRID:
            stats = summarize(run_backtest(group, cost_bps=cost), name)
            row[f"{cost:g}bps"] = round(stats["net_sharpe"], 2)
            row["break_even_bps"] = round(stats["break_even_bps"], 2)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("model")
    cols = [f"{c:g}bps" for c in COST_GRID] + ["break_even_bps"]
    return table[cols].sort_values("break_even_bps", ascending=False)


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    spread_floor()

    table = sensitivity()
    print("Net Sharpe ratio by assumed cost per unit traded")
    print(table.to_string())
    print("\nA strategy is viable only where its net Sharpe is positive.")
