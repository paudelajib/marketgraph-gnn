"""Join features to labels into the single table every model trains on.

Features describe a stock as of day ``t``; the label describes what happened between
``t`` and ``t+1``. The join is on ``(date, ticker)`` and is deliberately an **inner**
join: a row survives only if it has a complete feature vector *and* a known outcome.

Anything unmatched is dropped rather than imputed. A stock in its first 63 days has
no ``ret_63d``, and the final session has no tomorrow -- filling either with zeros
would hand the model a fabricated observation that looks exactly like a real one.
"""

from __future__ import annotations

import pandas as pd

from marketgraph.config import PROCESSED_DIR, ensure_dirs
from marketgraph.features.labels import build_labels
from marketgraph.features.node_features import FEATURE_NAMES, build_features

DATASET_PATH = PROCESSED_DIR / "dataset.parquet"

COLUMNS = ["date", "ticker"] + FEATURE_NAMES + ["label", "fwd_ret"]


def build_dataset(save: bool = True) -> pd.DataFrame:
    """Build the modelling table: ``[date, ticker, <10 features>, label, fwd_ret]``.

    ``fwd_ret`` is carried alongside the binary label because Phase 5's backtest
    needs the actual return to compute PnL -- knowing a prediction was correct says
    nothing about how much it earned.
    """
    features = build_features()
    labels = build_labels()

    data = features.merge(
        labels[["date", "ticker", "label", "fwd_ret"]],
        on=["date", "ticker"],
        how="inner",
        validate="one_to_one",
    )

    data = data[COLUMNS].sort_values(["date", "ticker"]).reset_index(drop=True)

    if save:
        ensure_dirs()
        data.to_parquet(DATASET_PATH, index=False, compression="snappy")
        print(f"Wrote {len(data):,} rows -> {DATASET_PATH}")

    return data


def load_dataset() -> pd.DataFrame:
    """Read the cached modelling table, building it first if absent."""
    if not DATASET_PATH.exists():
        return build_dataset()
    data = pd.read_parquet(DATASET_PATH)
    data["date"] = pd.to_datetime(data["date"])
    return data


if __name__ == "__main__":
    data = build_dataset()
    pd.set_option("display.width", 140)

    print(f"\nDataset: {len(data):,} rows | {data.date.nunique():,} days | "
          f"{data.ticker.nunique()} tickers")
    print(f"Range:   {data.date.min().date()} to {data.date.max().date()}")
    print(f"Balance: {data.label.mean():.4f}")
    print(f"Stocks per day: min {data.groupby('date').size().min()}, "
          f"max {data.groupby('date').size().max()}")

    print("\nFirst rows")
    print(data.head(3).to_string(index=False))
