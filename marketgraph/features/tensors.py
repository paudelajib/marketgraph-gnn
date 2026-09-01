"""Reshape the long dataset into the dense tensors a graph model needs.

Tabular models take one row per (stock, day) and treat rows as independent. A graph
model cannot: it must see **all 115 stocks on the same day at once**, because a
node's prediction depends on its neighbours' features that day. So the natural unit
of a training example changes from "a row" to "a day".

    X     (T, N, F)  features -- 4,044 days x 115 stocks x 10 features
    y     (T, N)     labels
    fwd   (T, N)     realised next-day returns, for PnL
    mask  (T, N)     True where the stock actually traded and has features

The mask is what lets a fixed 115-node graph cope with a changing universe. META did
not exist in 2010, so its slot is masked out on those days: it is padded with zeros
in ``X`` and excluded from the loss and from every metric. This is why we never had
to drop stocks with short histories.

Memory is a non-issue -- 4,044 x 115 x 10 float32 is about 19 MB, so the entire
sixteen-year panel sits in RAM and every model trains from memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from marketgraph.features.dataset import load_dataset
from marketgraph.features.node_features import FEATURE_NAMES


@dataclass
class PanelTensors:
    """Dense (time x node) view of the dataset."""

    X: np.ndarray
    y: np.ndarray
    fwd: np.ndarray
    mask: np.ndarray
    dates: pd.DatetimeIndex
    tickers: list[str]

    @property
    def n_days(self) -> int:
        return self.X.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.X.shape[1]

    @property
    def n_features(self) -> int:
        return self.X.shape[2]

    def index_for(self, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        """Positions of days within an inclusive date range -- used to slice folds."""
        return np.flatnonzero((self.dates >= start) & (self.dates <= end))

    def to_long(self, rows: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
        """Turn per-day model scores back into the long table the metrics expect.

        ``rows`` are day indices and ``scores`` is (len(rows), N). Masked slots are
        dropped, so a padded stock can never reach the scorer.
        """
        day_idx, node_idx = np.nonzero(self.mask[rows])
        return pd.DataFrame(
            {
                "date": self.dates[rows][day_idx],
                "ticker": np.asarray(self.tickers)[node_idx],
                "score": scores[day_idx, node_idx],
                "label": self.y[rows][day_idx, node_idx],
                "fwd_ret": self.fwd[rows][day_idx, node_idx],
            }
        )


def build_tensors(data: pd.DataFrame | None = None) -> PanelTensors:
    """Pivot the long dataset into dense arrays with a validity mask."""
    if data is None:
        data = load_dataset()

    dates = pd.DatetimeIndex(sorted(data.date.unique()))
    tickers = sorted(data.ticker.unique())
    date_pos = {d: i for i, d in enumerate(dates)}
    node_pos = {t: i for i, t in enumerate(tickers)}

    T, N, F = len(dates), len(tickers), len(FEATURE_NAMES)

    X = np.zeros((T, N, F), dtype=np.float32)
    y = np.zeros((T, N), dtype=np.float32)
    fwd = np.zeros((T, N), dtype=np.float32)
    mask = np.zeros((T, N), dtype=bool)

    di = data.date.map(date_pos).to_numpy()
    ni = data.ticker.map(node_pos).to_numpy()

    X[di, ni] = data[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y[di, ni] = data["label"].to_numpy(dtype=np.float32)
    fwd[di, ni] = data["fwd_ret"].to_numpy(dtype=np.float32)
    mask[di, ni] = True

    return PanelTensors(X=X, y=y, fwd=fwd, mask=mask, dates=dates, tickers=tickers)


if __name__ == "__main__":
    t = build_tensors()
    print(f"X    {t.X.shape}  {t.X.nbytes / 1e6:.1f} MB")
    print(f"y    {t.y.shape}")
    print(f"mask {t.mask.shape}  {t.mask.mean() * 100:.1f}% valid")
    print(f"days {t.dates[0].date()} .. {t.dates[-1].date()}")
    print(f"nodes per day: min {t.mask.sum(1).min()}, max {t.mask.sum(1).max()}")
