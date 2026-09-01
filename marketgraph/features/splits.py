"""Walk-forward splitting: train on the past, test on the future, repeatedly.

Why not ``train_test_split``
----------------------------
A random split scatters days across train and test, so the model trains on March 2015
while being tested on February 2015. It has then seen the future while predicting the
past, and reports an accuracy it could never achieve live. Every split here is by
**time**: an earlier block trains, a later block tests, and the window rolls forward.

The layout of one fold
----------------------
::

    |<---- train ---->|  gap  |<- val ->|  gap  |<- test ->|
                       ^^^^^            ^^^^^
                     embargo          embargo

* **train** fits the model.
* **val** chooses hyperparameters and stops training early. It has to sit *before*
  test in time, or model selection itself becomes a leak.
* **test** is scored once and never looked at while tuning.

Why the gaps exist
------------------
Two distinct reasons, and only the first is a hard leak:

1. **The label horizon.** The last training day ``T`` carries a label describing the
   return from ``T`` to ``T+1``. If validation began at ``T+1``, that day's outcome
   was already inside training. One day of gap removes this exactly.

2. **Overlapping feature windows.** ``ret_63d`` at ``T+1`` is computed from a window
   that overlaps training days. This is *not* leakage -- those are past observations,
   legitimately available at prediction time -- but it makes adjacent samples highly
   correlated, so a test set butted right against training looks easier than a test
   set a month later. Widening the gap past one day trades a little data for a more
   honest estimate.

``EMBARGO_DAYS = 5`` covers (1) with room to spare and takes a modest bite out of (2).
It is a judgement call, not a derived constant, which is why it is a parameter.

Rolling vs expanding
--------------------
Default is **rolling**: a fixed-length training window that slides forward, so the
model always learns from the most recent N years and can adapt as correlation
structure changes. ``expanding=True`` keeps every day since the start instead, which
gives more data but weights 2011 as heavily as 2025.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Defaults chosen so folds fit comfortably in 2010-2026 and each model trains in
# minutes rather than hours on a laptop.
TRAIN_YEARS = 5
VAL_MONTHS = 6
TEST_MONTHS = 12
EMBARGO_DAYS = 5


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold, as inclusive date ranges."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def describe(self) -> str:
        return (
            f"fold {self.index}: "
            f"train {self.train_start.date()}..{self.train_end.date()} | "
            f"val {self.val_start.date()}..{self.val_end.date()} | "
            f"test {self.test_start.date()}..{self.test_end.date()}"
        )


class WalkForwardSplitter:
    """Generate successive train/val/test folds ordered in time.

    Consecutive folds step forward by exactly one test period, so the test blocks
    tile the timeline without overlapping. Concatenating their predictions yields one
    continuous out-of-sample series covering every year the folds reach.
    """

    def __init__(
        self,
        train_years: int = TRAIN_YEARS,
        val_months: int = VAL_MONTHS,
        test_months: int = TEST_MONTHS,
        embargo_days: int = EMBARGO_DAYS,
        expanding: bool = False,
    ) -> None:
        if embargo_days < 1:
            raise ValueError(
                "embargo_days must be >= 1: the label at the last training day "
                "describes the next day's return, so a zero gap leaks it."
            )
        self.train_years = train_years
        self.val_months = val_months
        self.test_months = test_months
        self.embargo_days = embargo_days
        self.expanding = expanding

    def split(self, dates: pd.Series | pd.DatetimeIndex) -> list[Fold]:
        """Build folds covering ``dates``. Returns [] if the span is too short."""
        all_dates = pd.DatetimeIndex(sorted(pd.unique(pd.DatetimeIndex(dates))))
        if len(all_dates) == 0:
            return []

        first, last = all_dates[0], all_dates[-1]
        embargo = pd.Timedelta(days=self.embargo_days)

        folds: list[Fold] = []
        train_start = first
        train_end = first + pd.DateOffset(years=self.train_years)

        while True:
            val_start = train_end + embargo
            val_end = val_start + pd.DateOffset(months=self.val_months)
            test_start = val_end + embargo
            # Minus one day because Fold ranges are inclusive at both ends: without
            # it, fold N's test_end and fold N+1's test_start land on the same date
            # and that day gets scored twice in the concatenated out-of-sample series.
            test_end = (
                test_start + pd.DateOffset(months=self.test_months) - pd.Timedelta(days=1)
            )

            # Stop once a fold would run past the data. A partial final test period
            # would be scored on fewer days and is dropped rather than compared
            # against full-length folds.
            if test_end > last:
                break

            folds.append(
                Fold(
                    index=len(folds),
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )

            # Step forward by one test period so test blocks tile the timeline.
            step = pd.DateOffset(months=self.test_months)
            train_end = train_end + step
            if not self.expanding:
                train_start = train_start + step

        return folds

    @staticmethod
    def apply(
        data: pd.DataFrame, fold: Fold
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Slice ``data`` (needs a ``date`` column) into that fold's three blocks."""
        d = data["date"]
        train = data[(d >= fold.train_start) & (d <= fold.train_end)]
        val = data[(d >= fold.val_start) & (d <= fold.val_end)]
        test = data[(d >= fold.test_start) & (d <= fold.test_end)]
        return train, val, test


if __name__ == "__main__":
    from marketgraph.features.dataset import load_dataset

    data = load_dataset()
    splitter = WalkForwardSplitter()
    folds = splitter.split(data.date)

    print(f"{len(folds)} folds over {data.date.min().date()}..{data.date.max().date()}\n")
    for fold in folds:
        train, val, test = splitter.apply(data, fold)
        print(f"{fold.describe()}  ->  "
              f"{len(train):>7,} / {len(val):>6,} / {len(test):>6,} rows")

    total_test_days = sum(
        splitter.apply(data, f)[2].date.nunique() for f in folds
    )
    print(f"\nOut-of-sample coverage: {total_test_days:,} trading days "
          f"({total_test_days / 252:.1f} years)")
