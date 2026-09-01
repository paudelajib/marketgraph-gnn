"""Tests for walk-forward splitting.

The property under test: **within a fold, every training day strictly precedes every
validation day, which strictly precedes every test day, with the embargo gap between
them.** If that breaks, the model is scored on data it trained on and every metric in
the project becomes meaningless while still looking excellent.
"""

from __future__ import annotations

import pandas as pd
import pytest

from marketgraph.features.splits import EMBARGO_DAYS, WalkForwardSplitter


@pytest.fixture
def toy_data() -> pd.DataFrame:
    """Twelve years of business days with a couple of rows per day."""
    dates = pd.bdate_range("2010-01-01", "2022-12-31")
    return pd.DataFrame(
        {
            "date": dates.repeat(2),
            "ticker": ["A", "B"] * len(dates),
            "label": 0,
        }
    )


@pytest.fixture
def splitter() -> WalkForwardSplitter:
    return WalkForwardSplitter()


def test_blocks_are_ordered_in_time(toy_data, splitter):
    """train < val < test, with no date shared between any two blocks."""
    for fold in splitter.split(toy_data.date):
        train, val, test = splitter.apply(toy_data, fold)
        assert train.date.max() < val.date.min()
        assert val.date.max() < test.date.min()

        train_days = set(train.date)
        val_days = set(val.date)
        test_days = set(test.date)
        assert not train_days & val_days
        assert not val_days & test_days
        assert not train_days & test_days


def test_embargo_gap_is_honoured(toy_data, splitter):
    """The real gap between blocks must be at least EMBARGO_DAYS."""
    minimum = pd.Timedelta(days=EMBARGO_DAYS)
    for fold in splitter.split(toy_data.date):
        train, val, test = splitter.apply(toy_data, fold)
        assert val.date.min() - train.date.max() >= minimum
        assert test.date.min() - val.date.max() >= minimum


def test_test_blocks_tile_without_overlapping(toy_data, splitter):
    """Successive folds must not score the same day twice."""
    folds = splitter.split(toy_data.date)
    assert len(folds) >= 2

    seen: set[pd.Timestamp] = set()
    for fold in folds:
        days = set(splitter.apply(toy_data, fold)[2].date)
        assert not (days & seen), "a test day appeared in two folds"
        seen |= days

    starts = [f.test_start for f in folds]
    assert starts == sorted(starts)


def test_zero_embargo_is_rejected():
    """A zero gap leaks the last training label's outcome into validation."""
    with pytest.raises(ValueError, match="embargo_days"):
        WalkForwardSplitter(embargo_days=0)


def test_rolling_window_has_constant_length(toy_data):
    """Default rolling mode: the training span stays fixed as folds advance.

    Tolerance of a couple of days because calendar arithmetic makes a five-year
    span 1826 or 1827 days depending on how many leap days it happens to contain.
    """
    splitter = WalkForwardSplitter(expanding=False)
    folds = splitter.split(toy_data.date)
    spans = [(f.train_end - f.train_start).days for f in folds]
    assert max(spans) - min(spans) <= 2


def test_expanding_window_grows(toy_data):
    """Expanding mode: training always starts at the first date and lengthens."""
    splitter = WalkForwardSplitter(expanding=True)
    folds = splitter.split(toy_data.date)
    assert len({f.train_start for f in folds}) == 1
    lengths = [(f.train_end - f.train_start).days for f in folds]
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1]


def test_no_partial_final_fold(toy_data, splitter):
    """Folds never extend past the available data."""
    last = toy_data.date.max()
    for fold in splitter.split(toy_data.date):
        assert fold.test_end <= last


def test_short_history_yields_no_folds(splitter):
    """Too little data must produce zero folds, not a malformed one."""
    dates = pd.bdate_range("2020-01-01", periods=100)
    assert splitter.split(dates) == []


def test_every_block_is_populated_on_real_data(splitter):
    """On the real dataset each fold must have all three blocks non-empty."""
    from marketgraph.features.dataset import load_dataset

    data = load_dataset()
    folds = splitter.split(data.date)
    assert len(folds) >= 5

    for fold in folds:
        train, val, test = splitter.apply(data, fold)
        assert len(train) > 0 and len(val) > 0 and len(test) > 0
        assert train.date.max() < test.date.min()
