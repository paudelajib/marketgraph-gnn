"""Non-graph baselines. These exist to make the GNN prove itself.

A graph model that beats random is meaningless; a graph model that beats gradient
boosting on the same features is a result. Most published "GNN for stock prediction"
work omits this comparison entirely, which is precisely why its claims are hard to
believe.

Four reference points, in increasing order of what they'd cost you to build:

* ``random``      -- pure noise. Rank IC must land at ~0.000. If it doesn't, the
                     evaluation harness itself is broken, so this is really a test
                     of the scorer rather than a model.
* ``reversal_1d`` -- a one-line heuristic: yesterday's biggest loser is tomorrow's
                     best bet. Short-horizon reversal is a well-documented effect
                     in daily equity data, so this is a real hurdle, not a strawman.
* ``logistic``    -- a linear model on the 10 features. Anything nonlinear has to
                     justify its complexity against this.
* ``gbdt``        -- gradient-boosted trees, the workhorse that wins most tabular
                     problems. This is the honest bar for the GNN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from marketgraph.features.node_features import FEATURE_NAMES


class Baseline:
    """Minimal fit/predict interface shared by every baseline."""

    name = "base"

    def fit(self, train: pd.DataFrame) -> "Baseline":
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class RandomBaseline(Baseline):
    """Uniform noise. Establishes that the metrics report ~0 for no signal."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.rng.normal(size=len(data))


class ReversalBaseline(Baseline):
    """Score = minus yesterday's return: bet on short-term mean reversion.

    No fitting at all. Features are already cross-sectionally standardised, so
    negating ``ret_1d`` directly gives a within-day ranking.
    """

    name = "reversal_1d"

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return -data["ret_1d"].to_numpy()


class MomentumBaseline(Baseline):
    """Score = trailing 21-day return: bet that recent winners keep winning."""

    name = "momentum_21d"

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return data["ret_21d"].to_numpy()


class LogisticBaseline(Baseline):
    """L2-regularised logistic regression on the 10 features.

    No feature scaling step: the inputs are already cross-sectional z-scores, which
    is exactly the scale a regularised linear model wants.
    """

    name = "logistic"

    def __init__(self, C: float = 1.0) -> None:
        self.model = LogisticRegression(C=C, max_iter=1000, n_jobs=-1)

    def fit(self, train: pd.DataFrame) -> "LogisticBaseline":
        self.model.fit(train[FEATURE_NAMES].to_numpy(), train["label"].to_numpy())
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(data[FEATURE_NAMES].to_numpy())[:, 1]


class GBDTBaseline(Baseline):
    """Gradient-boosted trees -- the serious non-graph competitor.

    Shallow trees and a low learning rate on purpose. The signal-to-noise ratio in
    daily returns is brutal, and a deep forest will happily memorise 145,000 rows of
    noise and generalise worse than the linear model.
    """

    name = "gbdt"

    def __init__(self, seed: int = 0) -> None:
        self.model = HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.03,
            max_iter=300,
            min_samples_leaf=200,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=seed,
        )

    def fit(self, train: pd.DataFrame) -> "GBDTBaseline":
        self.model.fit(train[FEATURE_NAMES].to_numpy(), train["label"].to_numpy())
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(data[FEATURE_NAMES].to_numpy())[:, 1]


def all_baselines(seed: int = 0) -> list[Baseline]:
    return [
        RandomBaseline(seed),
        ReversalBaseline(),
        MomentumBaseline(),
        LogisticBaseline(),
        GBDTBaseline(seed),
    ]
