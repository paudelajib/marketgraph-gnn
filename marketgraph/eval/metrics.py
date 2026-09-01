"""Scoring. Every metric here is computed **per day, then averaged across days**.

That is not a stylistic choice. The task is to rank ~115 stocks against each other
within a day, so pooling all 462,070 rows into one global metric would mix together
days with wildly different volatility and let a handful of extreme days dominate.
Per-day-then-average matches how the strategy is actually run: one decision per day.

Which metric to believe
-----------------------
**Accuracy is nearly useless here** and is reported only because people expect it.
With a median-split label, chance is 50% by construction, and the difference between
a worthless model and a genuinely good one is roughly 50% versus 52%. Worse, accuracy
weights every stock equally, while a strategy only ever trades the extremes of the
ranking.

**Rank IC is the metric that matters.** It is the Spearman correlation, within each
day, between predicted scores and realised next-day returns -- i.e. "did we get the
*ordering* right". It uses the whole cross-section, is insensitive to how the scores
are calibrated, and is what quant researchers actually quote. For daily US large-cap
equity, a mean rank IC of 0.02-0.05 is a real, publishable-quality signal. Anything
above ~0.15 on daily data means a leak, not a discovery.

**The IC t-statistic** answers the question that matters more than the mean: is this
edge distinguishable from luck? A mean IC of 0.03 that flips sign every other day is
worthless. ``t = mean(IC) / std(IC) * sqrt(n_days)``, and above ~2 is the usual bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

# Names of the columns every scoring function expects.
SCORE = "score"
LABEL = "label"
FWD_RET = "fwd_ret"


def daily_rank_ic(preds: pd.DataFrame) -> pd.Series:
    """Spearman correlation between score and realised return, one value per day."""
    def _ic(group: pd.DataFrame) -> float:
        if group[SCORE].nunique() < 2 or len(group) < 10:
            return np.nan
        return spearmanr(group[SCORE], group[FWD_RET]).statistic

    return preds.groupby("date", sort=True).apply(_ic, include_groups=False).dropna()


def daily_accuracy(preds: pd.DataFrame) -> pd.Series:
    """Fraction correct per day, thresholding the score at its own daily median.

    Thresholding at the *cross-sectional median of the scores* rather than at 0.5
    keeps this consistent with the label's definition and makes models with
    differently-calibrated outputs comparable.
    """
    def _acc(group: pd.DataFrame) -> float:
        pred = (group[SCORE] > group[SCORE].median()).astype(int)
        return float((pred == group[LABEL]).mean())

    return preds.groupby("date", sort=True).apply(_acc, include_groups=False)


def daily_precision_at_k(preds: pd.DataFrame, k: int = 10) -> pd.Series:
    """Of the k highest-scored stocks each day, what fraction beat the median.

    This is the metric closest to how the model is used: a strategy trades the top
    and bottom of the ranking and never touches the middle.
    """
    def _prec(group: pd.DataFrame) -> float:
        top = group.nlargest(k, SCORE)
        return float(top[LABEL].mean())

    return preds.groupby("date", sort=True).apply(_prec, include_groups=False)


def daily_long_short_return(preds: pd.DataFrame, quantile: float = 0.1) -> pd.Series:
    """Gross daily return of a long-top / short-bottom decile portfolio.

    Equal-weighted within each leg, and **gross** -- no transaction costs. Phase 5's
    backtest applies costs; this is the raw signal strength.
    """
    def _ls(group: pd.DataFrame) -> float:
        n = max(1, int(len(group) * quantile))
        ranked = group.sort_values(SCORE)
        short_leg = ranked.head(n)[FWD_RET].mean()
        long_leg = ranked.tail(n)[FWD_RET].mean()
        return float(long_leg - short_leg)

    return preds.groupby("date", sort=True).apply(_ls, include_groups=False)


def evaluate(preds: pd.DataFrame, name: str = "") -> dict[str, float]:
    """Full metric set for a table of predictions.

    ``preds`` needs columns ``[date, ticker, score, label, fwd_ret]``, where higher
    ``score`` means "more likely to beat the median".
    """
    ic = daily_rank_ic(preds)
    acc = daily_accuracy(preds)
    prec = daily_precision_at_k(preds, k=10)
    ls = daily_long_short_return(preds)

    n = len(ic)
    ic_t = float(ic.mean() / ic.std() * np.sqrt(n)) if n > 1 and ic.std() > 0 else 0.0

    # Annualised Sharpe of the gross long-short series. 252 trading days a year.
    sharpe = float(ls.mean() / ls.std() * np.sqrt(252)) if ls.std() > 0 else 0.0

    try:
        auc = float(roc_auc_score(preds[LABEL], preds[SCORE]))
    except ValueError:
        auc = float("nan")

    return {
        "model": name,
        "rank_ic": float(ic.mean()),
        "ic_t_stat": ic_t,
        "ic_positive_pct": float((ic > 0).mean() * 100),
        "accuracy": float(acc.mean()),
        "auc": auc,
        "precision_at_10": float(prec.mean()),
        "ls_return_bps": float(ls.mean() * 1e4),
        "ls_sharpe_gross": sharpe,
        "n_days": n,
    }


def results_table(rows: list[dict[str, float]]) -> pd.DataFrame:
    """Format a list of ``evaluate`` outputs into a comparison table."""
    df = pd.DataFrame(rows).set_index("model")
    return df.round(
        {
            "rank_ic": 4,
            "ic_t_stat": 2,
            "ic_positive_pct": 1,
            "accuracy": 4,
            "auc": 4,
            "precision_at_10": 4,
            "ls_return_bps": 2,
            "ls_sharpe_gross": 2,
        }
    )
