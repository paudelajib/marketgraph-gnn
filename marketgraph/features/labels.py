"""Prediction targets: does a stock beat the cross-sectional median tomorrow?

The label
---------
For each stock ``i`` and trading day ``t``:

    fwd_ret[i, t] = close[i, t+1] / close[i, t] - 1          (next-day simple return)
    label[i, t]   = 1 if fwd_ret[i, t] > median_j(fwd_ret[j, t]) else 0

The median is taken **across stocks on day t**, over only the stocks actually
trading that day. Three consequences worth internalising:

* **Balanced by construction.** Roughly half the universe beats the median every
  single day, so a constant predictor scores ~50% and cannot fake skill. Compare
  with an "is it up?" label, where upward market drift makes "always up" score ~53%.

* **Market-neutral.** On a day the whole market falls 3%, the label still asks who
  fell *least*. The model is forced to learn relative structure -- exactly what a
  graph over related stocks can supply -- instead of re-learning market direction.

* **Directly tradable.** Long the predicted-above-median names, short the rest, and
  the label is the strategy. Phase 5's backtest needs no reinterpretation.

The alignment trap
------------------
``label[t]`` describes the return earned **between t and t+1**. Every feature at
``t`` must therefore use data up to and including ``t`` and nothing later. In pandas
this is one ``shift(-1)``, and getting its sign backwards is the single most common
way a stock-prediction project reports a fake 90% accuracy: ``shift(+1)`` would let
the model see tomorrow's price while predicting tomorrow.

``tests/test_labels.py`` pins the direction against a hand-computed example, so the
sign is verified rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketgraph.data.ingest import load_prices, to_wide
from marketgraph.data.universe import MARKET_PROXY

# Below this many trading names, the cross-sectional median is too noisy to define
# a meaningful ranking. Early-2010 days are all well above it; this guards the
# edges of the sample.
MIN_STOCKS_PER_DAY = 30


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, ``log(p_t / p_{t-1})``. Backward-looking: safe as a feature.

    Log returns are used for *features* because they are additive across time
    (a 5-day return is the sum of five daily ones) and closer to symmetric, which
    suits standardisation.
    """
    return np.log(prices).diff()


def compute_forward_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Next-day **simple** return, indexed by the day the prediction is made.

    ``shift(-1)`` pulls tomorrow's value back onto today's row. That is deliberate
    and correct *for a target*, and would be a look-ahead bug for a feature.

    Simple (not log) returns here because they are what a portfolio actually earns,
    so Phase 5 can sum them into PnL without conversion.
    """
    simple = prices.pct_change()
    return simple.shift(-1)


def build_labels(prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the label table in long format.

    Returns columns ``[date, ticker, fwd_ret, cs_median, label]``, one row per
    (day, tradable stock). Rows where the forward return is unknown -- a stock's
    last day, the final day of the sample, pre-IPO gaps -- are dropped rather than
    filled, so a missing label can never be mistaken for a real one.
    """
    if prices is None:
        panel = load_prices()
        prices = to_wide(panel, "close")

    # The market proxy is a benchmark, not a tradable node in the graph. Leaving it
    # in would drag the cross-sectional median toward the market return.
    stocks = prices.drop(columns=[MARKET_PROXY], errors="ignore")

    fwd = compute_forward_returns(stocks)

    # Median across stocks for each day, skipping absent names automatically.
    valid_per_day = fwd.notna().sum(axis=1)
    cs_median = fwd.median(axis=1, skipna=True)
    cs_median[valid_per_day < MIN_STOCKS_PER_DAY] = np.nan

    long = (
        fwd.stack(future_stack=True)
        .rename("fwd_ret")
        .rename_axis(index=["date", "ticker"])
        .reset_index()
        .dropna(subset=["fwd_ret"])
    )
    long["cs_median"] = long["date"].map(cs_median)
    long = long.dropna(subset=["cs_median"])

    # Strict '>' so exact ties (the median stock itself, on odd-sized days) fall to
    # class 0. This costs at most one row a day and keeps the rule unambiguous.
    long["label"] = (long["fwd_ret"] > long["cs_median"]).astype(np.int8)

    return long.sort_values(["date", "ticker"]).reset_index(drop=True)


def label_summary(labels: pd.DataFrame) -> pd.DataFrame:
    """Per-year diagnostics: class balance and universe size over time."""
    g = labels.groupby(labels.date.dt.year)
    return pd.DataFrame(
        {
            "days": g.date.nunique(),
            "rows": g.size(),
            "stocks_per_day": (g.size() / g.date.nunique()).round(1),
            "pct_label_1": (g.label.mean() * 100).round(2),
            "mean_fwd_ret_bps": (g.fwd_ret.mean() * 1e4).round(1),
        }
    )


if __name__ == "__main__":
    labels = build_labels()
    pd.set_option("display.width", 120)

    print(f"Labels: {len(labels):,} rows | {labels.date.nunique():,} days | "
          f"{labels.ticker.nunique()} tickers")
    print(f"Range:  {labels.date.min().date()} to {labels.date.max().date()}")
    print(f"Class balance: {labels.label.mean():.4f} (1 = beats median)\n")

    print("Per-year breakdown")
    print(label_summary(labels).to_string())

    print("\nWorked example -- AAPL, first five rows")
    ex = labels[labels.ticker == "AAPL"].head(5)
    print(ex.to_string(index=False))
