"""Node features: what the model sees about each stock on each day.

Ten features per (stock, day), in four families:

    momentum      ret_1d, ret_5d, ret_21d, ret_63d
    volatility    vol_21d, vol_ratio
    mean-revert   dist_ma21, dist_ma63
    liquidity     volume_z
    market        beta_63d

Deliberately small. Ten interpretable features that you can reason about beat fifty
technical indicators you cannot, and with ~4,100 training days a wide feature space
mostly buys overfitting. Phase 2's baselines will tell us whether any of these carry
signal at all before we consider adding more.

Causality
---------
Every feature at day ``t`` is computed from data at ``t`` and earlier. pandas'
``rolling(w)`` is backward-looking -- the window at ``t`` spans ``[t-w+1, t]`` -- so
a rolling mean is safe where a ``shift(-1)`` would not be. ``tests/test_features.py``
verifies this by rewriting all prices after ``t`` and asserting features at ``t`` are
bit-identical.

Cross-sectional normalisation
-----------------------------
Each feature is standardised **across stocks within a day**, not across time within a
stock. This is the choice that matters most here, and it follows from the label:

* The target asks "does this stock beat the median *today*", so the input should say
  "where does this stock stand relative to its peers *today*". Both sides of the
  problem then live in the same cross-sectional frame.

* It neutralises regime shifts for free. A 2% daily move was ordinary in March 2020
  and extreme in 2017; raw values from those two periods are not comparable, but
  "top decile of the universe today" means the same thing in both.

* It removes the market factor from the features almost entirely. On a day when
  everything drops 3%, every stock's ``ret_1d`` is large and negative, so after
  standardisation the common component is gone and only relative position remains --
  which is exactly the part a graph over related stocks can help explain.

Winsorising at the 1st/99th percentile precedes standardisation, because a single
earnings-day 40% move would otherwise dominate that day's mean and standard
deviation and squash all 114 other stocks toward zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from marketgraph.data.ingest import load_prices, to_wide
from marketgraph.data.universe import MARKET_PROXY

FEATURE_NAMES = [
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ret_63d",
    "vol_21d",
    "vol_ratio",
    "dist_ma21",
    "dist_ma63",
    "volume_z",
    "beta_63d",
]

# The longest lookback any feature needs. Rows before this are dropped rather than
# back-filled: an imputed feature is a fabricated observation, and here it would be
# fabricated at exactly the point where a stock has just listed.
WARMUP_DAYS = 63

WINSOR_Q = 0.01


def compute_raw_features(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    market_returns: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Compute the ten raw (un-normalised) features as (date x ticker) frames."""
    log_px = np.log(prices)
    rets = log_px.diff()

    feats: dict[str, pd.DataFrame] = {}

    # --- Momentum: cumulative log return over each horizon. Log returns are used
    # because they sum across time, so an n-day return is just a rolling sum.
    for horizon in (1, 5, 21, 63):
        feats[f"ret_{horizon}d"] = log_px.diff(horizon)

    # --- Volatility: realised vol, and whether it is rising or falling.
    vol21 = rets.rolling(21).std()
    vol63 = rets.rolling(63).std()
    feats["vol_21d"] = vol21
    # >1 means short-term vol is elevated versus the quarter -- a regime-shift signal.
    feats["vol_ratio"] = vol21 / vol63

    # --- Mean reversion: distance from moving averages, as a fraction.
    feats["dist_ma21"] = prices / prices.rolling(21).mean() - 1.0
    feats["dist_ma63"] = prices / prices.rolling(63).mean() - 1.0

    # --- Liquidity: today's volume against this stock's own recent norm. Volume is
    # standardised per-stock first because its raw scale differs by orders of
    # magnitude across the universe, which would otherwise swamp the later
    # cross-sectional step.
    log_vol = np.log(volumes.replace(0, np.nan))
    feats["volume_z"] = (
        log_vol - log_vol.rolling(63).mean()
    ) / log_vol.rolling(63).std()

    # --- Market sensitivity: rolling beta against SPY.
    cov = rets.rolling(WARMUP_DAYS).cov(market_returns)
    var = market_returns.rolling(WARMUP_DAYS).var()
    feats["beta_63d"] = cov.div(var, axis=0)

    return feats


def cross_sectional_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Winsorise then z-score **across stocks within each day** (i.e. row-wise).

    Every operation is along ``axis=1``. If any of these accidentally ran down a
    column, the feature would be standardised across time for one stock, which is a
    different -- and for this problem, wrong -- transformation.
    """
    lo = frame.quantile(WINSOR_Q, axis=1)
    hi = frame.quantile(1 - WINSOR_Q, axis=1)
    clipped = frame.clip(lower=lo, upper=hi, axis=0)

    mean = clipped.mean(axis=1)
    std = clipped.std(axis=1)
    # A day where every stock shares one value has no cross-sectional information;
    # leave it at zero rather than dividing by ~0 and manufacturing huge values.
    std = std.where(std > 1e-12)

    return clipped.sub(mean, axis=0).div(std, axis=0)


def build_features(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the normalised feature table in long format.

    Returns ``[date, ticker] + FEATURE_NAMES``, one row per (day, stock) that has a
    complete set of features. Rows with any missing feature are dropped, so a stock
    simply is not a node on days when it lacks the history to describe it.
    """
    if panel is None:
        panel = load_prices()

    prices = to_wide(panel, "close")
    volumes = to_wide(panel, "volume")

    if MARKET_PROXY not in prices.columns:
        raise ValueError(f"{MARKET_PROXY} missing; features need a market benchmark")
    market_returns = np.log(prices[MARKET_PROXY]).diff()

    stocks = prices.drop(columns=[MARKET_PROXY])
    volumes = volumes.drop(columns=[MARKET_PROXY], errors="ignore")

    raw = compute_raw_features(stocks, volumes, market_returns)
    normed = {name: cross_sectional_normalize(raw[name]) for name in FEATURE_NAMES}

    long = pd.concat(
        [
            frame.stack(future_stack=True).rename(name)
            for name, frame in normed.items()
        ],
        axis=1,
    )
    long.index = long.index.set_names(["date", "ticker"])
    long = long.reset_index().dropna(subset=FEATURE_NAMES)

    # Drop the warm-up period, where only long-listed stocks would have features and
    # the cross-section is therefore unrepresentative.
    first_valid = long.date.min() + pd.Timedelta(days=WARMUP_DAYS * 1.5)
    long = long[long.date >= first_valid]

    return long.sort_values(["date", "ticker"]).reset_index(drop=True)


def feature_report(features: pd.DataFrame) -> pd.DataFrame:
    """Per-feature diagnostics. Cross-sectional z-scores should sit near mean 0, sd 1."""
    stats = features[FEATURE_NAMES].agg(["mean", "std", "min", "max"]).T
    stats["abs_gt_3"] = [
        f"{(features[f].abs() > 3).mean() * 100:.2f}%" for f in FEATURE_NAMES
    ]
    return stats.round(3)


if __name__ == "__main__":
    feats = build_features()
    pd.set_option("display.width", 130)

    print(f"Features: {len(feats):,} rows | {feats.date.nunique():,} days | "
          f"{feats.ticker.nunique()} tickers | {len(FEATURE_NAMES)} features")
    print(f"Range:    {feats.date.min().date()} to {feats.date.max().date()}\n")

    print("Per-feature distribution (post cross-sectional z-score)")
    print(feature_report(feats).to_string())

    print("\nFeature correlation (values above 0.8 mean redundancy)")
    corr = feats[FEATURE_NAMES].corr()
    print(corr.round(2).to_string())

    high = [
        (a, b, corr.loc[a, b])
        for i, a in enumerate(FEATURE_NAMES)
        for b in FEATURE_NAMES[i + 1 :]
        if abs(corr.loc[a, b]) > 0.8
    ]
    print(f"\nRedundant pairs (|corr| > 0.8): {high if high else 'none'}")
