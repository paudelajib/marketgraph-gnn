#!/usr/bin/env python3
"""Does the long-short book survive market crises?

Annualised return over a decade answers the wrong question. A dollar-neutral
long-short portfolio makes a specific promise -- that its returns are independent of
the market's direction -- and the only honest test of that promise is what it did when
the market fell hard. If it holds, the strategy is worth something even at a low
return, because it diversifies an equity portfolio. If it breaks precisely during
crashes, it is a leveraged equity bet wearing a hedge fund's clothes.

Run at **zero transaction cost**, deliberately: the question here is whether the
*signal* is robust across regimes, which is separate from whether it is tradable.

Drawdown episodes are detected from SPY itself rather than hardcoded, so the analysis
does not depend on remembering which crises happened.

    python scripts/crisis_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from marketgraph.config import ARTIFACTS_DIR, DOCS_DIR, ensure_dirs
from marketgraph.data.ingest import load_prices
from marketgraph.data.universe import MARKET_PROXY
from marketgraph.eval.backtest import run_backtest

# Kept small so the figure stays readable: best heuristic, best graph model, and the
# non-graph control that isolates the graph's contribution.
MODELS = ["reversal_1d", "gcn_sector", "mlp_no_graph"]
COLORS = {"reversal_1d": "#1f77b4", "gcn_sector": "#d62728", "mlp_no_graph": "#7f7f7f"}
MIN_DRAWDOWN = 0.10
WEEK = "W-FRI"


def load_market(index: pd.DatetimeIndex) -> pd.Series:
    """SPY daily returns aligned to the out-of-sample dates."""
    prices = load_prices()
    spy = prices[prices.ticker == MARKET_PROXY].set_index("date").close.sort_index()
    return spy.pct_change().reindex(index).fillna(0.0)


def find_drawdowns(returns: pd.Series, min_depth: float = MIN_DRAWDOWN) -> pd.DataFrame:
    """Locate peak-to-trough episodes deeper than ``min_depth``.

    Walks the equity curve tracking each new high; when the drop from that high
    exceeds the threshold, the stretch from the prior peak to the eventual trough is
    recorded as one episode.
    """
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    underwater = equity / peak - 1

    episodes = []
    in_episode = False
    start = trough_date = equity.index[0]
    trough = 0.0

    for date, level in underwater.items():
        if not in_episode and level < -min_depth:
            in_episode = True
            start = peak[:date][peak[:date] == peak[date]].index[0]
            trough, trough_date = level, date
        elif in_episode:
            if level < trough:
                trough, trough_date = level, date
            if level >= -1e-9:  # recovered to the old high
                episodes.append((start, trough_date, date, trough))
                in_episode = False

    if in_episode:
        episodes.append((start, trough_date, underwater.index[-1], trough))

    return pd.DataFrame(
        episodes, columns=["peak", "trough", "recovery", "depth"]
    ).sort_values("depth")


def crisis_table(
    strategies: dict[str, pd.Series], market: pd.Series, episodes: pd.DataFrame
) -> pd.DataFrame:
    """Cumulative return of each strategy over each market drawdown, peak to trough."""
    rows = []
    for _, ep in episodes.iterrows():
        window = slice(ep.peak, ep.trough)
        row = {
            "period": f"{ep.peak.date()} to {ep.trough.date()}",
            "days": len(market[window]),
            "spy_pct": float((1 + market[window]).prod() - 1) * 100,
        }
        for name, series in strategies.items():
            row[f"{name}_pct"] = float((1 + series[window]).prod() - 1) * 100
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def market_stress_stats(
    strategies: dict[str, pd.Series], market: pd.Series
) -> pd.DataFrame:
    """Beta, correlation, and behaviour in the market's worst weeks.

    ``worst_decile_weeks`` is the mean annualised strategy return during the 10% of
    weeks in which SPY fell most -- the single most direct answer to "does it hold up
    when things break".
    """
    mkt_w = market.resample(WEEK).apply(lambda r: (1 + r).prod() - 1)
    cutoff = mkt_w.quantile(0.10)
    worst = mkt_w <= cutoff

    rows = []
    for name, series in strategies.items():
        w = series.resample(WEEK).apply(lambda r: (1 + r).prod() - 1)
        aligned = pd.concat([w, mkt_w], axis=1).dropna()
        beta = float(np.polyfit(aligned.iloc[:, 1], aligned.iloc[:, 0], 1)[0])
        rows.append(
            {
                "model": name,
                "corr_with_spy": float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])),
                "beta_to_spy": beta,
                "all_weeks_ann_pct": float(w.mean() * 52 * 100),
                "worst_decile_weeks_ann_pct": float(w[worst].mean() * 52 * 100),
                "best_week_pct": float(w.max() * 100),
                "worst_week_pct": float(w.min() * 100),
            }
        )
    rows.append(
        {
            "model": "SPY (benchmark)",
            "corr_with_spy": 1.0,
            "beta_to_spy": 1.0,
            "all_weeks_ann_pct": float(mkt_w.mean() * 52 * 100),
            "worst_decile_weeks_ann_pct": float(mkt_w[worst].mean() * 52 * 100),
            "best_week_pct": float(mkt_w.max() * 100),
            "worst_week_pct": float(mkt_w.min() * 100),
        }
    )
    return pd.DataFrame(rows).set_index("model").round(2)


def make_figure(
    strategies: dict[str, pd.Series],
    market: pd.Series,
    episodes: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    def shade(ax):
        for _, ep in episodes.iterrows():
            ax.axvspan(ep.peak, ep.trough, color="red", alpha=0.12, lw=0)

    # (a) Cumulative growth, log scale so equal percentage moves look equal.
    ax = axes[0, 0]
    for name, series in strategies.items():
        ax.plot((1 + series).cumprod(), label=name, color=COLORS[name], lw=1.4)
    ax.plot((1 + market).cumprod(), label="SPY", color="black", lw=1.6, ls="--")
    ax.set_yscale("log")
    ax.set_title("(a) Growth of $1, zero costs  (red = SPY drawdowns >10%)")
    ax.set_ylabel("growth of $1 (log scale)")
    ax.legend(fontsize=8)
    shade(ax)

    # (b) Weekly returns, annualised -- the requested view.
    ax = axes[0, 1]
    best = MODELS[0]
    w = strategies[best].resample(WEEK).apply(lambda r: (1 + r).prod() - 1) * 52 * 100
    mw = market.resample(WEEK).apply(lambda r: (1 + r).prod() - 1) * 52 * 100
    ax.plot(mw, color="black", lw=0.7, alpha=0.5, label="SPY")
    ax.plot(w, color=COLORS[best], lw=0.9, label=best)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("(b) Weekly return, annualised")
    ax.set_ylabel("annualised %")
    ax.legend(fontsize=8)
    shade(ax)

    # (c) Weekly strategy vs weekly market: the market-neutrality claim, visually.
    ax = axes[1, 0]
    aligned = pd.concat([w / 52, mw / 52], axis=1).dropna()
    aligned.columns = ["strategy", "market"]
    ax.scatter(aligned.market, aligned.strategy, s=9, alpha=0.45, color=COLORS[best])
    slope, intercept = np.polyfit(aligned.market, aligned.strategy, 1)
    xs = np.linspace(aligned.market.min(), aligned.market.max(), 50)
    ax.plot(xs, slope * xs + intercept, color="red", lw=1.5,
            label=f"beta = {slope:.3f}")
    ax.axhline(0, color="black", lw=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel("SPY weekly return (%)")
    ax.set_ylabel(f"{best} weekly return (%)")
    ax.set_title("(c) Is it actually market-neutral?")
    ax.legend(fontsize=9)

    # (d) Peak-to-trough performance during each market drawdown.
    ax = axes[1, 1]
    labels = [f"{ep.peak.date()}\n({ep.depth * 100:.0f}%)" for _, ep in episodes.iterrows()]
    x = np.arange(len(episodes))
    width = 0.8 / (len(strategies) + 1)

    spy_vals = [
        (1 + market[ep.peak : ep.trough]).prod() * 100 - 100
        for _, ep in episodes.iterrows()
    ]
    ax.bar(x, spy_vals, width, label="SPY", color="black")
    for i, (name, series) in enumerate(strategies.items(), start=1):
        vals = [
            (1 + series[ep.peak : ep.trough]).prod() * 100 - 100
            for _, ep in episodes.iterrows()
        ]
        ax.bar(x + i * width, vals, width, label=name, color=COLORS[name])

    ax.set_xticks(x + width * len(strategies) / 2)
    ax.set_xticklabels(labels, fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("return over the drawdown (%)")
    ax.set_title("(d) Performance during market drawdowns")
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.3)

    fig.suptitle(
        "Crisis robustness of the long-short book, out-of-sample 2016-2026 (zero costs)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"Saved -> {path}")


def main() -> int:
    ensure_dirs()
    frames = [
        pd.read_parquet(ARTIFACTS_DIR / f)
        for f in ("preds_baselines.parquet", "preds_gnn.parquet")
        if (ARTIFACTS_DIR / f).exists()
    ]
    preds = pd.concat(frames, ignore_index=True)

    strategies = {}
    for name in MODELS:
        bt = run_backtest(preds[preds.model == name], cost_bps=0.0)
        strategies[name] = bt.gross_return

    index = next(iter(strategies.values())).index
    market = load_market(index)
    episodes = find_drawdowns(market)

    pd.set_option("display.width", 170)

    print(f"SPY drawdowns >10% in 2016-2026: {len(episodes)}\n")
    table = crisis_table(strategies, market, episodes)
    print(table.to_string(index=False))

    print("\nMarket-neutrality and stress behaviour (weekly)")
    stats = market_stress_stats(strategies, market)
    print(stats.to_string())

    table.to_csv(DOCS_DIR / "crisis_periods.csv", index=False)
    stats.to_csv(DOCS_DIR / "crisis_stats.csv")
    make_figure(strategies, market, episodes, DOCS_DIR / "crisis_analysis.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
