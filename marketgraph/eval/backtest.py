"""Long-short backtest: turn a ranking into a portfolio, then charge it for trading.

The strategy each day: go long the top decile of predicted scores, short the bottom
decile, equal-weighted within each leg, dollar-neutral across them. Because the label
is defined against the cross-sectional median, this is the trade the model was
literally trained to make.

Why costs decide everything
---------------------------
A daily-rebalanced long-short book replaces most of its positions every day.
Turnover of ~100% a day at 5 bps per unit traded is ~5 bps daily, which compounds to
roughly 12% a year of drag -- comfortably larger than the gross edge that a rank IC
of 0.02-0.05 produces. So this module reports the equity curve **twice**, gross and
net, and the honest headline number is the net one.

``COST_BPS = 5`` is a reasonable estimate for liquid US large caps: roughly half the
quoted spread plus commission. It is optimistic in that it ignores market impact, and
it ignores borrow fees on the short leg entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COST_BPS = 5.0
TRADING_DAYS = 252


def build_positions(
    preds: pd.DataFrame, quantile: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily weights and realised returns as aligned (date x ticker) matrices.

    Each stock gets +1/n if it is in the top decile of that day's scores, -1/n if in
    the bottom decile, and 0 otherwise. Legs are scaled to 1.0 of capital each, so the
    book is dollar-neutral and the market's own move cancels between them.

    Ranking is done row-wise on the pivoted score matrix rather than with
    ``groupby.apply``: it is far quicker over 2,500 days, and the decile cut adapts to
    however many stocks actually traded that day.
    """
    scores = preds.pivot(index="date", columns="ticker", values="score")
    returns = preds.pivot(index="date", columns="ticker", values="fwd_ret")

    # rank leaves NaN in place, so absent stocks are never selected into a leg.
    ranks = scores.rank(axis=1, method="first")
    n_valid = scores.notna().sum(axis=1)
    n = (n_valid * quantile).astype(int).clip(lower=1)

    long_leg = ranks.gt(n_valid - n, axis=0)
    short_leg = ranks.le(n, axis=0)

    weights = long_leg.astype(float).div(n, axis=0) - short_leg.astype(float).div(n, axis=0)
    return weights.fillna(0.0), returns.fillna(0.0)


def run_backtest(
    preds: pd.DataFrame, cost_bps: float = COST_BPS, quantile: float = 0.1
) -> pd.DataFrame:
    """Daily gross return, turnover, cost and net return.

    Turnover is the sum of absolute weight changes between consecutive days, which
    correctly charges for both opening and closing a position. On the first day the
    whole book is new, so turnover is the full gross exposure.
    """
    weights, returns = build_positions(preds, quantile=quantile)

    gross = (weights * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()

    cost = turnover * (cost_bps / 1e4)
    net = gross - cost

    return pd.DataFrame(
        {
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": net,
            "gross_equity": (1 + gross).cumprod(),
            "net_equity": (1 + net).cumprod(),
        }
    )


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def summarize(bt: pd.DataFrame, name: str = "") -> dict[str, float]:
    """Headline statistics for a backtest, gross and net."""
    years = len(bt) / TRADING_DAYS

    def _annual(series: pd.Series) -> tuple[float, float]:
        total = float(series.add(1).prod())
        cagr = total ** (1 / years) - 1 if years > 0 and total > 0 else -1.0
        sharpe = (
            float(series.mean() / series.std() * np.sqrt(TRADING_DAYS))
            if series.std() > 0
            else 0.0
        )
        return cagr, sharpe

    gross_cagr, gross_sharpe = _annual(bt.gross_return)
    net_cagr, net_sharpe = _annual(bt.net_return)

    # The cost level at which the gross edge is exactly consumed:
    #   mean(gross) = mean(turnover) * c   ->   c = mean(gross) / mean(turnover)
    # This is the single most useful number in the table. It says what execution
    # quality the signal would need to be viable, independent of the cost we assumed.
    mean_turnover = float(bt.turnover.mean())
    break_even = (
        float(bt.gross_return.mean() / mean_turnover * 1e4) if mean_turnover > 0 else 0.0
    )

    return {
        "model": name,
        "break_even_bps": break_even,
        "gross_cagr_pct": gross_cagr * 100,
        "net_cagr_pct": net_cagr * 100,
        "gross_sharpe": gross_sharpe,
        "net_sharpe": net_sharpe,
        "cost_drag_pct_yr": float(bt.cost.mean() * TRADING_DAYS * 100),
        "daily_turnover_pct": float(bt.turnover.mean() * 100),
        "net_max_drawdown_pct": _max_drawdown(bt.net_equity) * 100,
        "hit_rate_pct": float((bt.net_return > 0).mean() * 100),
        "n_days": len(bt),
    }


def summary_table(rows: list[dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("model").round(2)
