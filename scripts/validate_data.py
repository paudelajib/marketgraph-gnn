#!/usr/bin/env python3
"""Sanity-check the price cache before we build features on top of it.

Each check targets a specific way daily price data is silently wrong. A green run
does not prove the data is perfect -- it proves the four most common and most
damaging failures are absent.

    python scripts/validate_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from marketgraph.data.ingest import load_prices, to_wide
from marketgraph.data.universe import MARKET_PROXY, get_sector_map

# Well-known splits. If adjustment worked, the adjusted series shows NO jump on
# these dates -- a 10:1 split would otherwise look like a -90% single-day return.
KNOWN_SPLITS = [
    ("NVDA", "2024-06-10", "10:1"),
    ("AAPL", "2020-08-31", "4:1"),
    ("TSLA", "2022-08-25", "3:1"),
    ("AMZN", "2022-06-06", "20:1"),
    ("GOOGL", "2022-07-18", "20:1"),
]

_results: list[tuple[bool, str]] = []


def check(passed: bool, label: str, detail: str = "") -> None:
    _results.append((passed, label))
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    if detail:
        print(detail)


def check_splits(rets: pd.DataFrame) -> None:
    """A split that was not adjusted shows up as a huge fake one-day loss."""
    print("\n1. Split adjustment (adjusted series must show no artificial jump)")
    worst = 0.0
    lines = []
    for ticker, day, ratio in KNOWN_SPLITS:
        if ticker not in rets.columns:
            continue
        d = pd.Timestamp(day)
        window = rets[ticker].loc[d - pd.Timedelta(days=4) : d + pd.Timedelta(days=2)]
        mx = float(window.abs().max())
        worst = max(worst, mx)
        lines.append(f"        {ticker:6s} {day} {ratio:>5s} split -> max |return| {mx * 100:5.2f}%")
    # A real trading day rarely exceeds 20%; an unadjusted split exceeds 50%.
    check(worst < 0.20, "no unadjusted splits detected", "\n".join(lines))


def check_extremes(rets: pd.DataFrame) -> None:
    """Extreme moves should be recognizable real events, not data glitches."""
    print("\n2. Extreme daily moves (should map to real, nameable events)")
    stacked = rets.stack(future_stack=True).dropna()
    ex = stacked[stacked.abs() > 0.30].sort_values()
    lines = [f"        {d.date()} {t:6s} {v * 100:+6.1f}%" for (d, t), v in ex.items()]
    # A handful over 16 years is normal. Hundreds means the series is broken.
    check(len(ex) < 40, f"{len(ex)} moves beyond +/-30% (expect a small handful)",
          "\n".join(lines[:15]))


def check_volatility(rets: pd.DataFrame) -> None:
    """Annualized vol outside [8%, 100%] for a large cap signals a bad series."""
    print("\n3. Annualized volatility within a plausible band for large caps")
    vol = (rets.std() * np.sqrt(252)).sort_values()
    lo, hi = vol.iloc[0], vol.iloc[-1]
    detail = (
        f"        lowest   {vol.index[0]:6s} {lo:.1%}\n"
        f"        median            {vol.median():.1%}\n"
        f"        highest  {vol.index[-1]:6s} {hi:.1%}"
    )
    check(lo > 0.08 and hi < 1.00, "all volatilities plausible", detail)


def check_sector_structure(rets: pd.DataFrame) -> None:
    """The signal the GNN will exploit must actually be present in the data.

    If stocks in the same sector are not more correlated than stocks in different
    sectors, there is no relational structure to learn and the whole premise of the
    project is wrong. This is the single most important check here.
    """
    print("\n4. Sector structure in correlations (the GNN's premise)")
    sector = get_sector_map()
    recent = rets.loc["2023":].dropna(axis=1, how="any")
    cols = [c for c in recent.columns if c in sector]
    corr = recent[cols].corr().to_numpy()

    same = np.array([[sector[a] == sector[b] for b in cols] for a in cols])
    offdiag = ~np.eye(len(cols), dtype=bool)

    intra = corr[same & offdiag].mean()
    inter = corr[~same & offdiag].mean()
    detail = (
        f"        same sector      {intra:.3f}\n"
        f"        different sector {inter:.3f}\n"
        f"        gap              {intra - inter:+.3f}"
    )
    check(intra > inter + 0.02, "same-sector pairs are more correlated", detail)


def check_calendar(panel: pd.DataFrame, rets: pd.DataFrame) -> None:
    """Weekend rows or a wrong session count means the calendar is corrupted."""
    print("\n5. Trading calendar")
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    weekend = int((dates.dayofweek >= 5).sum())
    per_year = dates.to_series().groupby(dates.year).size()
    full_years = per_year.iloc[1:-1]  # first/last years are partial
    dupes = int(panel.duplicated(["date", "ticker"]).sum())

    detail = (
        f"        sessions         {len(dates):,}\n"
        f"        weekend rows     {weekend}\n"
        f"        duplicate rows   {dupes}\n"
        f"        sessions/year    {full_years.min()}-{full_years.max()} (expect ~250-252)"
    )
    ok = weekend == 0 and dupes == 0 and full_years.between(245, 255).all()
    check(ok, "calendar is clean", detail)


def check_market_proxy(rets: pd.DataFrame) -> None:
    """Our 115 names should track SPY closely; if not, the panel is unrepresentative."""
    print("\n6. Universe tracks the market proxy")
    if MARKET_PROXY not in rets.columns:
        check(False, f"{MARKET_PROXY} missing from panel")
        return
    stocks = rets.drop(columns=[MARKET_PROXY])
    equal_weight = stocks.mean(axis=1)
    corr = float(equal_weight.corr(rets[MARKET_PROXY]))
    check(corr > 0.90, f"equal-weight universe vs {MARKET_PROXY}: corr = {corr:.3f}")


def main() -> int:
    panel = load_prices()
    prices = to_wide(panel, "close")
    rets = np.log(prices).diff()

    print(f"Panel: {len(panel):,} rows | {panel.ticker.nunique()} tickers | "
          f"{panel.date.min().date()} to {panel.date.max().date()}")

    check_splits(rets)
    check_extremes(rets)
    check_volatility(rets)
    check_sector_structure(rets)
    check_calendar(panel, rets)
    check_market_proxy(rets)

    failed = [name for ok, name in _results if not ok]
    print(f"\n{'=' * 62}")
    if failed:
        print(f"{len(failed)} of {len(_results)} checks FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print(f"All {len(_results)} checks passed. Data is safe to build features on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
