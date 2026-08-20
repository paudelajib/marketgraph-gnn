#!/usr/bin/env python3
"""Build or update the local price cache.

    python scripts/fetch_data.py                 # incremental update
    python scripts/fetch_data.py --full          # re-download all history
    python scripts/fetch_data.py --start 2015-01-01
    python scripts/fetch_data.py --report        # coverage summary, no download
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from marketgraph.config import DEFAULT_START
from marketgraph.data.ingest import coverage_report, load_prices, update_prices
from marketgraph.data.universe import get_universe


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default=DEFAULT_START, help="first date to fetch")
    p.add_argument("--full", action="store_true",
                   help="re-download everything instead of updating incrementally")
    p.add_argument("--report", action="store_true",
                   help="print the coverage report for the existing cache and exit")
    args = p.parse_args()

    if args.report:
        panel = load_prices()
    else:
        uni = get_universe()
        print(f"Universe: {len(uni)} tickers across {uni.sector.nunique()} sectors "
              f"(+1 market proxy)\n")
        panel = update_prices(start=args.start, full_refresh=args.full)

    rep = coverage_report(panel)

    pd.set_option("display.width", 120)
    print("\n=== Coverage: shortest histories ===")
    print(rep.head(12).to_string(index=False))

    flagged = rep[(rep.missing_days > 0) | (rep.nan_close > 0) | (rep.zero_volume_days > 5)]
    print(f"\n=== Data-quality flags ({len(flagged)} tickers) ===")
    print(flagged.to_string(index=False) if len(flagged) else "none")

    full_hist = (rep.start <= pd.Timestamp(args.start).date() + pd.Timedelta(days=10)).sum()
    print(f"\n{full_hist}/{len(rep)} tickers have history back to ~{args.start}")
    print(f"Median history: {rep.years.median():.1f} years")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
