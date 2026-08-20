"""Download daily OHLCV bars from Yahoo Finance into a local Parquet cache.

The output is a **long/tidy** frame -- one row per (date, ticker) -- rather than the
wide one-column-per-ticker layout that spreadsheets encourage:

    date        ticker   open    high    low     close   volume
    2010-01-04  AAPL     6.43    6.47    6.40    6.45    493729600
    2010-01-04  MSFT     23.12   23.29   22.98   23.20   38409100

Long format is worth the reshape: adding a ticker adds rows instead of columns, every
groupby-per-ticker operation in Phase 1 becomes natural, and Parquet compresses it far
better because each column holds a single dtype.

Two adjustment subtleties that matter for correctness
-----------------------------------------------------
* We pass ``auto_adjust=True``, so OHLC are **back-adjusted** for splits *and*
  dividends, and ``close`` is what other sources call "adjusted close". Returns
  computed from this series are total returns, which is what we want.

* Back-adjusted prices are **retroactively restated**: today's dividend rewrites every
  historical price. So the price *level* you see for 2015 depends on events after 2015.
  This is harmless for us because every feature we build is a *return* or a
  cross-sectional rank, both of which are invariant to the adjustment factor. It would
  be a genuine look-ahead bug if we fed raw price levels to a model -- one more reason
  not to.

  The same restatement is why ``--full`` exists: incremental updates are a development
  convenience, but a periodic full refresh keeps history consistent with the present.
"""

from __future__ import annotations

import time
from datetime import date

import pandas as pd
import yfinance as yf

from marketgraph.config import PRICES_PATH, DEFAULT_START, ensure_dirs
from marketgraph.data.universe import get_tickers

# Yahoo rate-limits aggressive callers. Batching keeps the number of HTTP round
# trips low while staying well inside what a single request will serve.
BATCH_SIZE = 25
MAX_RETRIES = 3
RETRY_SLEEP_S = 5.0

OHLCV_COLS = ["open", "high", "low", "close", "volume"]
SCHEMA = ["date", "ticker"] + OHLCV_COLS


def _to_long(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert yfinance's (Price, Ticker) MultiIndex frame into the long schema.

    yfinance returns MultiIndex columns even for a single ticker, so this path is
    uniform and needs no special case.
    """
    if wide.empty:
        return pd.DataFrame(columns=SCHEMA)

    long = (
        wide.stack(level="Ticker", future_stack=True)
        .rename_axis(index=["date", "ticker"])
        .reset_index()
    )
    long.columns = [str(c).lower() for c in long.columns]

    # Pre-IPO and post-delisting rows come back as all-NaN; drop them so that a
    # ticker's history simply starts when it starts.
    long = long.dropna(subset=["open", "high", "low", "close"], how="all")

    missing = [c for c in SCHEMA if c not in long.columns]
    if missing:
        raise ValueError(f"yfinance response is missing columns {missing}")

    long["date"] = pd.to_datetime(long["date"]).dt.normalize()
    long["ticker"] = long["ticker"].astype(str)
    return long[SCHEMA]


def _download_batch(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Download one batch with retries. Returns an empty frame if all attempts fail."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            wide = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )
            return _to_long(wide)
        except Exception as exc:  # noqa: BLE001 - network layer throws many types
            if attempt == MAX_RETRIES:
                print(f"  ! batch failed after {MAX_RETRIES} attempts: {exc}")
                return pd.DataFrame(columns=SCHEMA)
            print(f"  . attempt {attempt} failed ({exc}); retrying in {RETRY_SLEEP_S}s")
            time.sleep(RETRY_SLEEP_S)
    return pd.DataFrame(columns=SCHEMA)


def fetch_prices(
    tickers: list[str],
    start: str = DEFAULT_START,
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily bars for ``tickers`` over [start, end) in the long schema.

    ``end`` is exclusive, following the yfinance convention. ``None`` means "up to
    the most recent available bar".
    """
    frames = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i : i + BATCH_SIZE]
        print(f"  fetching {i + 1}-{i + len(batch)} of {len(tickers)}: "
              f"{batch[0]}...{batch[-1]}")
        frames.append(_download_batch(batch, start, end))

    if not frames:
        return pd.DataFrame(columns=SCHEMA)

    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        return out

    returned = set(out.ticker.unique())
    absent = [t for t in tickers if t not in returned]
    if absent:
        print(f"  ! no data returned for: {', '.join(absent)}")

    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_prices() -> pd.DataFrame:
    """Read the cached price panel. Raises if it has not been built yet."""
    if not PRICES_PATH.exists():
        raise FileNotFoundError(
            f"No price cache at {PRICES_PATH}. Run: python scripts/fetch_data.py"
        )
    df = pd.read_parquet(PRICES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def update_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    full_refresh: bool = False,
) -> pd.DataFrame:
    """Bring the local Parquet cache up to date and return the full panel.

    Incremental by default: only bars after the last cached date are fetched, plus
    complete history for any ticker not yet in the cache. ``full_refresh=True``
    re-downloads everything, which is what you want periodically so that dividend
    restatements propagate into history.
    """
    ensure_dirs()
    tickers = tickers or get_tickers(include_market=True)

    if full_refresh or not PRICES_PATH.exists():
        print(f"Full download: {len(tickers)} tickers from {start}")
        panel = fetch_prices(tickers, start=start)
    else:
        cached = load_prices()
        known = set(cached.ticker.unique())
        new_tickers = [t for t in tickers if t not in known]
        last_date = cached.date.max()

        # Re-fetch the trailing few sessions as well as anything genuinely new: the
        # most recent bar can still be provisional while the session is open.
        resume_from = (last_date - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        print(f"Cache holds {len(known)} tickers through {last_date.date()}")

        frames = [cached]
        existing = [t for t in tickers if t in known]
        if existing:
            print(f"Incremental update from {resume_from}")
            frames.append(fetch_prices(existing, start=resume_from))
        if new_tickers:
            print(f"Backfilling {len(new_tickers)} new tickers from {start}")
            frames.append(fetch_prices(new_tickers, start=start))

        panel = pd.concat(frames, ignore_index=True)

    if panel.empty:
        raise RuntimeError("Download returned no data at all -- check connectivity.")

    # Newest row wins for any (date, ticker) collision, so a re-fetch corrects a
    # provisional bar rather than duplicating it.
    panel = (
        panel.drop_duplicates(subset=["date", "ticker"], keep="last")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )

    panel.to_parquet(PRICES_PATH, index=False, compression="snappy")
    size_mb = PRICES_PATH.stat().st_size / 1e6
    print(
        f"\nWrote {len(panel):,} rows x {panel.ticker.nunique()} tickers "
        f"({panel.date.min().date()} to {panel.date.max().date()}) "
        f"-> {PRICES_PATH.name}, {size_mb:.1f} MB"
    )
    return panel


def coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker summary: history span, row count, and gaps against the trading calendar.

    ``missing_days`` counts sessions where the broader panel traded but this ticker
    did not -- after its first bar, so a late IPO does not read as missing data.
    Anything above zero is a halt, a data hole, or a ticker change worth eyeballing.
    """
    all_sessions = pd.Index(sorted(panel.date.unique()), name="date")

    rows = []
    for ticker, g in panel.groupby("ticker", sort=True):
        first, last = g.date.min(), g.date.max()
        expected = all_sessions[(all_sessions >= first) & (all_sessions <= last)]
        rows.append(
            {
                "ticker": ticker,
                "start": first.date(),
                "end": last.date(),
                "rows": len(g),
                "years": round((last - first).days / 365.25, 1),
                "missing_days": len(expected) - len(g),
                "zero_volume_days": int((g.volume == 0).sum()),
                "nan_close": int(g.close.isna().sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(["years", "ticker"]).reset_index(drop=True)


def to_wide(panel: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """Pivot one field to a (date x ticker) matrix -- the layout Phase 1 models want."""
    return panel.pivot(index="date", columns="ticker", values=field).sort_index()
