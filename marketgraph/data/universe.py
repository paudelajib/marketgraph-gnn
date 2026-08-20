"""The stock universe: a fixed list of US-listed large caps with sector labels.

Design decisions worth understanding, because they shape every result downstream:

1. **The universe is hardcoded, not fetched.** A universe pulled live from an index
   provider changes under you, which makes experiments irreproducible. Freezing it
   in source control means a run from six months ago is comparable to one today.

2. **Survivorship bias is present and we are explicit about it.** These are large
   caps that exist *now*. Companies that were large caps in 2012 and then blew up
   (or were acquired) are absent, so backtests on this universe are optimistically
   biased. Removing the bias properly needs point-in-time index membership, which
   is paid data. The honest move is to state the bias, not to hide it -- and it is
   less severe here than it looks, because we predict *cross-sectional* direction
   (relative rank within the universe) rather than absolute returns.

3. **No ADRs.** TSM, BABA, ASML, NVO, SAP, TM and friends are excluded even though
   they are enormous. Their home markets close hours before the US open, so their
   US-listed prices largely *echo* moves that already happened overnight. In a
   next-day prediction task that quietly leaks future information into the graph.

4. **No dual-class duplicates.** GOOGL but not GOOG, BRK-B but not BRK-A. Two share
   classes of one company correlate at ~0.99 and would dominate any correlation
   graph with an edge that teaches the model nothing.

5. **Yahoo ticker conventions.** Yahoo writes class shares with a hyphen (`BRK-B`,
   not `BRK.B`). We store the Yahoo form as the canonical id to avoid a translation
   layer.
"""

from __future__ import annotations

import pandas as pd

# (ticker, company name, GICS sector)
_UNIVERSE: list[tuple[str, str, str]] = [
    # --- Information Technology ---
    ("AAPL", "Apple", "Information Technology"),
    ("MSFT", "Microsoft", "Information Technology"),
    ("NVDA", "NVIDIA", "Information Technology"),
    ("AVGO", "Broadcom", "Information Technology"),
    ("ORCL", "Oracle", "Information Technology"),
    ("CRM", "Salesforce", "Information Technology"),
    ("AMD", "Advanced Micro Devices", "Information Technology"),
    ("ADBE", "Adobe", "Information Technology"),
    ("ACN", "Accenture", "Information Technology"),
    ("CSCO", "Cisco Systems", "Information Technology"),
    ("INTC", "Intel", "Information Technology"),
    ("TXN", "Texas Instruments", "Information Technology"),
    ("QCOM", "Qualcomm", "Information Technology"),
    ("IBM", "IBM", "Information Technology"),
    ("NOW", "ServiceNow", "Information Technology"),
    ("INTU", "Intuit", "Information Technology"),
    ("AMAT", "Applied Materials", "Information Technology"),
    ("MU", "Micron Technology", "Information Technology"),
    ("LRCX", "Lam Research", "Information Technology"),
    ("ADI", "Analog Devices", "Information Technology"),
    # --- Communication Services ---
    ("GOOGL", "Alphabet", "Communication Services"),
    ("META", "Meta Platforms", "Communication Services"),
    ("NFLX", "Netflix", "Communication Services"),
    ("DIS", "Walt Disney", "Communication Services"),
    ("CMCSA", "Comcast", "Communication Services"),
    ("VZ", "Verizon", "Communication Services"),
    ("T", "AT&T", "Communication Services"),
    ("TMUS", "T-Mobile US", "Communication Services"),
    # --- Consumer Discretionary ---
    ("AMZN", "Amazon", "Consumer Discretionary"),
    ("TSLA", "Tesla", "Consumer Discretionary"),
    ("HD", "Home Depot", "Consumer Discretionary"),
    ("MCD", "McDonald's", "Consumer Discretionary"),
    ("NKE", "Nike", "Consumer Discretionary"),
    ("LOW", "Lowe's", "Consumer Discretionary"),
    ("SBUX", "Starbucks", "Consumer Discretionary"),
    ("BKNG", "Booking Holdings", "Consumer Discretionary"),
    ("TJX", "TJX Companies", "Consumer Discretionary"),
    ("GM", "General Motors", "Consumer Discretionary"),
    ("F", "Ford Motor", "Consumer Discretionary"),
    ("CMG", "Chipotle Mexican Grill", "Consumer Discretionary"),
    # --- Consumer Staples ---
    ("WMT", "Walmart", "Consumer Staples"),
    ("PG", "Procter & Gamble", "Consumer Staples"),
    ("KO", "Coca-Cola", "Consumer Staples"),
    ("PEP", "PepsiCo", "Consumer Staples"),
    ("COST", "Costco Wholesale", "Consumer Staples"),
    ("PM", "Philip Morris International", "Consumer Staples"),
    ("MO", "Altria Group", "Consumer Staples"),
    ("MDLZ", "Mondelez International", "Consumer Staples"),
    ("CL", "Colgate-Palmolive", "Consumer Staples"),
    ("TGT", "Target", "Consumer Staples"),
    # --- Financials ---
    ("BRK-B", "Berkshire Hathaway", "Financials"),
    ("JPM", "JPMorgan Chase", "Financials"),
    ("V", "Visa", "Financials"),
    ("MA", "Mastercard", "Financials"),
    ("BAC", "Bank of America", "Financials"),
    ("WFC", "Wells Fargo", "Financials"),
    ("GS", "Goldman Sachs", "Financials"),
    ("MS", "Morgan Stanley", "Financials"),
    ("SPGI", "S&P Global", "Financials"),
    ("BLK", "BlackRock", "Financials"),
    ("AXP", "American Express", "Financials"),
    ("C", "Citigroup", "Financials"),
    ("SCHW", "Charles Schwab", "Financials"),
    ("CB", "Chubb", "Financials"),
    ("PGR", "Progressive", "Financials"),
    # --- Health Care ---
    ("LLY", "Eli Lilly", "Health Care"),
    ("UNH", "UnitedHealth Group", "Health Care"),
    ("JNJ", "Johnson & Johnson", "Health Care"),
    ("ABBV", "AbbVie", "Health Care"),
    ("MRK", "Merck", "Health Care"),
    ("TMO", "Thermo Fisher Scientific", "Health Care"),
    ("ABT", "Abbott Laboratories", "Health Care"),
    ("PFE", "Pfizer", "Health Care"),
    ("DHR", "Danaher", "Health Care"),
    ("BMY", "Bristol-Myers Squibb", "Health Care"),
    ("AMGN", "Amgen", "Health Care"),
    ("GILD", "Gilead Sciences", "Health Care"),
    ("CVS", "CVS Health", "Health Care"),
    ("MDT", "Medtronic", "Health Care"),
    ("ISRG", "Intuitive Surgical", "Health Care"),
    ("VRTX", "Vertex Pharmaceuticals", "Health Care"),
    # --- Industrials ---
    ("GE", "GE Aerospace", "Industrials"),
    ("CAT", "Caterpillar", "Industrials"),
    ("RTX", "RTX Corporation", "Industrials"),
    ("HON", "Honeywell International", "Industrials"),
    ("UNP", "Union Pacific", "Industrials"),
    ("BA", "Boeing", "Industrials"),
    ("DE", "Deere & Company", "Industrials"),
    ("LMT", "Lockheed Martin", "Industrials"),
    ("UPS", "United Parcel Service", "Industrials"),
    ("ADP", "Automatic Data Processing", "Industrials"),
    ("ETN", "Eaton", "Industrials"),
    ("MMM", "3M", "Industrials"),
    # --- Energy ---
    ("XOM", "Exxon Mobil", "Energy"),
    ("CVX", "Chevron", "Energy"),
    ("COP", "ConocoPhillips", "Energy"),
    ("EOG", "EOG Resources", "Energy"),
    ("SLB", "SLB", "Energy"),
    ("MPC", "Marathon Petroleum", "Energy"),
    ("PSX", "Phillips 66", "Energy"),
    # --- Utilities ---
    ("NEE", "NextEra Energy", "Utilities"),
    ("DUK", "Duke Energy", "Utilities"),
    ("SO", "Southern Company", "Utilities"),
    ("D", "Dominion Energy", "Utilities"),
    ("AEP", "American Electric Power", "Utilities"),
    # --- Real Estate ---
    ("PLD", "Prologis", "Real Estate"),
    ("AMT", "American Tower", "Real Estate"),
    ("EQIX", "Equinix", "Real Estate"),
    ("CCI", "Crown Castle", "Real Estate"),
    ("PSA", "Public Storage", "Real Estate"),
    # --- Materials ---
    ("LIN", "Linde", "Materials"),
    ("SHW", "Sherwin-Williams", "Materials"),
    ("APD", "Air Products and Chemicals", "Materials"),
    ("ECL", "Ecolab", "Materials"),
    ("FCX", "Freeport-McMoRan", "Materials"),
]

# Broad-market proxy. Not a node in the graph -- used later to compute
# market-relative features (beta, idiosyncratic return) in Phase 1.
MARKET_PROXY = "SPY"


def get_universe() -> pd.DataFrame:
    """Return the universe as a DataFrame with columns [ticker, name, sector]."""
    return pd.DataFrame(_UNIVERSE, columns=["ticker", "name", "sector"])


def get_tickers(include_market: bool = False) -> list[str]:
    """Return the list of ticker symbols, optionally with the market proxy appended."""
    tickers = [t for t, _, _ in _UNIVERSE]
    if include_market:
        tickers.append(MARKET_PROXY)
    return tickers


def get_sector_map() -> dict[str, str]:
    """Map ticker -> GICS sector. Used to build the sector graph in Phase 3."""
    return {t: sector for t, _, sector in _UNIVERSE}


if __name__ == "__main__":
    uni = get_universe()
    print(f"{len(uni)} tickers across {uni.sector.nunique()} sectors\n")
    print(uni.sector.value_counts().to_string())
