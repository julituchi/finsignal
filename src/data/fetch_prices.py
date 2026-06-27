"""
src/data/fetch_prices.py
------------------------
Pulls historical OHLCV data for the FinSignal stock universe via yfinance.
Saves one CSV per ticker to data/raw/prices/.
"""

import os
import yfinance as yf
import pandas as pd
from pathlib import Path

# ── Universe ──────────────────────────────────────────────────────────────────
# 20 large-cap S&P 500 stocks with reliable earnings transcript history.
# Mix of sectors so sentiment patterns don't all look the same.
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",   # Tech
    "JPM", "GS", "BAC",                          # Financials
    "JNJ", "PFE", "UNH",                         # Healthcare
    "XOM", "CVX",                                # Energy
    "WMT", "HD", "NKE",                          # Consumer
    "TSLA", "NVDA",                              # Growth / Semi
    "V", "MA",                                   # Payments
]

# ── Config ────────────────────────────────────────────────────────────────────
START_DATE = "2020-01-01"   # 4+ years → enough for time-series CV
END_DATE   = "2024-12-31"
RAW_DIR    = Path(__file__).resolve().parents[2] / "data" / "raw" / "prices"


def fetch_and_save(tickers: list[str] = UNIVERSE,
                   start: str = START_DATE,
                   end: str = END_DATE) -> pd.DataFrame:
    """
    Download adjusted close + volume for all tickers in one yfinance call.
    Returns a combined DataFrame and saves individual CSVs per ticker.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (field, ticker) — e.g. ('Close', 'AAPL')
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusts for splits & dividends — use this, not Adj Close
        progress=True,
    )

    # ── Validation ────────────────────────────────────────────────────────────
    if raw.empty:
        raise ValueError("yfinance returned empty data. Check your internet connection.")

    missing = [t for t in tickers if t not in raw["Close"].columns]
    if missing:
        print(f"  ⚠️  No data returned for: {missing}")

    # ── Save individual CSVs ─────────────────────────────────────────────────
    for ticker in tickers:
        if ticker not in raw["Close"].columns:
            continue
        df_ticker = raw.xs(ticker, axis=1, level=1)   # shape: (days, fields)
        df_ticker.index.name = "date"
        path = RAW_DIR / f"{ticker}.csv"
        df_ticker.to_csv(path)

    print(f"  ✓ Saved {len(tickers) - len(missing)} CSVs to {RAW_DIR}")
    return raw


if __name__ == "__main__":
    fetch_and_save()