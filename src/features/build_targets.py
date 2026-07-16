"""
src/features/build_targets.py
------------------------------
Computes the two target variables for FinSignal from raw price data.

Module 1 target: Post-Earnings Return Direction:
  • 1-day and 3-day log-return after the earnings date
  • Binary label: 1 = positive return, 0 = negative return
  • ⚠️  Only computed AFTER the earnings date, no look-ahead leakage

Module 2 target: Realized Volatility:
  • Rolling 21-day realized volatility (annualized std dev of log-returns)
  • Also compute the 5-day forward realized vol (the prediction target)
  • ⚠️  Forward vol uses only future dates relative to prediction point

Output: saves data/processed/module1_targets.csv + module2_features.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PRICES_DIR  = Path(__file__).resolve().parents[2] / "data" / "raw" / "prices"
PROCESSED_DIR   = Path(__file__).resolve().parents[2] / "data" / "processed"

# Approximate earnings dates for 2020–2024.
# In production you'd pull these from a financial data API (e.g. Nasdaq Data Link).
# These are Q1–Q4 approximate report dates (companies report ~4 weeks after quarter end).
# We'll use these as anchors; the notebook will refine them.
SAMPLE_EARNINGS_DATES = {
    "AAPL": ["2020-04-30", "2020-07-30", "2020-10-29", "2021-01-27",
              "2021-04-28", "2021-07-27", "2021-10-28", "2022-01-27",
              "2022-04-28", "2022-07-28", "2022-10-27", "2023-02-02",
              "2023-05-04", "2023-08-03", "2023-11-02", "2024-02-01"],
    "MSFT": ["2020-04-29", "2020-07-22", "2020-10-28", "2021-01-27",
              "2021-04-28", "2021-07-27", "2021-10-27", "2022-01-25",
              "2022-04-26", "2022-07-26", "2022-10-25", "2023-01-24",
              "2023-04-25", "2023-07-25", "2023-10-25", "2024-01-30"],
    "JPM":  ["2020-04-14", "2020-07-14", "2020-10-13", "2021-01-15",
              "2021-04-14", "2021-07-13", "2021-10-13", "2022-01-14",
              "2022-04-13", "2022-07-14", "2022-10-14", "2023-01-13",
              "2023-04-14", "2023-07-14", "2023-10-13", "2024-01-12"],
    "JNJ":  ["2020-04-14", "2020-07-16", "2020-10-13", "2021-01-26",
              "2021-04-20", "2021-07-20", "2021-10-19", "2022-01-25",
              "2022-04-19", "2022-07-19", "2022-10-18", "2023-01-24",
              "2023-04-18", "2023-07-18", "2023-10-17", "2024-01-23"],
    "NVDA": ["2020-05-21", "2020-08-19", "2020-11-18", "2021-02-24",
              "2021-05-26", "2021-08-18", "2021-11-17", "2022-02-16",
              "2022-05-25", "2022-08-24", "2022-11-16", "2023-02-22",
              "2023-05-24", "2023-08-23", "2023-11-21", "2024-02-21"],
}


def load_prices(ticker: str) -> pd.DataFrame:
    """Load raw price CSV for a ticker. Returns DataFrame with DatetimeIndex."""
    path = RAW_PRICES_DIR / f"{ticker}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No price file for {ticker}. Run fetch_prices.py first.")
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df.sort_index(inplace=True)
    return df


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log-returns from a price series."""
    return np.log(prices / prices.shift(1))


def compute_realized_vol(log_returns: pd.Series,
                          window: int = 21,
                          annualize: bool = True) -> pd.Series:
    """
    Rolling realized volatility = std dev of log-returns over `window` days.
    Annualized by multiplying by sqrt(252), the number of trading days/year.

    Parameters
    ----------
    log_returns : pd.Series
        Daily log-returns.
    window : int
        Lookback window in trading days. 21 ≈ 1 month.
    annualize : bool
        If True, multiply by sqrt(252) to express as annualized vol.
    """
    vol = log_returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def build_module1_targets(ticker: str,
                           earnings_dates: list[str],
                           horizons: list[int] = [1, 3]) -> pd.DataFrame:
    """
    For each earnings date, compute post-earnings log-returns and binary direction labels.

    ⚠️  Look-ahead safety: we only look FORWARD from the earnings date.
    The training label for a given earnings call is computed from prices
    that occur AFTER the call: this is the correct causal ordering.

    Parameters
    ----------
    ticker : str
    earnings_dates : list[str]
        List of earnings announcement dates in 'YYYY-MM-DD' format.
    horizons : list[int]
        Number of trading days after earnings to measure returns.

    Returns
    -------
    pd.DataFrame
        One row per earnings date. Columns: ret_1d, ret_3d, label_1d, label_3d, etc.
    """
    df_prices = load_prices(ticker)
    log_rets  = compute_log_returns(df_prices["Close"])

    records = []
    trading_days = df_prices.index  # only actual trading days

    for date_str in earnings_dates:
        earnings_dt = pd.Timestamp(date_str)

        # Find the next trading day ON or AFTER the earnings date
        # (earnings are often announced after market close, so next-day is the move)
        next_days = trading_days[trading_days >= earnings_dt]
        if len(next_days) == 0:
            continue
        t0 = next_days[0]   # first trading day at/after announcement

        row = {"ticker": ticker, "earnings_date": earnings_dt, "t0": t0}

        for h in horizons:
            future_days = trading_days[trading_days > t0]
            if len(future_days) < h:
                row[f"ret_{h}d"]   = np.nan
                row[f"label_{h}d"] = np.nan
                continue

            t_h = future_days[h - 1]  # h-th trading day after t0

            # Cumulative log-return from t0 close to t_h close
            # ⚠️  This is future data relative to the earnings call, correct.
            cum_ret = log_rets.loc[
                (log_rets.index > t0) & (log_rets.index <= t_h)
            ].sum()

            row[f"ret_{h}d"]   = cum_ret
            row[f"label_{h}d"] = int(cum_ret > 0)   # 1 = up, 0 = down

        records.append(row)

    return pd.DataFrame(records)


def build_module2_features(ticker: str,
                            vol_window: int = 21,
                            forward_window: int = 5) -> pd.DataFrame:
    """
    Build the time-series feature matrix for Module 2 (volatility forecasting).

    Features (all use only historical data at each point, no look-ahead):
      - realized_vol_21d   : trailing 21-day realized vol (the main input)
      - log_return         : daily log-return
      - volume_norm        : volume normalized by its 21-day rolling mean

    Target (forward-looking: this is what the model predicts):
      - forward_vol_5d     : realized vol over the next 5 trading days

    ⚠️  The forward_vol column uses future prices. This column must NEVER
    be used as a feature, only as the label during training.

    Parameters
    ----------
    ticker : str
    vol_window : int
        Lookback window for trailing realized vol (default 21 days ≈ 1 month).
    forward_window : int
        Horizon for the forward vol target (default 5 days ≈ 1 week).

    Returns
    -------
    pd.DataFrame
        Daily rows. Features + target, with NaN rows dropped.
    """
    df = load_prices(ticker)
    log_rets = compute_log_returns(df["Close"])

    result = pd.DataFrame(index=df.index)
    result["ticker"]           = ticker
    result["close"]            = df["Close"]
    result["log_return"]       = log_rets
    result["realized_vol_21d"] = compute_realized_vol(log_rets, window=vol_window)
    result["volume_norm"]      = df["Volume"] / df["Volume"].rolling(vol_window).mean()

    # Forward vol: std dev of the NEXT `forward_window` returns
    # Shift by -forward_window so each row gets the future vol.
    # ⚠️  This creates look-ahead; that's intentional for the LABEL only.
    result["forward_vol_5d"] = (
        log_rets
        .rolling(forward_window)
        .std()
        .shift(-forward_window)  # align: each date now shows next-5d vol
        * np.sqrt(252)
    )

    result.dropna(inplace=True)
    return result


def build_and_save_all(tickers: list[str]) -> None:
    """Run both target builders for all tickers and save to data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Module 1
    m1_frames = []
    for ticker in tickers:
        if ticker not in SAMPLE_EARNINGS_DATES:
            print(f"  ⚠️  No earnings dates configured for {ticker}, skipping M1")
            continue
        df = build_module1_targets(ticker, SAMPLE_EARNINGS_DATES[ticker])
        m1_frames.append(df)
        print(f"  ✓ {ticker}: {len(df)} earnings events (Module 1)")

    if m1_frames:
        m1 = pd.concat(m1_frames, ignore_index=True)
        m1.to_csv(PROCESSED_DIR / "module1_targets.csv", index=False)
        print(f"\n✓ Saved module1_targets.csv: {len(m1)} rows")

    # Module 2
    m2_frames = []
    for ticker in tickers:
        try:
            df = build_module2_features(ticker)
            m2_frames.append(df)
            print(f"  ✓ {ticker}: {len(df)} trading days (Module 2)")
        except FileNotFoundError as e:
            print(f"  ⚠️  {e}")

    if m2_frames:
        m2 = pd.concat(m2_frames)
        m2.to_csv(PROCESSED_DIR / "module2_features.csv")
        print(f"\n✓ Saved module2_features.csv: {len(m2)} rows")


if __name__ == "__main__":
    TEST_TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "NVDA"]
    build_and_save_all(TEST_TICKERS)