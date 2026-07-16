"""
src/features/build_nlp_features.py
------------------------------------
Merges FinBERT sentiment scores with Module 1 earnings targets
to produce the final feature matrix for the XGBoost classifier.

Logic:
  For each row in module1_targets.csv (one earnings event per ticker/date),
  find the matching sentiment JSON file and join the features.

  ⚠️  Look-ahead check: the sentiment features come from the transcript
  on the earnings date itself; the labels come from future prices.
  This is the correct causal ordering.

Output: data/processed/module1_features.csv
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR  = Path(__file__).resolve().parents[2] / "data" / "processed"
SENTIMENT_DIR  = PROCESSED_DIR / "sentiment"
TARGETS_CSV    = PROCESSED_DIR / "module1_targets.csv"
OUTPUT_CSV     = PROCESSED_DIR / "module1_features.csv"

# Sentiment features produced by sentiment.py
SENTIMENT_COLS = [
    "mean_positive", "mean_negative", "mean_neutral",
    "sentiment_score", "sentiment_std",
    "pct_positive", "pct_negative",
    "prepared_sentiment_score", "qa_sentiment_score",
    "qa_delta_score", "transcript_length",
    "prepared_n_sentences", "qa_n_sentences",
]


def load_sentiment_for_ticker(ticker: str) -> pd.DataFrame:
    """
    Load all sentiment JSON files for a ticker into a DataFrame.
    Returns DataFrame indexed by date string (YYYY-MM-DD).
    """
    ticker_dir = SENTIMENT_DIR / ticker
    if not ticker_dir.exists():
        return pd.DataFrame()

    records = []
    for json_path in sorted(ticker_dir.glob("*.json")):
        with open(json_path) as f:
            records.append(json.load(f))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def find_nearest_transcript(earnings_date: pd.Timestamp,
                              sentiment_df: pd.DataFrame,
                              max_days: int = 5) -> pd.Series | None:
    """
    Match an earnings date to the nearest transcript date within `max_days`.

    Why we need this: earnings are sometimes announced after market close
    and the 8-K filing date may differ by 1–2 days from the actual call date.
    We allow a window of ±5 days to handle this.
    """
    if sentiment_df.empty:
        return None

    deltas = abs(sentiment_df.index - earnings_date)
    nearest_idx = deltas.argmin()
    nearest_delta = deltas[nearest_idx]

    if nearest_delta > pd.Timedelta(days=max_days):
        return None

    return sentiment_df.iloc[nearest_idx]


def build_module1_features() -> pd.DataFrame:
    """
    Main function: merge sentiment features with Module 1 targets.

    Returns
    -------
    pd.DataFrame
        One row per earnings event with both sentiment features and labels.
        Rows where no transcript was found are dropped with a warning.
    """
    targets = pd.read_csv(TARGETS_CSV, parse_dates=["earnings_date", "t0"])
    print(f"Loaded {len(targets)} earnings events from module1_targets.csv")

    rows = []
    missing = 0

    for _, event in targets.iterrows():
        ticker        = event["ticker"]
        earnings_date = event["earnings_date"]

        # Load sentiment data for this ticker
        sentiment_df = load_sentiment_for_ticker(ticker)

        # Find the nearest transcript
        sentiment_row = find_nearest_transcript(earnings_date, sentiment_df)

        if sentiment_row is None:
            print(f"  ⚠️  No transcript found for {ticker} near {earnings_date.date()}")
            missing += 1
            continue

        # Build the combined row
        row = event.to_dict()
        for col in SENTIMENT_COLS:
            row[col] = sentiment_row.get(col, np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\n✓ Merged {len(df)} events ({missing} dropped, no transcript match)")
    return df


def save_features() -> pd.DataFrame:
    """Build and save the final feature matrix."""
    df = build_module1_features()

    if df.empty:
        print("⚠️  No features built: check that sentiment JSONs exist.")
        return df

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved module1_features.csv: {len(df)} rows, {len(df.columns)} columns")

    # Quick summary
    print("\nFeature columns:")
    for col in df.columns:
        null_pct = df[col].isnull().mean() * 100
        print(f"  {col:<35} {null_pct:.0f}% null")

    return df


if __name__ == "__main__":
    save_features()
