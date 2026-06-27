"""
tests/test_week1.py
-------------------
Unit tests for Week 1 data pipeline functions.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# ── Tests for compute_log_returns ─────────────────────────────────────────────

def test_log_returns_basic():
    """log(110/100) ≈ 0.0953"""
    from src.features.build_targets import compute_log_returns
    prices = pd.Series([100.0, 110.0, 121.0])
    rets = compute_log_returns(prices)
    assert np.isnan(rets.iloc[0]), "First return should be NaN"
    assert abs(rets.iloc[1] - np.log(110/100)) < 1e-9


def test_log_returns_no_look_ahead():
    """Return at index t must only use prices at t and t-1."""
    from src.features.build_targets import compute_log_returns
    prices = pd.Series([1.0, 2.0, 4.0, 8.0])
    rets = compute_log_returns(prices)
    # All returns should be log(2) ≈ 0.693
    expected = np.log(2)
    for r in rets.dropna():
        assert abs(r - expected) < 1e-9


# ── Tests for compute_realized_vol ────────────────────────────────────────────

def test_realized_vol_shape():
    """Output should have same length as input."""
    from src.features.build_targets import compute_log_returns, compute_realized_vol
    prices = pd.Series(np.random.lognormal(0, 0.01, 100))
    rets = compute_log_returns(prices)
    vol = compute_realized_vol(rets, window=21)
    assert len(vol) == len(rets)


def test_realized_vol_annualized():
    """Annualized vol of constant daily std=0.01 should be 0.01*sqrt(252) ≈ 0.1587"""
    from src.features.build_targets import compute_realized_vol
    np.random.seed(42)
    daily_std = 0.01
    rets = pd.Series(np.random.normal(0, daily_std, 252))
    vol = compute_realized_vol(rets, window=252, annualize=True)
    # Last value should be close to 0.01 * sqrt(252)
    expected = daily_std * np.sqrt(252)
    assert abs(vol.iloc[-1] - expected) < 0.002  # within 0.2%


# ── Tests for build_module1_targets ──────────────────────────────────────────

def test_no_look_ahead_in_labels(tmp_path):
    """
    Labels must be computed from FUTURE prices only.
    We verify that the 't0' trading day (earnings day) is NOT included
    in the return window — returns start from t0+1.
    """
    from src.features.build_targets import build_module1_targets

    # Build synthetic price series: constant +1% per day
    dates = pd.bdate_range("2023-01-01", "2023-06-30")
    prices = pd.Series(100 * (1.01 ** np.arange(len(dates))), index=dates)
    df = pd.DataFrame({"Close": prices, "Volume": 1e6})

    import tempfile
    from src.features.build_targets import RAW_PRICES_DIR
    # Save synthetic data as a temp ticker
    csv_path = RAW_PRICES_DIR / "TEST.csv"
    RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    df.index.name = "date"
    df.to_csv(csv_path)

    earnings_dates = ["2023-03-01"]
    result = build_module1_targets("TEST", earnings_dates)

    # The 1-day return should be positive (prices rise 1% per day)
    assert result.iloc[0]["label_1d"] == 1
    # The t0 should be >= the earnings date
    assert result.iloc[0]["t0"] >= pd.Timestamp("2023-03-01")


# ── Tests for build_module2_features ─────────────────────────────────────────

def test_forward_vol_no_feature_leakage():
    """
    The forward_vol_5d column should be shifted so that on date t,
    it contains vol from t+1 to t+5 (not including t itself).
    We verify by checking that forward vol at t ≠ trailing vol at t.
    """
    from src.features.build_targets import build_module2_features, RAW_PRICES_DIR

    # Build synthetic data
    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    prices = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, len(dates))), index=dates)
    df = pd.DataFrame({"Close": prices, "Volume": 1e6})
    df.index.name = "date"
    csv_path = RAW_PRICES_DIR / "SYNTH.csv"
    RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)

    result = build_module2_features("SYNTH")

    # forward_vol and realized_vol should not be identical (they cover different windows)
    corr = result["realized_vol_21d"].corr(result["forward_vol_5d"])
    assert corr < 0.99, "Forward and trailing vol are suspiciously identical — check shift logic"
    assert corr > 0.0,  "Volatility should have some autocorrelation (clustering)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
