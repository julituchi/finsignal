"""
tests/test_week3.py
-------------------
Unit tests for Week 3 volatility-forecasting functions.
Run with: pytest tests/test_week3.py -v
"""

import numpy as np
import pandas as pd
import pytest


# ── Tests for make_walk_forward_folds ─────────────────────────────────────────

def test_folds_are_contiguous_and_expanding():
    """Each fold's test window should immediately follow the previous one,
    and train always means 'everything before this fold's test_start'."""
    from src.models.volatility import make_walk_forward_folds

    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    folds = make_walk_forward_folds(dates, n_folds=6, burn_in_frac=0.4)

    assert len(folds) == 6
    for f in folds:
        assert f["train_end"] == f["test_start"]
        assert f["test_start"] <= f["test_end"]

    # Fold i+1's test_start should be the day right after fold i's test_end
    # in the underlying date index (contiguous, no gaps or overlap).
    uniq = np.array(sorted(pd.unique(np.asarray(dates))))
    for a, b in zip(folds, folds[1:]):
        end_idx = np.searchsorted(uniq, a["test_end"])
        start_idx = np.searchsorted(uniq, b["test_start"])
        assert start_idx == end_idx + 1


def test_folds_burn_in_reserved():
    """No fold should test on dates before the burn-in period ends."""
    from src.models.volatility import make_walk_forward_folds

    dates = pd.bdate_range("2020-01-01", "2024-12-31")
    folds = make_walk_forward_folds(dates, n_folds=6, burn_in_frac=0.4)
    uniq = np.array(sorted(pd.unique(np.asarray(dates))))
    expected_burn_in_end = uniq[int(len(uniq) * 0.4)]
    assert folds[0]["test_start"] == expected_burn_in_end


# ── Tests for garch_variance_to_vol5d ─────────────────────────────────────────

def test_garch_variance_to_vol5d_formula():
    """Given known scaled daily variances, output should match
    sqrt(mean(variance)/100**2) * sqrt(252) exactly."""
    from src.models.volatility import garch_variance_to_vol5d

    # Constant daily variance of 4.0 in (return*100)^2 units
    # -> daily variance in raw-return units = 4.0 / 100**2 = 0.0004
    # -> daily vol = 0.02, annualized = 0.02 * sqrt(252)
    variances = [4.0, 4.0, 4.0, 4.0, 4.0]
    result = garch_variance_to_vol5d(variances)
    expected = np.sqrt(0.0004) * np.sqrt(252)
    assert abs(result - expected) < 1e-9


def test_garch_variance_to_vol5d_varying_input():
    """Non-constant variances should use the mean, not e.g. the last value."""
    from src.models.volatility import garch_variance_to_vol5d

    variances = [1.0, 2.0, 3.0, 4.0, 5.0]  # mean = 3.0
    result = garch_variance_to_vol5d(variances)
    expected = np.sqrt(3.0 / 100**2) * np.sqrt(252)
    assert abs(result - expected) < 1e-9


# ── Tests for build_lstm_sequences ────────────────────────────────────────────

def test_lstm_sequences_never_cross_ticker_boundary():
    """
    module2_features.csv is ticker-major (all of ticker A's rows, then all
    of ticker B's rows). If build_lstm_sequences rolled a window across the
    raw concatenated rows instead of grouping by ticker first, a window
    near the boundary would splice A's tail onto B's head. Build a synthetic
    frame with a deliberate value jump at the boundary and assert no
    sequence contains both.
    """
    from src.models.volatility import build_lstm_sequences

    window = 10
    dates_a = pd.bdate_range("2023-01-01", periods=15)
    dates_b = pd.bdate_range("2023-01-01", periods=15)  # tickers can share dates

    df_a = pd.DataFrame({
        "ticker": "A", "date": dates_a,
        "log_return": 0.0, "volume_norm": 1.0,
        "realized_vol_21d": 100.0,          # deliberately huge, distinct value
        "forward_vol_5d": 0.5,
    })
    df_b = pd.DataFrame({
        "ticker": "B", "date": dates_b,
        "log_return": 0.0, "volume_norm": 1.0,
        "realized_vol_21d": 0.2,            # normal-range value
        "forward_vol_5d": 0.5,
    })
    df = pd.concat([df_a, df_b], ignore_index=True)  # ticker-major, like the real file

    X, y, seq_dates, seq_tickers = build_lstm_sequences(
        df, feature_cols=["log_return", "volume_norm", "realized_vol_21d"],
        target_col="forward_vol_5d", window=window,
    )

    assert len(X) > 0
    realized_vol_col_idx = 2
    for i in range(len(X)):
        window_values = X[i][:, realized_vol_col_idx]
        if seq_tickers[i] == "A":
            assert np.all(window_values == 100.0), "Ticker A sequence contaminated with ticker B data"
        else:
            assert np.all(window_values == 0.2), "Ticker B sequence contaminated with ticker A data"


def test_lstm_sequence_window_length_and_alignment():
    """Each sequence should be exactly `window` rows, ending at the target's own date."""
    from src.models.volatility import build_lstm_sequences

    window = 5
    dates = pd.bdate_range("2023-01-01", periods=8)
    df = pd.DataFrame({
        "ticker": "A", "date": dates,
        "log_return": np.arange(8) * 0.01,
        "volume_norm": 1.0,
        "realized_vol_21d": np.arange(8) * 0.1,
        "forward_vol_5d": np.arange(8) * 0.2,
    })

    X, y, seq_dates, seq_tickers = build_lstm_sequences(
        df, feature_cols=["log_return"], target_col="forward_vol_5d", window=window,
    )

    # 8 rows, window=5 -> sequences ending at indices 4,5,6,7 -> 4 sequences
    assert X.shape == (4, window, 1)
    # First sequence ends at index 4 (0-based): log_return values 0..4 * 0.01
    np.testing.assert_allclose(X[0][:, 0], np.arange(0, 5) * 0.01)
    assert y[0] == pytest.approx(4 * 0.2)
    assert seq_dates[0] == dates[4]


# ── Tests for VolatilityLSTM ─────────────────────────────────────────────────

def test_lstm_output_is_non_negative():
    """Softplus output head should guarantee non-negative predictions, even
    on random/untrained weights and inputs with negative values."""
    from src.models.volatility import VolatilityLSTM
    import torch

    torch.manual_seed(0)
    model = VolatilityLSTM(input_size=3, hidden_size=8, num_layers=1)
    x = torch.randn(16, 60, 3) * 10  # deliberately large, includes negatives
    with torch.no_grad():
        out = model(x)
    assert out.shape == (16,)
    assert torch.all(out >= 0)


def test_lstm_output_shape_matches_batch():
    from src.models.volatility import VolatilityLSTM
    import torch

    model = VolatilityLSTM(input_size=3, hidden_size=8, num_layers=2)
    x = torch.randn(5, 60, 3)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (5,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
