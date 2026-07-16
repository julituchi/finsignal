"""
src/models/volatility.py
--------------------------
GARCH(1,1) baseline + LSTM forecaster for Module 2: predicting 5-day forward
realized volatility (forward_vol_5d) from data/processed/module2_features.csv.

Design decisions baked in:
  1. Walk-forward validation -- expanding window, date-based, the same
     train-cutoff date applies to every ticker in a given fold, so there's
     no cross-sectional leakage between tickers.
  2. GARCH gets a rolling per-day forecast within each fold (via arch's
     `start=` forecasting), not a single 5-day-ahead point per fold -- that
     makes it directly comparable to the LSTM's per-day predictions.
  3. module2_features.csv is ticker-major (all of one ticker's rows, then
     the next). LSTM sequences are built per-ticker via groupby before
     sliding a window, or one ticker's tail would splice onto the next
     ticker's head.
  4. The LSTM trains on log1p-transformed vol (right-skewed, COVID-era
     outliers up to ~1.7 annualized) and outputs through a Softplus head,
     so predictions are guaranteed non-negative by construction.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from arch import arch_model

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
FEATURES_CSV = PROCESSED_DIR / "module2_features.csv"

FEATURE_COLS = ["log_return", "volume_norm", "realized_vol_21d"]
TARGET_COL = "forward_vol_5d"
WINDOW = 60


# ── Data loading ────────────────────────────────────────────────────────────

def load_features() -> pd.DataFrame:
    """Load module2_features.csv, sorted by ticker then date (ticker-major)."""
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    df.sort_values(["ticker", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ── Walk-forward folds ───────────────────────────────────────────────────────

def make_walk_forward_folds(dates, n_folds: int = 6, burn_in_frac: float = 0.4) -> list[dict]:
    """
    Expanding-window, date-based folds shared across all tickers.

    `dates` is any array of (possibly repeated) trading dates -- unique
    values are extracted and sorted. The first `burn_in_frac` of dates are
    reserved as pure training history; the rest is split into `n_folds`
    contiguous test blocks. For fold i: train = every date < test_start,
    test = [test_start, test_end] inclusive.
    """
    uniq = np.array(sorted(pd.unique(np.asarray(dates))))
    n = len(uniq)
    burn_in_idx = int(n * burn_in_frac)
    remaining = n - burn_in_idx
    fold_size = remaining // n_folds

    folds = []
    for i in range(n_folds):
        test_start_idx = burn_in_idx + i * fold_size
        test_end_idx = n if i == n_folds - 1 else test_start_idx + fold_size
        if test_start_idx >= n:
            break
        folds.append({
            "fold": i + 1,
            "train_end": uniq[test_start_idx],       # train = dates < train_end
            "test_start": uniq[test_start_idx],
            "test_end": uniq[test_end_idx - 1],
        })
    return folds


# ── LSTM sequence construction ───────────────────────────────────────────────

def build_lstm_sequences(df: pd.DataFrame, feature_cols=FEATURE_COLS,
                          target_col: str = TARGET_COL, window: int = WINDOW):
    """
    Build (X, y, dates, tickers) sequences, one per (ticker, date), never
    crossing a ticker boundary -- this is the fix for module2_features.csv
    being ticker-major rather than date-major.

    Sequence for date i uses the `window` days ending at and including day i
    (i.e. today's own realized_vol_21d/log_return/volume_norm are the last
    row of the input window). The label at day i, forward_vol_5d, is defined
    purely from days i+1..i+5, so including day i's own trailing features as
    input is causally safe.
    """
    X_list, y_list, date_list, ticker_list = [], [], [], []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype=float)
        target = g[target_col].to_numpy(dtype=float)
        dates = g["date"].to_numpy()
        for i in range(window - 1, len(g)):
            X_list.append(feats[i - window + 1: i + 1])
            y_list.append(target[i])
            date_list.append(dates[i])
            ticker_list.append(ticker)

    X = np.stack(X_list) if X_list else np.empty((0, window, len(feature_cols)))
    y = np.array(y_list, dtype=float)
    dates = np.array(date_list)
    tickers = np.array(ticker_list)
    return X, y, dates, tickers


def train_val_split_by_date(dates: np.ndarray, val_frac: float = 0.12):
    """
    Chronological train/val split for early stopping: the last `val_frac`
    of dates (not rows -- dates, since multiple tickers share a date) go to
    validation. Returns (train_mask, val_mask) booleans aligned to `dates`.
    """
    uniq_sorted = np.sort(np.unique(dates))
    cutoff_idx = max(1, int(len(uniq_sorted) * (1 - val_frac)))
    cutoff_date = uniq_sorted[cutoff_idx]
    val_mask = dates >= cutoff_date
    return ~val_mask, val_mask


def fit_scaler(X_train: np.ndarray):
    """Per-feature mean/std over the training window only."""
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


# ── LSTM model ────────────────────────────────────────────────────────────────

class VolatilityLSTM(nn.Module):
    """
    A small LSTM predicting 5-day forward realized vol from a 60-day window
    of (log_return, volume_norm, realized_vol_21d). The Softplus output head
    guarantees predictions are non-negative -- volatility can't be negative.
    """

    def __init__(self, input_size: int = len(FEATURE_COLS), hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # out: (batch, window, hidden_size)
        last = out[:, -1, :]           # final time step's hidden state
        return self.head(last).squeeze(-1)


def train_lstm(X_train, y_train, X_val, y_val, hidden_size: int = 32,
               num_layers: int = 1, epochs: int = 60, lr: float = 1e-3,
               patience: int = 8, batch_size: int = 64, seed: int = 42,
               verbose: bool = False) -> VolatilityLSTM:
    """
    Train on log1p-transformed vol (loss computed in log-space on the
    model's already-non-negative Softplus output), with early stopping on a
    chronological validation slice.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = VolatilityLSTM(input_size=X_train.shape[-1], hidden_size=hidden_size,
                            num_layers=num_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    n = len(X_train_t)
    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(torch.log1p(pred), torch.log1p(yb))
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = nn.functional.mse_loss(torch.log1p(val_pred), torch.log1p(y_val_t)).item()

        if verbose:
            print(f"  epoch {epoch + 1:3d}  val_loss={val_loss:.5f}")

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


# ── GARCH(1,1) baseline ──────────────────────────────────────────────────────

def garch_variance_to_vol5d(daily_variances) -> float:
    """
    Convert horizon-1..5 forecasted *daily* variances (in units of
    (return*100)^2, since returns are scaled by 100 before fitting) into an
    annualized 5-day-forward-vol estimate directly comparable to
    forward_vol_5d = std(next 5 daily log-returns) * sqrt(252).
    """
    variances = np.asarray(daily_variances, dtype=float)
    mean_var = variances.mean() / (100.0 ** 2)
    return float(np.sqrt(mean_var) * np.sqrt(252))


def fit_garch_and_forecast(returns: pd.Series, train_end, test_start, test_end,
                            horizon: int = 5) -> pd.Series:
    """
    Fit GARCH(1,1) on `returns` up to (not including) `train_end`, then
    produce a rolling forecast of vol5d for every date in
    [test_start, test_end].

    `returns` must be indexed by date and cover the training AND test
    period -- arch needs the full series to roll the variance recursion
    forward using each day's *realized* return as it becomes available.
    This is not leakage: GARCH parameters (omega, alpha, beta) are frozen
    from the `last_obs=train_end` fit, and each origin day's h=1..5
    forecast only ever uses the model's own recursive expectation, never
    an actual future return.
    """
    scaled = returns * 100.0
    am = arch_model(scaled, mean="Zero", vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(last_obs=train_end, disp="off")
    fc = res.forecast(horizon=horizon, start=test_start, reindex=False)
    variances = fc.variance
    variances = variances.loc[(variances.index >= pd.Timestamp(test_start)) &
                               (variances.index <= pd.Timestamp(test_end))]
    vol5d = variances.apply(lambda row: garch_variance_to_vol5d(row.to_numpy()), axis=1)
    return vol5d


def fit_final_garch_params(df: pd.DataFrame, tickers=None) -> dict:
    """Fit GARCH(1,1) on each ticker's full return history; return fitted params."""
    tickers = tickers or sorted(df["ticker"].unique())
    params = {}
    for ticker in tickers:
        g = df[df["ticker"] == ticker].sort_values("date")
        returns = pd.Series(g["log_return"].to_numpy() * 100.0,
                             index=pd.DatetimeIndex(g["date"].to_numpy()))
        am = arch_model(returns, mean="Zero", vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res = am.fit(disp="off")
        params[ticker] = {
            "omega": float(res.params["omega"]),
            "alpha[1]": float(res.params["alpha[1]"]),
            "beta[1]": float(res.params["beta[1]"]),
        }
    return params


# ── Naive baseline ───────────────────────────────────────────────────────────

def naive_persistence_baseline(realized_vol_21d) -> np.ndarray:
    """Predict forward_vol_5d as simply the current trailing realized_vol_21d."""
    return np.asarray(realized_vol_21d, dtype=float)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred, realized_vol_21d) -> dict:
    """
    MAE, RMSE, and directional accuracy -- did vol expand or contract
    relative to the trailing realized_vol_21d, and did the model call that
    direction correctly.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    realized_vol_21d = np.asarray(realized_vol_21d, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    true_dir = (y_true > realized_vol_21d).astype(int)
    pred_dir = (y_pred > realized_vol_21d).astype(int)
    dir_acc = float(np.mean(true_dir == pred_dir))
    return {"mae": mae, "rmse": rmse, "directional_accuracy": dir_acc, "n": int(len(y_true))}


# ── Walk-forward orchestration ───────────────────────────────────────────────

def run_walk_forward(df: pd.DataFrame, feature_cols=FEATURE_COLS, n_folds: int = 6,
                      burn_in_frac: float = 0.4, window: int = WINDOW,
                      hidden_size: int = 32, num_layers: int = 1, epochs: int = 60,
                      patience: int = 8, seed: int = 42, verbose: bool = True) -> pd.DataFrame:
    """
    Run naive / GARCH / LSTM across all walk-forward folds. Returns one row
    per (fold, ticker, date) with actual, realized_vol_21d, and each model's
    prediction -- ready for both metric aggregation and plotting.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    folds = make_walk_forward_folds(df["date"].to_numpy(), n_folds=n_folds, burn_in_frac=burn_in_frac)
    tickers = sorted(df["ticker"].unique())

    all_preds = []

    for f in folds:
        fold_id, train_end, test_start, test_end = f["fold"], f["train_end"], f["test_start"], f["test_end"]
        if verbose:
            print(f"Fold {fold_id}: train < {pd.Timestamp(train_end).date()}  "
                  f"test [{pd.Timestamp(test_start).date()} .. {pd.Timestamp(test_end).date()}]")

        # ---- LSTM ----
        seq_df = df[df["date"] <= test_end]
        X, y, seq_dates, seq_tickers = build_lstm_sequences(seq_df, feature_cols=feature_cols, window=window)
        train_seq_mask = seq_dates < np.datetime64(train_end)
        test_seq_mask = (seq_dates >= np.datetime64(test_start)) & (seq_dates <= np.datetime64(test_end))

        X_tr_all, y_tr_all, dates_tr_all = X[train_seq_mask], y[train_seq_mask], seq_dates[train_seq_mask]
        tr_keep, val_keep = train_val_split_by_date(dates_tr_all)

        mean, std = fit_scaler(X_tr_all[tr_keep])
        X_tr = apply_scaler(X_tr_all[tr_keep], mean, std)
        X_val = apply_scaler(X_tr_all[val_keep], mean, std)
        y_tr, y_val = y_tr_all[tr_keep], y_tr_all[val_keep]

        model = train_lstm(X_tr, y_tr, X_val, y_val, hidden_size=hidden_size,
                            num_layers=num_layers, epochs=epochs, patience=patience, seed=seed)

        X_test = apply_scaler(X[test_seq_mask], mean, std)
        with torch.no_grad():
            lstm_pred = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

        lstm_df = pd.DataFrame({
            "ticker": seq_tickers[test_seq_mask],
            "date": seq_dates[test_seq_mask],
            "lstm_pred": lstm_pred,
        })

        # ---- GARCH (per ticker) ----
        garch_frames = []
        for ticker in tickers:
            g = df[df["ticker"] == ticker].sort_values("date")
            returns = pd.Series(g["log_return"].to_numpy(), index=pd.DatetimeIndex(g["date"].to_numpy()))
            vol5d = fit_garch_and_forecast(returns, train_end, test_start, test_end)
            garch_frames.append(pd.DataFrame({
                "ticker": ticker, "date": vol5d.index.values, "garch_pred": vol5d.to_numpy(),
            }))
        garch_df = pd.concat(garch_frames, ignore_index=True)

        # ---- Actuals + naive baseline ----
        actual_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)][
            ["ticker", "date", "forward_vol_5d", "realized_vol_21d"]
        ].rename(columns={"forward_vol_5d": "actual"})
        actual_df["naive_pred"] = naive_persistence_baseline(actual_df["realized_vol_21d"])

        merged = (actual_df
                  .merge(lstm_df, on=["ticker", "date"], how="inner")
                  .merge(garch_df, on=["ticker", "date"], how="inner"))
        merged["fold"] = fold_id
        all_preds.append(merged)

    return pd.concat(all_preds, ignore_index=True)


def summarize_results(preds_df: pd.DataFrame):
    """Pooled and per-ticker MAE/RMSE/directional-accuracy for each model."""
    model_cols = [("naive_pred", "naive"), ("garch_pred", "garch"), ("lstm_pred", "lstm")]

    pooled_rows = []
    for col, name in model_cols:
        m = evaluate(preds_df["actual"], preds_df[col], preds_df["realized_vol_21d"])
        m["model"] = name
        pooled_rows.append(m)
    pooled = pd.DataFrame(pooled_rows)[["model", "mae", "rmse", "directional_accuracy", "n"]]

    per_ticker_rows = []
    for ticker, g in preds_df.groupby("ticker"):
        for col, name in model_cols:
            m = evaluate(g["actual"], g[col], g["realized_vol_21d"])
            m["model"] = name
            m["ticker"] = ticker
            per_ticker_rows.append(m)
    per_ticker = pd.DataFrame(per_ticker_rows)[["ticker", "model", "mae", "rmse", "directional_accuracy", "n"]]

    return pooled, per_ticker


# ── Final artifact training + save/load ──────────────────────────────────────

def train_final_lstm(df: pd.DataFrame, feature_cols=FEATURE_COLS, window: int = WINDOW,
                      hidden_size: int = 32, num_layers: int = 1, epochs: int = 60,
                      patience: int = 8, seed: int = 42):
    """Refit the LSTM on the full dataset for the deployment artifact (not a fold model)."""
    X, y, dates, _ = build_lstm_sequences(df, feature_cols=feature_cols, window=window)
    tr_keep, val_keep = train_val_split_by_date(dates, val_frac=0.1)
    mean, std = fit_scaler(X[tr_keep])
    X_tr, X_val = apply_scaler(X[tr_keep], mean, std), apply_scaler(X[val_keep], mean, std)
    y_tr, y_val = y[tr_keep], y[val_keep]
    model = train_lstm(X_tr, y_tr, X_val, y_val, hidden_size=hidden_size, num_layers=num_layers,
                        epochs=epochs, patience=patience, seed=seed)
    return model, mean, std


def save_lstm(model: VolatilityLSTM, mean, std, feature_cols=FEATURE_COLS, window: int = WINDOW):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = MODEL_DIR / "module2_lstm.pt"
    config_path = MODEL_DIR / "module2_lstm_config.json"
    torch.save(model.state_dict(), weights_path)
    config = {
        "feature_cols": feature_cols,
        "window": window,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "scaler_mean": np.asarray(mean).tolist(),
        "scaler_std": np.asarray(std).tolist(),
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return weights_path, config_path


def load_lstm(weights_path=None, config_path=None):
    weights_path = weights_path or (MODEL_DIR / "module2_lstm.pt")
    config_path = config_path or (MODEL_DIR / "module2_lstm_config.json")
    with open(config_path) as f:
        config = json.load(f)
    model = VolatilityLSTM(
        input_size=len(config["feature_cols"]),
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model, config


def save_garch_params(params: dict, path=None):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = path or (MODEL_DIR / "module2_garch_params.json")
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path


def load_garch_params(path=None) -> dict:
    path = path or (MODEL_DIR / "module2_garch_params.json")
    with open(path) as f:
        return json.load(f)
