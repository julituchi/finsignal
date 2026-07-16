"""
app/streamlit_app.py
---------------------
FinSignal dashboard: pick a ticker, see the earnings-sentiment signal
(Module 1), the volatility forecast (Module 2), and a combined summary card.

Loads the artifacts already trained in notebooks 02/03 (models/module1_xgb.pkl,
models/module2_lstm.pt, models/module2_garch_params.json) rather than retraining
anything: this is a read-only signal viewer, consistent with the project's
"no live trading / real-time feed" scope (see README Future Work).
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from arch import arch_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.volatility import garch_variance_to_vol5d, load_lstm  # noqa: E402

DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="FinSignal", page_icon="📈", layout="wide")


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data
def load_module1():
    df = pd.read_csv(DATA_DIR / "module1_features.csv", parse_dates=["earnings_date"])
    feature_cols = json.loads((MODELS_DIR / "module1_features.json").read_text())
    return df, feature_cols


@st.cache_resource
def load_module1_model():
    with open(MODELS_DIR / "module1_xgb.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_module2():
    feat = pd.read_csv(DATA_DIR / "module2_features.csv", parse_dates=["date"])
    backtest = pd.read_csv(DATA_DIR / "module2_walk_forward_predictions.csv", parse_dates=["date"])
    summary = json.loads((DATA_DIR / "module2_results_summary.json").read_text())
    return feat, backtest, summary


@st.cache_resource
def load_module2_models():
    lstm_model, lstm_config = load_lstm()
    return lstm_model, lstm_config


# ── Live inference on the most recent window ──────────────────────────────────

def current_lstm_forecast(feat: pd.DataFrame, ticker: str, model, config: dict):
    g = feat[feat["ticker"] == ticker].sort_values("date")
    window = config["window"]
    if len(g) < window:
        return None
    X = g[config["feature_cols"]].to_numpy(dtype=float)[-window:]
    mean, std = np.array(config["scaler_mean"]), np.array(config["scaler_std"])
    X_scaled = (X - mean) / std
    with torch.no_grad():
        pred = model(torch.tensor(X_scaled[None, :, :], dtype=torch.float32)).item()
    return pred


def current_garch_forecast(feat: pd.DataFrame, ticker: str, horizon: int = 5):
    g = feat[feat["ticker"] == ticker].sort_values("date")
    returns = pd.Series(g["log_return"].to_numpy() * 100.0, index=pd.DatetimeIndex(g["date"]))
    am = arch_model(returns, mean="Zero", vol="Garch", p=1, q=1, dist="normal", rescale=False)
    res = am.fit(disp="off")
    fc = res.forecast(horizon=horizon, reindex=False)
    return garch_variance_to_vol5d(fc.variance.iloc[-1].to_numpy())


# ── Data / model loading ───────────────────────────────────────────────────────

try:
    df1, feature_cols1 = load_module1()
    model1 = load_module1_model()
    feat2, backtest2, summary2 = load_module2()
    lstm_model, lstm_config = load_module2_models()
except FileNotFoundError as e:
    st.error(
        f"Missing artifact: `{e.filename}`. Run the notebooks (02_module1_nlp.ipynb, "
        "03_module2_volatility.ipynb) to generate model/data artifacts before launching the dashboard."
    )
    st.stop()

df1["pred_prob_up"] = model1.predict_proba(df1[feature_cols1].fillna(0))[:, 1]
df1["pred_label"] = (df1["pred_prob_up"] >= 0.5).astype(int)

module1_tickers = sorted(df1["ticker"].unique())
module2_tickers = sorted(feat2["ticker"].unique())
all_tickers = sorted(set(module1_tickers) | set(module2_tickers))


# ── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("FinSignal")
ticker = st.sidebar.selectbox("Ticker", all_tickers)
st.sidebar.markdown("---")
st.sidebar.caption(f"**Sentiment coverage** (Module 1): {', '.join(module1_tickers)}")
st.sidebar.caption(f"**Volatility coverage** (Module 2): {', '.join(module2_tickers)}")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Backtested signal viewer, not a live feed; see README Limitations & Future Work."
)

st.title(f"{ticker}: Signal Dashboard")


# ── Sentiment panel ───────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    st.subheader("Sentiment Signal (Module 1)")
    if ticker in module1_tickers:
        g1 = df1[df1["ticker"] == ticker].sort_values("earnings_date")

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=g1["earnings_date"], y=g1["sentiment_mean"], name="Mean sentiment",
            marker_color=["#2ca02c" if v > 0 else "#d62728" for v in g1["sentiment_mean"]],
        ))
        fig1.update_layout(
            yaxis_title="Mean FinBERT sentiment", xaxis_title="Earnings date",
            height=350, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig1, width="stretch")

        table = g1[["earnings_date", "sentiment_mean", "pred_prob_up", "label_1d"]].rename(columns={
            "earnings_date": "Earnings date", "sentiment_mean": "Sentiment",
            "pred_prob_up": "Model P(up)", "label_1d": "Actual 1d move",
        }).set_index("Earnings date")
        table["Actual 1d move"] = table["Actual 1d move"].map({1: "up", 0: "down"})
        st.dataframe(table.sort_index(ascending=False), width="stretch")
    else:
        st.info(
            f"{ticker} has no earnings-call transcript coverage in Module 1 "
            f"(only {', '.join(module1_tickers)}; see README Limitations)."
        )


# ── Volatility panel ───────────────────────────────────────────────────────────

with col2:
    st.subheader("Volatility Forecast (Module 2)")
    if ticker in module2_tickers:
        g2 = backtest2[backtest2["ticker"] == ticker].sort_values("date").tail(120)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=g2["date"], y=g2["actual"], name="Actual", line=dict(color="black")))
        fig2.add_trace(go.Scatter(x=g2["date"], y=g2["lstm_pred"], name="LSTM"))
        fig2.add_trace(go.Scatter(x=g2["date"], y=g2["garch_pred"], name="GARCH"))
        fig2.update_layout(
            yaxis_title="Annualized 5-day forward vol", xaxis_title="Date (last 120 backtest days)",
            height=350, margin=dict(t=20, b=20), legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig2, width="stretch")
        st.caption("Walk-forward backtest predictions (2022-2024); see notebook 03, Section 8.")
    else:
        st.info(
            f"{ticker} has no price-volatility feature coverage in Module 2 "
            f"(only {', '.join(module2_tickers)}; see README Limitations)."
        )


# ── Signal summary card ────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Signal Summary")
c1, c2, c3 = st.columns(3)

with c1:
    if ticker in module1_tickers:
        latest = g1.iloc[-1]
        direction = "Bullish" if latest["pred_prob_up"] >= 0.5 else "Bearish"
        st.metric(
            "Latest sentiment signal", direction,
            f"P(up) = {latest['pred_prob_up']:.0%} · {pd.Timestamp(latest['earnings_date']).date()}",
        )
    else:
        st.metric("Latest sentiment signal", "n/a")

with c2:
    if ticker in module2_tickers:
        lstm_fc = current_lstm_forecast(feat2, ticker, lstm_model, lstm_config)
        realized = feat2[feat2["ticker"] == ticker].sort_values("date")["realized_vol_21d"].iloc[-1]
        vol_direction = "Expanding" if lstm_fc > realized else "Contracting"
        st.metric(
            "Current 5-day vol forecast (LSTM)", f"{lstm_fc:.1%}",
            f"{vol_direction} vs {realized:.1%} trailing 21d",
        )
    else:
        st.metric("Current 5-day vol forecast (LSTM)", "n/a")

with c3:
    if ticker in module2_tickers:
        garch_fc = current_garch_forecast(feat2, ticker)
        st.metric(
            "Current 5-day vol forecast (GARCH)", f"{garch_fc:.1%}",
            "Industry baseline; see README on why it underperforms here",
        )
    else:
        st.metric("Current 5-day vol forecast (GARCH)", "n/a")

st.caption(
    f"Module 2 pooled backtest: LSTM MAE {summary2['pooled']['lstm']['mae']:.3f} vs "
    f"GARCH {summary2['pooled']['garch']['mae']:.3f} vs naive {summary2['pooled']['naive']['mae']:.3f} "
    f"({summary2['pooled']['lstm']['n']} predictions, {summary2['n_folds']} walk-forward folds)."
)
