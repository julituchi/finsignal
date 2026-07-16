# FinSignal

**Earnings Sentiment + Volatility Forecasting: A Two-Signal Alpha & Risk Pipeline**

> End-to-end ML project combining NLP on earnings call transcripts with time-series volatility forecasting.

---

## Quick Start

```bash
# 1. Clone and create virtual environment
git clone https://github.com/your-username/finsignal.git
cd finsignal
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Week 1 data pipeline
python src/data/fetch_prices.py
python src/features/build_targets.py
# Transcripts + Module 1 targets are fetched/built from notebooks/02_module1_nlp.ipynb
# (src/data/fetch_transcripts_motleyfool.py replaced the SEC EDGAR fetcher — see notebook Section 2)

# 4. Run Module 2 (GARCH + LSTM volatility forecasting) via notebooks/03_module2_volatility.ipynb,
# which imports its model code from src/models/volatility.py

# 5. Launch dashboard (Week 4)
streamlit run app/streamlit_app.py
```

---

## Architecture

```
Earnings Call Transcripts (Motley Fool)      Historical Prices (yfinance)
         │                                            │
         ▼                                            ▼
   FinBERT Sentiment                       Realized Volatility
   (HuggingFace)                           (21-day rolling)
         │                                            │
         ▼                                            ▼
  XGBoost Classifier                    GARCH(1,1) Baseline
  Post-Earnings Return                  + LSTM Forecaster
  Direction (↑ / ↓)                    5-day Forward Vol
         │                                            │
         └────────────────┬───────────────────────────┘
                          ▼
                  Streamlit Dashboard
            (Signal Summary per Ticker)
```

---

## Results

| Module | Metric | Model | Baseline |
|--------|--------|-------|----------|
| 1 — Sentiment | ROC-AUC | 0.667 | 0.50 (random) |
| 1 — Sentiment | Accuracy | 65.0% | 60.7% (always-up) |
| 2 — Volatility | MAE | 0.1011 (LSTM) | 0.1143 (GARCH) / 0.1086 (naive) |
| 2 — Volatility | Directional accuracy | 61.8% (LSTM) | 51.8% (GARCH) / 61.2% (naive) |

Module 1 numbers come from 4-fold walk-forward CV over 28 earnings events across 5 tickers (AAPL, MSFT, NVDA, HD, MA — the set with usable Motley Fool transcripts, 2023–2024). The model beats the naive baseline, but with only 28 samples that edge isn't strong evidence of a real signal on its own — see `notebooks/02_module1_nlp.ipynb` for the full walk-through, including two data issues I found and documented rather than papered over: a prepared-remarks/Q&A splitter bug that affects most transcripts, and a SHAP ranking that points more toward a ticker-identity artifact than genuine sentiment signal.

Module 2 numbers come from 6-fold walk-forward CV over ~3,700 test predictions across the same 5 tickers used for the price data (AAPL, MSFT, JPM, JNJ, NVDA), 2022–2024. The LSTM beats both baselines on MAE, RMSE, and directional accuracy, but the more interesting result is that **GARCH(1,1) — the "industry-standard baseline" — underperforms even the naive "today's vol = tomorrow's vol" guess**, and is worse than a coin flip on directional accuracy for 3 of 5 tickers (AAPL, JPM, MSFT). That traces to GARCH's forecast reverting toward its own long-run variance: it's the best model in the one fold covering the 2022 rate-hike selloff (a real high-vol regime where reversion was the right call) and one of the worst in every calmer fold since. See `notebooks/03_module2_volatility.ipynb` Section 8 for the full per-ticker and per-fold breakdown this is based on.

---

## Project Structure

```
finsignal/
├── data/
│   ├── raw/
│   │   ├── prices/                    # One CSV per ticker (gitignored)
│   │   └── transcripts/               # One folder per ticker (gitignored)
│   └── processed/                     # Cleaned DataFrames (gitignored)
├── notebooks/
│   ├── 01_data_collection.ipynb       # Price + transcript sourcing, target validation
│   ├── 02_module1_nlp.ipynb           # FinBERT sentiment -> XGBoost classifier, SHAP
│   ├── 03_module2_volatility.ipynb    # GARCH + LSTM walk-forward volatility forecasting
│   └── figures/                       # PNGs exported from the notebooks (gitignored)
├── models/                            # Saved model artifacts (.pt/.pkl gitignored, configs tracked)
├── src/
│   ├── data/                          # Price + transcript fetchers
│   ├── features/                      # Target/feature engineering
│   ├── models/                        # volatility.py (GARCH + LSTM, used by notebook 3)
│   └── visualization/                 # Plot helpers
├── app/
│   └── streamlit_app.py               # Interactive dashboard (Week 4, not started)
├── tests/                             # pytest unit tests
└── requirements.txt
```

---

## ML Design Decisions

**Time-series CV instead of random split**  
Random splits leak future data into training — in finance this is called look-ahead bias and it's a fatal flaw in backtesting. All models are evaluated with walk-forward validation to simulate real deployment.

**GARCH as baseline before LSTM**  
GARCH(1,1) is the industry standard for volatility clustering, so the LSTM only counts as an improvement if it beats GARCH, not just a naive "today's vol = tomorrow's vol" guess. In practice GARCH itself lost to the naive baseline on this data — see the Results section above and the notebook's Section 8 reflections for why (its forecast mean-reverts toward a long-run variance, which helps in the one fold covering a real vol spike and hurts in the calmer folds since). That's a more useful outcome than it sounds: it means the LSTM's edge is real relative to *both* baselines, not just a weak one, and it's a reminder that "industry standard" doesn't mean "wins on this specific dataset."

**FinBERT over generic BERT**  
FinBERT was pre-trained on financial text, so it understands domain vocabulary ('guidance', 'headwinds', 'beat') in the right context.

**SHAP for explainability**  
Banks operate under strict model governance requirements. Explainability is built in from the start so a risk officer could audit the model's drivers.

**Modest accuracy framed correctly**  
The Efficient Market Hypothesis predicts directional prediction should be hard, so I didn't expect a large edge. The current model gets 65.0% mean CV accuracy against a 60.7% "always predict up" baseline on 28 earnings events — a real but small edge that, given the sample size, is not strong enough on its own to call a validated signal. The point of Module 1 so far has been getting the methodology right (no look-ahead leakage, an honest baseline, walk-forward CV); see the notebook for why I'm not overselling the accuracy number itself.

**Separate Q&A vs prepared remarks**  
The idea: prepared remarks are scripted, Q&A is where analysts pressure-test management, so sentiment divergence between the two should carry more signal than the transcript average. In practice, building this exposed a bug — the Q&A-start marker matches the operator's opening line (which mentions "a question-and-answer session will follow") on 24 of 28 transcripts, so the "prepared remarks" section is often just a few words. That feature isn't validated yet as a result; it's documented as a known issue in `notebooks/02_module1_nlp.ipynb` rather than left silently broken.

**Pooled LSTM, not one model per ticker**  
Module 2's LSTM trains on all 5 tickers' sequences together rather than fitting 5 separate small models. With ~1200 rows per ticker, an early walk-forward fold would give a per-ticker model only a couple hundred usable 60-day sequences — too little for even a small LSTM. `module2_features.csv` is ticker-major, so sequence construction groups by ticker before sliding a window; a unit test (`tests/test_week3.py`) checks directly that no sequence splices one ticker's data onto another's.

**GARCH gets a rolling forecast, not one point per fold**  
Each fold refits GARCH(1,1) once on that fold's training data (frozen parameters, no peeking at test-period returns), but then produces a forecast for *every* day in the test window via `arch`'s rolling `start=` forecasting, not a single 5-day-ahead prediction per fold. Without this, GARCH's ~6 predictions per ticker wouldn't be comparable to the LSTM's hundreds — this is what makes the Results section's GARCH-vs-LSTM comparison fair rather than apples-to-oranges.

---

## Limitations

- **Module 1's usable universe is currently 5 tickers, not 20–30**: AAPL, MSFT, NVDA, HD, MA are the only ones with reliable Motley Fool transcript coverage (2023–2024). JPM and JNJ have price data and targets but no transcripts, so they aren't part of the NLP model. A larger, more diverse ticker set is needed before ticker identity stops being a plausible confound in the SHAP results (see notebook Section 8).
- **The Q&A/prepared-remarks splitter misfires on most transcripts** (24 of 28) because its start-of-Q&A regex matches the operator's opening boilerplate instead of the real Q&A session. `qa_prep_delta` and the prepared-only features are not reliable until this is fixed.
- No transaction costs or market impact modeled
- Earnings dates for the price/target pipeline are sourced manually — a production system would use a financial data API
- Transcripts come from scraping Motley Fool's site (SEC EDGAR 8-Ks turned out not to reliably contain full transcript text — see notebook 1); some quarters 404 and are simply missing rather than backfilled
- **Module 2's GARCH(1,1) is a fixed, untuned specification** — order (1,1), zero mean, normal errors. It wasn't compared against higher-order or asymmetric variants (GJR-GARCH, EGARCH) that better capture the leverage effect, so "GARCH loses to naive persistence" (see Results) is a finding about this specific setup, not a claim that GARCH can't be made competitive with more tuning.
- **Only 6 walk-forward folds exist for Module 2**, each covering about 6 months. The fold-level pattern discussed in the notebook (GARCH doing well in the 2022 selloff, poorly since) rests on one instance of a high-vol regime — not enough distinct regime changes to be confident it's not partly noise.
- Module 2's LSTM (hidden_size=32, 1 layer) wasn't hyperparameter-tuned — it's a deliberately small architecture, not a search over what's achievable on this data.

---

## Future Work

- Fix the Q&A/prepared-remarks splitter and re-validate `qa_prep_delta` on the corrected split
- Expand the NLP ticker universe well past 5 so ticker identity can't proxy for sentiment in the model or in SHAP
- Try a GJR-GARCH or EGARCH variant to see whether GARCH's directional-accuracy gap versus naive persistence is a property of this data or of the fixed (1,1) specification
- Hyperparameter-tune the LSTM (hidden size, layers, window length) rather than the current fixed architecture
- Portfolio optimizer combining both signals (Markowitz or risk-parity)
- Options implied volatility surface as an additional feature
- Live data feed for real-time signal generation

---

## References

- Araci, D. (2019). [FinBERT: Financial Sentiment Analysis with BERT](https://arxiv.org/abs/1908.10063)
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*
- Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*
- [SEC EDGAR Full-Text Search](https://efts.sec.gov/LATEST/search-index)
