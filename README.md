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
# (src/data/fetch_transcripts_motleyfool.py replaced the SEC EDGAR fetcher; see notebook Section 2)

# 4. Run Module 2 (GARCH + LSTM volatility forecasting) via notebooks/03_module2_volatility.ipynb,
# which imports its model code from src/models/volatility.py

# 5. Launch the dashboard
# The trained artifacts it needs (models/module1_xgb.pkl, models/module2_lstm.pt, and the
# processed feature/backtest CSVs) are checked into the repo, so this works right after step 2.
# Steps 3-4 are only needed if you want to regenerate them from scratch.
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
| 1: Sentiment | ROC-AUC | 0.531 | 0.50 (random ≈ 46.4% accuracy) |
| 1: Sentiment | Accuracy | 65.0% | 60.7% (always-up) |
| 2: Volatility | MAE | 0.1011 (LSTM) | 0.1143 (GARCH) / 0.1086 (naive) |
| 2: Volatility | Directional accuracy | 61.8% (LSTM) | 51.8% (GARCH) / 61.2% (naive) |

Module 1 numbers come from 4-fold walk-forward CV over 28 earnings events across 5 tickers (AAPL, MSFT, NVDA, HD, MA: the set with usable Motley Fool transcripts, 2023–2024), after fixing a prepared-remarks/Q&A splitter bug that had been computing 24 of 28 transcripts' "prepared remarks" features from a near-empty snippet (see `notebooks/02_module1_nlp.ipynb` Section 4.4). The fix left accuracy unchanged (65.0%) but dropped ROC-AUC from an inflated 0.667 to 0.531, barely above random, because one fold's previously "perfect" 1.000 ROC-AUC turned out to be the buggy feature acting as a near-lookup table rather than real skill (Section 7.1). A second issue is still open and unaffected by this fix: the top SHAP feature (`transcript_length`) looks more like a per-ticker fingerprint than a genuine content signal (Section 8).

Module 2 numbers come from 6-fold walk-forward CV over ~3,700 test predictions across the same 5 tickers used for the price data (AAPL, MSFT, JPM, JNJ, NVDA), 2022–2024. The LSTM beats both baselines on MAE, RMSE, and directional accuracy, but the more interesting result is that **GARCH(1,1), the "industry-standard baseline", underperforms even the naive "today's vol = tomorrow's vol" guess**, and is worse than a coin flip on directional accuracy for 3 of 5 tickers (AAPL, JPM, MSFT). That traces to GARCH's forecast reverting toward its own long-run variance: it's the best model in the one fold covering the 2022 rate-hike selloff (a real high-vol regime where reversion was the right call) and one of the worst in every calmer fold since. See `notebooks/03_module2_volatility.ipynb` Section 8 for the full per-ticker and per-fold breakdown this is based on.

---

## Project Structure

```
finsignal/
├── data/
│   ├── raw/                            # Scraped prices/transcripts (gitignored)
│   └── processed/                     # Cleaned DataFrames (mostly gitignored; the few
│                                       # CSVs/JSON the dashboard reads directly are tracked)
├── notebooks/
│   ├── 01_data_collection.ipynb       # Price + transcript sourcing, target validation
│   ├── 02_module1_nlp.ipynb           # FinBERT sentiment -> XGBoost classifier, SHAP
│   ├── 03_module2_volatility.ipynb    # GARCH + LSTM walk-forward volatility forecasting
│   └── figures/                       # PNGs exported from the notebooks (gitignored)
├── models/                            # Model artifacts (module1_xgb.pkl + module2_lstm.pt
│                                       # tracked since the dashboard loads them directly)
├── src/
│   ├── data/                          # Price + transcript fetchers
│   ├── features/                      # Target/feature engineering
│   └── models/                        # classifier.py, volatility.py (GARCH + LSTM)
├── app/
│   └── streamlit_app.py               # Dashboard: ticker picker, sentiment trend, vol forecast, signal card
├── tests/                             # pytest unit tests
└── requirements.txt
```

---

## ML Design Decisions

**Time-series CV instead of random split**  
Random splits leak future data into training: in finance this is called look-ahead bias and it's a fatal flaw in backtesting. All models are evaluated with walk-forward validation to simulate real deployment.

**GARCH as baseline before LSTM**  
GARCH(1,1) is the industry standard for volatility clustering, so the LSTM only counts as an improvement if it beats GARCH, not just a naive "today's vol = tomorrow's vol" guess. In practice GARCH itself lost to the naive baseline on this data. See the Results section above and the notebook's Section 8 reflections for why (its forecast mean-reverts toward a long-run variance, which helps in the one fold covering a real vol spike and hurts in the calmer folds since). That's a more useful outcome than it sounds: it means the LSTM's edge is real relative to *both* baselines, not just a weak one, and it's a reminder that "industry standard" doesn't mean "wins on this specific dataset."

**FinBERT over generic BERT**  
FinBERT was pre-trained on financial text, so it understands domain vocabulary ('guidance', 'headwinds', 'beat') in the right context.

**SHAP for explainability**  
Banks operate under strict model governance requirements. Explainability is built in from the start so a risk officer could audit the model's drivers.

**Modest accuracy framed correctly**  
The Efficient Market Hypothesis predicts directional prediction should be hard, so I didn't expect a large edge. The current model gets 65.0% mean CV accuracy against a 60.7% "always predict up" baseline on 28 earnings events: a real but small edge that, given the sample size, is not strong enough on its own to call a validated signal. ROC-AUC is more telling here: at 0.531 it's barely above the 0.50 random baseline, which is a more honest read on this model than the accuracy number alone (see below for why the two metrics disagree). The point of Module 1 so far has been getting the methodology right (no look-ahead leakage, an honest baseline, walk-forward CV); see the notebook for why I'm not overselling the accuracy number itself.

**Separate Q&A vs prepared remarks**  
The idea: prepared remarks are scripted, Q&A is where analysts pressure-test management, so sentiment divergence between the two should carry more signal than the transcript average. Building this originally exposed a bug: the Q&A-start marker matched the operator's opening line (which mentions "a question-and-answer session will follow") on 24 of 28 transcripts, so the "prepared remarks" section was often just a few words. **Fixed**: every transcript has a literal `Questions & Answers:` section header exactly where the real Q&A begins, so the splitter now anchors on that instead. The fix is a mixed result, not a clean win: accuracy was unchanged, but ROC-AUC dropped from 0.667 to 0.531 once a fold's "perfect" 1.000 ROC-AUC (traced to the buggy near-empty `prep_sentiment_mean` feature) collapsed to exactly 0.500 with real data. That's the more honest number; see `notebooks/02_module1_nlp.ipynb` Sections 4.4, 7.1, and 8 for the full walk-through, including why `qa_prep_delta` (this feature's whole point) is only now computed from trustworthy data.

**Pooled LSTM, not one model per ticker**  
Module 2's LSTM trains on all 5 tickers' sequences together rather than fitting 5 separate small models. With ~1200 rows per ticker, an early walk-forward fold would give a per-ticker model only a couple hundred usable 60-day sequences, too little for even a small LSTM. `module2_features.csv` is ticker-major, so sequence construction groups by ticker before sliding a window; a unit test (`tests/test_week3.py`) checks directly that no sequence splices one ticker's data onto another's.

**GARCH gets a rolling forecast, not one point per fold**  
Each fold refits GARCH(1,1) once on that fold's training data (frozen parameters, no peeking at test-period returns), but then produces a forecast for *every* day in the test window via `arch`'s rolling `start=` forecasting, not a single 5-day-ahead prediction per fold. Without this, GARCH's ~6 predictions per ticker wouldn't be comparable to the LSTM's hundreds. This is what makes the Results section's GARCH-vs-LSTM comparison fair rather than apples-to-oranges.

---

## Limitations

- **Module 1's usable universe is currently 5 tickers, not 20–30**: AAPL, MSFT, NVDA, HD, MA are the only ones with reliable Motley Fool transcript coverage (2023–2024). JPM and JNJ have price data and targets but no transcripts, so they aren't part of the NLP model. A larger, more diverse ticker set is needed before ticker identity stops being a plausible confound in the SHAP results (see notebook Section 8). Fixing the Q&A splitter (see below) didn't address this; `transcript_length` is still the top SHAP feature and looks like a per-ticker fingerprint rather than a content signal.
- **Module 1's ROC-AUC (0.531) is barely above random**, even after fixing the Q&A/prepared-remarks splitter bug (it previously matched the operator's boilerplate opening line instead of the real Q&A start on 24 of 28 transcripts; see notebook Section 4.4). Fixing it removed an inflated 0.667 ROC-AUC that traced to one fold's buggy `prep_sentiment_mean` feature acting as a near-lookup table (notebook Section 7.1). The corrected number is a more honest, and weaker, read on the model.
- No transaction costs or market impact modeled
- Earnings dates for the price/target pipeline are sourced manually: a production system would use a financial data API
- Transcripts come from scraping Motley Fool's site (SEC EDGAR 8-Ks turned out not to reliably contain full transcript text; see notebook 1); some quarters 404 and are simply missing rather than backfilled
- **Module 2's GARCH(1,1) is a fixed, untuned specification**: order (1,1), zero mean, normal errors. It wasn't compared against higher-order or asymmetric variants (GJR-GARCH, EGARCH) that better capture the leverage effect, so "GARCH loses to naive persistence" (see Results) is a finding about this specific setup, not a claim that GARCH can't be made competitive with more tuning.
- **Only 6 walk-forward folds exist for Module 2**, each covering about 6 months. The fold-level pattern discussed in the notebook (GARCH doing well in the 2022 selloff, poorly since) rests on one instance of a high-vol regime, not enough distinct regime changes to be confident it's not partly noise.
- Module 2's LSTM (hidden_size=32, 1 layer) wasn't hyperparameter-tuned: it's a deliberately small architecture, not a search over what's achievable on this data.

---

## Future Work

- Expand the NLP ticker universe well past 5 so ticker identity can't proxy for sentiment in the model or in SHAP (the Q&A-splitter fix didn't resolve this; `transcript_length` is still the top SHAP feature)
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
