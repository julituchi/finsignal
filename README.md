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

# 4. Launch dashboard (Week 4)
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
| 2 — Volatility | MAE | TBD | TBD (GARCH) |
| 2 — Volatility | RMSE | TBD | TBD (GARCH) |

Module 1 numbers come from 4-fold walk-forward CV over 28 earnings events across 5 tickers (AAPL, MSFT, NVDA, HD, MA — the set with usable Motley Fool transcripts, 2023–2024). The model beats the naive baseline, but with only 28 samples that edge isn't strong evidence of a real signal on its own — see `notebooks/02_module1_nlp.ipynb` for the full walk-through, including two data issues I found and documented rather than papered over: a prepared-remarks/Q&A splitter bug that affects most transcripts, and a SHAP ranking that points more toward a ticker-identity artifact than genuine sentiment signal. Module 2 (volatility forecasting) is Week 3 and hasn't started yet.

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
│   ├── 03_module2_volatility.ipynb    # GARCH + LSTM forecasting (Week 3, not started)
│   └── figures/                       # PNGs exported from the notebooks (gitignored)
├── models/                            # Saved model artifacts (gitignored)
├── src/
│   ├── data/                          # Price + transcript fetchers
│   ├── features/                      # Target/feature engineering
│   ├── models/                        # Model class definitions
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
GARCH(1,1) is the industry standard for volatility clustering. The LSTM is only considered an improvement if it beats this interpretable baseline, not just a naive model.

**FinBERT over generic BERT**  
FinBERT was pre-trained on financial text, so it understands domain vocabulary ('guidance', 'headwinds', 'beat') in the right context.

**SHAP for explainability**  
Banks operate under strict model governance requirements. Explainability is built in from the start so a risk officer could audit the model's drivers.

**Modest accuracy framed correctly**  
The Efficient Market Hypothesis predicts directional prediction should be hard, so I didn't expect a large edge. The current model gets 65.0% mean CV accuracy against a 60.7% "always predict up" baseline on 28 earnings events — a real but small edge that, given the sample size, is not strong enough on its own to call a validated signal. The point of Module 1 so far has been getting the methodology right (no look-ahead leakage, an honest baseline, walk-forward CV); see the notebook for why I'm not overselling the accuracy number itself.

**Separate Q&A vs prepared remarks**  
The idea: prepared remarks are scripted, Q&A is where analysts pressure-test management, so sentiment divergence between the two should carry more signal than the transcript average. In practice, building this exposed a bug — the Q&A-start marker matches the operator's opening line (which mentions "a question-and-answer session will follow") on 24 of 28 transcripts, so the "prepared remarks" section is often just a few words. That feature isn't validated yet as a result; it's documented as a known issue in `notebooks/02_module1_nlp.ipynb` rather than left silently broken.

---

## Limitations

- **Module 1's usable universe is currently 5 tickers, not 20–30**: AAPL, MSFT, NVDA, HD, MA are the only ones with reliable Motley Fool transcript coverage (2023–2024). JPM and JNJ have price data and targets but no transcripts, so they aren't part of the NLP model. A larger, more diverse ticker set is needed before ticker identity stops being a plausible confound in the SHAP results (see notebook Section 8).
- **The Q&A/prepared-remarks splitter misfires on most transcripts** (24 of 28) because its start-of-Q&A regex matches the operator's opening boilerplate instead of the real Q&A session. `qa_prep_delta` and the prepared-only features are not reliable until this is fixed.
- No transaction costs or market impact modeled
- Earnings dates for the price/target pipeline are sourced manually — a production system would use a financial data API
- Transcripts come from scraping Motley Fool's site (SEC EDGAR 8-Ks turned out not to reliably contain full transcript text — see notebook 1); some quarters 404 and are simply missing rather than backfilled

---

## Future Work

- Fix the Q&A/prepared-remarks splitter and re-validate `qa_prep_delta` on the corrected split
- Expand the NLP ticker universe well past 5 so ticker identity can't proxy for sentiment in the model or in SHAP
- Portfolio optimizer combining both signals (Markowitz or risk-parity)
- Options implied volatility surface as an additional feature
- Live data feed for real-time signal generation

---

## References

- Araci, D. (2019). [FinBERT: Financial Sentiment Analysis with BERT](https://arxiv.org/abs/1908.10063)
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*
- [SEC EDGAR Full-Text Search](https://efts.sec.gov/LATEST/search-index)
