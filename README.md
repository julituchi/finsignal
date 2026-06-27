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
python src/data/fetch_transcripts.py
python src/features/build_targets.py

# 4. Launch dashboard (Week 4)
streamlit run app/streamlit_app.py
```

---

## Architecture

```
Earnings Call Transcripts (SEC EDGAR)        Historical Prices (yfinance)
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
| 1 — Sentiment | ROC-AUC | TBD | 0.50 (random) |
| 1 — Sentiment | Accuracy | TBD | ~52% (always-up) |
| 2 — Volatility | MAE | TBD | TBD (GARCH) |
| 2 — Volatility | RMSE | TBD | TBD (GARCH) |

*Results will be filled in after Week 3.*

---

## Project Structure

```
finsignal/
├── data/
│   ├── raw/
│   │   ├── prices/          # One CSV per ticker (gitignored)
│   │   └── transcripts/     # One folder per ticker (gitignored)
│   └── processed/           # Cleaned DataFrames (gitignored)
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_module1_nlp.ipynb
│   └── 03_module2_volatility.ipynb
├── src/
│   ├── data/                # Scrapers and loaders
│   ├── features/            # Feature engineering
│   ├── models/              # Model definitions
│   └── visualization/       # Plot helpers
├── app/
│   └── streamlit_app.py     # Interactive dashboard
├── tests/                   # pytest unit tests
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
The Efficient Market Hypothesis predicts directional prediction should be hard. A 54% accuracy on post-earnings moves is economically meaningful at scale. The goal is a signal, not a crystal ball.

**Separate Q&A vs prepared remarks**  
The prepared remarks are scripted; Q&A is where analysts pressure-test management. Sentiment divergence between the two sections often carries more signal than the transcript average.

---

## Limitations

- Limited universe: 20–30 large-cap stocks (data wrangling complexity grows fast)
- No transaction costs or market impact modeled
- Earnings dates sourced manually — a production system would use a financial data API
- EDGAR transcripts are not always cleanly formatted; some 8-K filings don't contain the full transcript

---

## Future Work

- Portfolio optimizer combining both signals (Markowitz or risk-parity)
- Options implied volatility surface as an additional feature
- Live data feed for real-time signal generation
- Expand universe to mid-cap stocks

---

## References

- Araci, D. (2019). [FinBERT: Financial Sentiment Analysis with BERT](https://arxiv.org/abs/1908.10063)
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*
- [SEC EDGAR Full-Text Search](https://efts.sec.gov/LATEST/search-index)
