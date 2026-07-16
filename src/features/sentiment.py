"""
src/features/sentiment.py
--------------------------
Runs FinBERT sentiment scoring on earnings call transcripts.

FinBERT (ProsusAI/finbert) outputs three probabilities per sentence:
  - positive, negative, neutral

We chunk each transcript into sentences, score each one, then aggregate
into a set of features per transcript (mean sentiment, variance, Q&A delta, etc.)

Output: a dict of features per (ticker, date), consumed by build_nlp_features.py
"""

import re
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "ProsusAI/finbert"
MAX_TOKENS     = 512          # FinBERT's context window
BATCH_SIZE     = 16           # sentences per forward pass
TRANSCRIPT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "transcripts"
SENTIMENT_DIR  = Path(__file__).resolve().parents[2] / "data" / "processed" / "sentiment"

# Labels in the order FinBERT outputs them
LABELS = ["positive", "negative", "neutral"]


def load_finbert(device: str | None = None):
    """
    Load FinBERT tokenizer and model. Downloads on first run (~440 MB).
    Automatically uses GPU if available, otherwise CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading FinBERT on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model     = model.to(device)
    model.eval()
    return tokenizer, model, device


def split_transcript(text: str) -> tuple[list[str], list[str]]:
    """
    Split transcript into prepared remarks and Q&A sections.
    Returns (prepared_sentences, qa_sentences).

    The heuristic looks for common Q&A dividers used in earnings calls.
    """
    # Common patterns that signal the start of Q&A
    qa_patterns = [
        r"question[- ]and[- ]answer",
        r"q&a session",
        r"open.{0,10}questions",
        r"operator.*please.*question",
        r"we will now begin.*question",
    ]
    qa_regex = re.compile("|".join(qa_patterns), re.IGNORECASE)

    # Find Q&A start position
    match = qa_regex.search(text)
    if match:
        prepared_text = text[:match.start()]
        qa_text       = text[match.start():]
    else:
        prepared_text = text
        qa_text       = ""

    prepared_sentences = _split_sentences(prepared_text)
    qa_sentences       = _split_sentences(qa_text)
    return prepared_sentences, qa_sentences


def _split_sentences(text: str) -> list[str]:
    """
    Naive sentence splitter: splits on '. ', '! ', '? ' and newlines.
    Filters out very short fragments (< 10 chars) that aren't real sentences.
    """
    # Split on sentence-ending punctuation or newlines
    raw = re.split(r"(?<=[.!?])\s+|\n", text)
    sentences = [s.strip() for s in raw if len(s.strip()) >= 10]
    return sentences


def score_sentences(sentences: list[str],
                     tokenizer,
                     model,
                     device: str) -> np.ndarray:
    """
    Run FinBERT on a list of sentences in batches.

    Returns
    -------
    np.ndarray of shape (n_sentences, 3)
        Columns: [positive_prob, negative_prob, neutral_prob]
    """
    if not sentences:
        return np.empty((0, 3))

    all_probs = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

        all_probs.append(probs)

    return np.vstack(all_probs)   # shape: (n_sentences, 3)


def aggregate_sentiment(probs: np.ndarray) -> dict:
    """
    Collapse per-sentence probabilities into transcript-level features.

    Features produced:
      - mean_positive, mean_negative, mean_neutral : average probabilities
      - sentiment_score  : mean_positive - mean_negative  (net sentiment, -1 to +1)
      - sentiment_std    : std dev of net sentiment across sentences (uncertainty)
      - pct_positive     : fraction of sentences where positive is the top label
      - pct_negative     : fraction of sentences where negative is the top label
      - n_sentences      : how many sentences were scored
    """
    if len(probs) == 0:
        return {k: np.nan for k in [
            "mean_positive", "mean_negative", "mean_neutral",
            "sentiment_score", "sentiment_std",
            "pct_positive", "pct_negative", "n_sentences"
        ]}

    net = probs[:, 0] - probs[:, 1]   # positive - negative per sentence
    top_labels = probs.argmax(axis=1)

    return {
        "mean_positive":  float(probs[:, 0].mean()),
        "mean_negative":  float(probs[:, 1].mean()),
        "mean_neutral":   float(probs[:, 2].mean()),
        "sentiment_score": float(net.mean()),
        "sentiment_std":  float(net.std()),
        "pct_positive":   float((top_labels == 0).mean()),
        "pct_negative":   float((top_labels == 1).mean()),
        "n_sentences":    int(len(probs)),
    }


def score_transcript(text: str, tokenizer, model, device: str) -> dict:
    """
    Full pipeline for one transcript: split → score → aggregate.
    Returns a flat dict of features including prepared vs Q&A deltas.
    """
    prepared_sents, qa_sents = split_transcript(text)

    prepared_probs = score_sentences(prepared_sents, tokenizer, model, device)
    qa_probs       = score_sentences(qa_sents,       tokenizer, model, device)
    all_probs      = np.vstack([prepared_probs, qa_probs]) if len(qa_probs) > 0 else prepared_probs

    # Aggregate each section separately
    overall  = aggregate_sentiment(all_probs)
    prepared = aggregate_sentiment(prepared_probs)
    qa       = aggregate_sentiment(qa_probs)

    # Delta features: Q&A minus prepared remarks
    # A negative delta (more negative in Q&A) often signals management under pressure
    qa_delta_score = (
        qa["sentiment_score"] - prepared["sentiment_score"]
        if not np.isnan(qa["sentiment_score"]) else np.nan
    )

    features = {**overall}
    features["prepared_sentiment_score"] = prepared["sentiment_score"]
    features["qa_sentiment_score"]       = qa["sentiment_score"]
    features["qa_delta_score"]           = qa_delta_score   # key interview feature!
    features["prepared_n_sentences"]     = prepared["n_sentences"]
    features["qa_n_sentences"]           = qa["n_sentences"]
    features["transcript_length"]        = len(prepared_sents) + len(qa_sents)

    return features


def score_all_transcripts(tickers: list[str] | None = None) -> None:
    """
    Score all transcripts for all tickers and save results as JSON files
    to data/processed/sentiment/{TICKER}/{date}.json
    """
    if tickers is None:
        tickers = [d.name for d in TRANSCRIPT_DIR.iterdir() if d.is_dir()]

    tokenizer, model, device = load_finbert()

    for ticker in tickers:
        ticker_dir  = TRANSCRIPT_DIR / ticker
        out_dir     = SENTIMENT_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        txt_files = sorted(ticker_dir.glob("*.txt"))
        if not txt_files:
            print(f"  ⚠️  No transcripts found for {ticker}")
            continue

        print(f"\n{ticker}: scoring {len(txt_files)} transcripts...")
        for txt_path in txt_files:
            out_path = out_dir / txt_path.with_suffix(".json").name
            if out_path.exists():
                print(f"  → {txt_path.stem} (cached)")
                continue

            text     = txt_path.read_text(encoding="utf-8")
            features = score_transcript(text, tokenizer, model, device)
            features["date"]   = txt_path.stem
            features["ticker"] = ticker

            out_path.write_text(json.dumps(features, indent=2))
            score = features.get("sentiment_score", float("nan"))
            delta = features.get("qa_delta_score",  float("nan"))
            print(f"  ✓ {txt_path.stem}  score={score:+.3f}  qa_delta={delta:+.3f}")

    print("\n✓ All transcripts scored.")


if __name__ == "__main__":
    score_all_transcripts()
