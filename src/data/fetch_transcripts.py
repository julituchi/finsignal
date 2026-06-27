"""
src/data/fetch_transcripts.py
------------------------------
Fetches earnings call transcripts from SEC EDGAR 8-K filings.

Strategy:
  1. Query EDGAR full-text search for 8-K filings mentioning "earnings call"
     or "conference call transcript" for each ticker + CIK.
  2. Parse the filing index to find the actual .htm/.txt document.
  3. Extract and clean the transcript text.
  4. Save as data/raw/transcripts/{TICKER}/{date}.txt

SEC Fair-Use Note:
  EDGAR allows automated access at ≤10 requests/second.
  We add a 0.15 s delay between requests to stay well under the limit.
  Always include a descriptive User-Agent header (required by SEC).
"""

import time
import re
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR   = Path(__file__).resolve().parents[2] / "data" / "raw" / "transcripts"
EDGAR_URL = "https://efts.sec.gov/LATEST/search-index?q=%22conference+call%22+%22earnings%22&dateRange=custom&startdt={start}&enddt={end}&forms=8-K&hits.hits._source.period_of_report=true"

# SEC requires a descriptive User-Agent: "Name Email"
HEADERS = {
    "User-Agent": "FinSignal Research Project finSignal@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "efts.sec.gov",
}

# CIK numbers for our universe (static — CIKs don't change)
TICKER_TO_CIK = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
    "AMZN":  "0001018724",
    "META":  "0001326801",
    "JPM":   "0000019617",
    "GS":    "0000886982",
    "BAC":   "0000070858",
    "JNJ":   "0000200406",
    "PFE":   "0000078003",
    "UNH":   "0000731766",
    "XOM":   "0000034088",
    "CVX":   "0000093410",
    "WMT":   "0000104169",
    "HD":    "0000354950",
    "NKE":   "0000320187",
    "TSLA":  "0001318605",
    "NVDA":  "0001045810",
    "V":     "0001403161",
    "MA":    "0001141391",
}


def get_8k_filings(cik: str, start: str = "2020-01-01", end: str = "2024-12-31") -> list[dict]:
    """
    Query EDGAR for 8-K filings for a given CIK in the date range.
    Returns a list of filing metadata dicts (accession number, date, documents URL).
    """
    url = (
        f"https://data.sec.gov/submissions/CIK{cik}.json"
    )
    headers = {**HEADERS, "Host": "data.sec.gov"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    forms       = recent.get("form", [])
    dates       = recent.get("filingDate", [])
    accessions  = recent.get("accessionNumber", [])

    for form, date, acc in zip(forms, dates, accessions):
        if form != "8-K":
            continue
        if not (start <= date <= end):
            continue
        filings.append({
            "accession": acc.replace("-", ""),
            "date": date,
            "cik": cik,
        })

    return filings


def get_transcript_text(cik: str, accession: str) -> str | None:
    """
    Given a CIK and accession number, fetch the filing index and extract
    the largest .htm document (usually the 8-K body with the transcript exhibit).
    Returns cleaned plain text, or None if no usable document found.
    """
    index_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/index.json"
    headers = {**HEADERS, "Host": "www.sec.gov"}

    try:
        resp = requests.get(index_url, headers=headers, timeout=15)
        resp.raise_for_status()
        index = resp.json()
    except Exception as e:
        print(f"    Could not fetch index for {accession}: {e}")
        return None

    # Find the largest .htm file — it's usually the transcript exhibit (EX-99)
    docs = index.get("directory", {}).get("item", [])
    htm_docs = [d for d in docs if d.get("name", "").lower().endswith(".htm")]
    if not htm_docs:
        return None

    # Sort by file size descending; the transcript is usually the biggest doc
    htm_docs.sort(key=lambda d: int(d.get("size", 0)), reverse=True)
    doc_name = htm_docs[0]["name"]
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc_name}"

    try:
        doc_resp = requests.get(doc_url, headers=headers, timeout=20)
        doc_resp.raise_for_status()
    except Exception as e:
        print(f"    Could not fetch document {doc_name}: {e}")
        return None

    soup = BeautifulSoup(doc_resp.content, "lxml")

    # Remove script/style noise
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = _clean_text(text)

    # Quick heuristic: real transcripts mention speakers / Q&A
    is_transcript = any(kw in text.lower() for kw in [
        "operator", "question-and-answer", "q&a session",
        "thank you for joining", "conference call",
    ])
    return text if is_transcript else None


def _clean_text(text: str) -> str:
    """Remove excessive whitespace and non-ASCII noise from raw HTML text."""
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l]               # drop blank lines
    text  = "\n".join(lines)
    text  = re.sub(r"\n{3,}", "\n\n", text)        # max 2 consecutive newlines
    text  = re.sub(r"[^\x00-\x7F]+", " ", text)   # strip non-ASCII
    return text.strip()


def fetch_transcripts_for_ticker(ticker: str,
                                  start: str = "2020-01-01",
                                  end: str = "2024-12-31",
                                  delay: float = 0.2) -> int:
    """
    Fetch and save all earnings call transcripts for a single ticker.
    Returns the number of transcripts saved.
    """
    cik = TICKER_TO_CIK.get(ticker)
    if not cik:
        print(f"  ⚠️  No CIK found for {ticker}")
        return 0

    out_dir = RAW_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    filings = get_8k_filings(cik, start, end)
    print(f"  {ticker}: found {len(filings)} 8-K filings, scanning for transcripts...")

    saved = 0
    for filing in filings:
        time.sleep(delay)                         # Be polite to EDGAR servers
        text = get_transcript_text(cik, filing["accession"])
        if text is None:
            continue
        out_path = out_dir / f"{filing['date']}.txt"
        out_path.write_text(text, encoding="utf-8")
        saved += 1
        print(f"    ✓ Saved transcript: {filing['date']}")

    print(f"  → {ticker}: {saved} transcripts saved")
    return saved


def fetch_all_transcripts(tickers: list[str] | None = None, **kwargs) -> None:
    """Fetch transcripts for all tickers in the universe."""
    if tickers is None:
        tickers = list(TICKER_TO_CIK.keys())
    for ticker in tickers:
        fetch_transcripts_for_ticker(ticker, **kwargs)


if __name__ == "__main__":
    # Quick smoke test — fetch just one ticker
    fetch_transcripts_for_ticker("AAPL")