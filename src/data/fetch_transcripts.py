"""
src/data/fetch_transcripts.py
------------------------------
Fetches earnings call transcripts from SEC EDGAR 8-K filings.
Strategy:
  1. Pull all 8-K filings for a ticker via the EDGAR submissions API.
  2. For each filing, fetch the index and find the largest .htm exhibit.
  3. Parse the HTML, strip noise, and apply a multi-signal heuristic to
     confirm the document is an actual call transcript (not a press release).
  4. Save as data/raw/transcripts/{TICKER}/{date}.txt
SEC Fair-Use Note:
  EDGAR allows automated access at ≤10 requests/second.
  We add a 0.2 s delay between requests to stay well under the limit.
  A descriptive User-Agent header is required by the SEC.
"""
import re
import time
import warnings
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from pathlib import Path

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "transcripts"

# ── EDGAR config ──────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "FinSignal Research Project finSignal@example.com",
    "Accept-Encoding": "gzip, deflate",
}

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

# ── EDGAR fetchers ────────────────────────────────────────────────────────────
def get_8k_filings(cik: str, start: str, end: str) -> list[dict]:
    """
    Return metadata for all 8-K filings in [start, end] for the given CIK.
    Uses the EDGAR submissions JSON API (no scraping required).
    Handles paginated older filings via the 'files' key.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers={**HEADERS, "Host": "data.sec.gov"}, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results = []

    def _extract(recent: dict) -> None:
        forms      = recent.get("form", [])
        dates      = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        for form, date, acc in zip(forms, dates, accessions):
            if form == "8-K" and start <= date <= end:
                results.append({
                    "accession": acc.replace("-", ""),
                    "date": date,
                    "cik": cik,
                })

    # Recent filings (always present)
    _extract(data.get("filings", {}).get("recent", {}))

    # Older paginated filings (present when filing history is long)
    for extra_file in data.get("filings", {}).get("files", []):
        extra_url = f"https://data.sec.gov/submissions/{extra_file['name']}"
        try:
            r = requests.get(
                extra_url,
                headers={**HEADERS, "Host": "data.sec.gov"},
                timeout=15,
            )
            r.raise_for_status()
            _extract(r.json())
        except Exception as e:
            print(f"    ⚠  Could not fetch paginated filings ({extra_file['name']}): {e}")

    return results


def _fetch_index(cik: str, accession: str) -> list[dict]:
    """Fetch the filing index and return the list of document entries."""
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession}/index.json"
    )
    resp = requests.get(url, headers={**HEADERS, "Host": "www.sec.gov"}, timeout=15)
    resp.raise_for_status()
    return resp.json().get("directory", {}).get("item", [])


def _fetch_html(cik: str, accession: str, doc_name: str) -> tuple[bytes, str]:
    """Fetch the raw bytes of a single document from a filing."""
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession}/{doc_name}"
    )
    resp = requests.get(url, headers={**HEADERS, "Host": "www.sec.gov"}, timeout=20)
    resp.raise_for_status()
    return resp.content, resp.headers.get("Content-Type", "")


# ── Text extraction & cleaning ────────────────────────────────────────────────
def _html_to_text(content: bytes, content_type: str) -> str:
    """Parse HTML/XML bytes and return plain text."""
    parser = "xml" if "xml" in content_type else "lxml"
    soup = BeautifulSoup(content, parser)
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _clean_text(text: str) -> str:
    """Normalise whitespace and drop non-ASCII noise."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text  = "\n".join(lines)
    text  = re.sub(r"\n{3,}", "\n\n", text)
    text  = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def _is_transcript(text: str) -> bool:
    """
    Return True only if the document looks like a real earnings call transcript.
    Press releases mention 'conference call' in passing; actual transcripts
    contain an Operator turn, Q&A handoffs, and named speaker exchanges.
    We require at least 2 independent signals to avoid false positives.
    """
    t = text.lower()
    signals = [
        bool(re.search(r"\boperator\b", t)),
        bool(re.search(r"q&a|question.and.answer", t)),
        bool(re.search(r"your (next |first )?question (comes from|is from)", t)),
        bool(re.search(r"\[operator (instructions|assist)\]", t)),
        bool(re.search(
            r"(good morning|good afternoon|good evening).{0,80}"
            r"(welcome|thank you for joining|thank you for participating)", t
        )),
    ]
    return sum(signals) >= 2


# ── Main per-filing logic ─────────────────────────────────────────────────────
def get_transcript_text(cik: str, accession: str) -> str | None:
    """
    Given a CIK + accession number, try to extract an earnings call transcript.

    Exhibit selection priority:
      1. Any .htm file whose name contains '99.2' or 'ex992' (transcript exhibit)
      2. Remaining .htm files sorted largest-first (transcript exhibits tend to be big)

    Returns cleaned plain text, or None if no transcript-like document is found.
    """
    try:
        docs = _fetch_index(cik, accession)
    except Exception as e:
        print(f"    ✗ Index fetch failed ({accession}): {e}")
        return None

    htm_docs = [d for d in docs if d.get("name", "").lower().endswith(".htm")]
    if not htm_docs:
        return None

    def _priority(doc: dict) -> int:
        """Lower = higher priority."""
        name = doc.get("name", "").lower()
        
        # Explicit transcript exhibit naming (AAPL, MSFT, JPM style)
        if re.search(r"ex99.?2", name):
            return 0
        # Ticker-named filing (NVDA style: nvda-20240522.htm)
        if re.search(r"^[a-z]{1,5}-\d{8}\.htm$", name):
            return 1
        # CFO commentary / press release: explicitly deprioritise
        if any(x in name for x in ["cfocommentary", "pr.htm", "8k.htm", "r1.htm"]):
            return 3
        # Fallback: largest file wins
        return 2

    ordered = sorted(htm_docs, key=lambda d: (_priority(d), -int(d.get("size", 0))))

    for doc in ordered:
        try:
            content, ct = _fetch_html(cik, accession, doc["name"])
        except Exception as e:
            print(f"    ✗ Doc fetch failed ({doc['name']}): {e}")
            continue

        text = _clean_text(_html_to_text(content, ct))
        if _is_transcript(text):
            return text

    return None  # No transcript found in this filing


# ── Public API ────────────────────────────────────────────────────────────────
def fetch_transcripts_for_ticker(
    ticker: str,
    start:  str   = "2020-01-01",
    end:    str   = "2024-12-31",
    delay:  float = 0.2,
) -> int:
    """
    Fetch and save all earnings call transcripts for a single ticker.
    Skips dates that already have a saved file.
    Returns the number of new transcripts saved.
    """
    cik = TICKER_TO_CIK.get(ticker)
    if not cik:
        print(f"  ⚠  No CIK found for {ticker}")
        return 0

    out_dir = RAW_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    filings = get_8k_filings(cik, start, end)
    print(f"  {ticker}: {len(filings)} 8-K filings found, scanning...")

    saved = 0
    for filing in filings:
        out_path = out_dir / f"{filing['date']}.txt"
        if out_path.exists():
            continue  # already fetched, skip without hitting the network

        time.sleep(delay)
        text = get_transcript_text(cik, filing["accession"])
        if text is None:
            continue

        out_path.write_text(text, encoding="utf-8")
        saved += 1
        print(f"    ✓ {filing['date']}")

    print(f"  → {ticker}: {saved} new transcript(s) saved")
    return saved


def fetch_all_transcripts(
    tickers: list[str] | None = None,
    **kwargs,
) -> dict[str, int]:
    """
    Fetch transcripts for every ticker in the universe (or a provided subset).
    Returns a dict of {ticker: n_saved}.
    """
    if tickers is None:
        tickers = list(TICKER_TO_CIK.keys())
    return {t: fetch_transcripts_for_ticker(t, **kwargs) for t in tickers}


if __name__ == "__main__":
    fetch_transcripts_for_ticker("AAPL")