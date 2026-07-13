# src/data/fetch_transcripts_motleyfool.py

import re
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "transcripts"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Known earnings dates per ticker — YYYY-MM-DD matching Motley Fool URL dates
EARNINGS_DATES = {
    "AAPL": [
        ("2023-02-02", "apple-aapl-q1-2023-earnings-call-transcript"),
        ("2023-05-04", "apple-aapl-q2-2023-earnings-call-transcript"),
        ("2023-08-03", "apple-aapl-q3-2023-earnings-call-transcript"),
        ("2023-11-02", "apple-aapl-q4-2023-earnings-call-transcript"),
        ("2024-02-01", "apple-aapl-q1-2024-earnings-call-transcript"),
        ("2024-05-02", "apple-aapl-q2-2024-earnings-call-transcript"),
        ("2024-08-01", "apple-aapl-q3-2024-earnings-call-transcript"),
        ("2024-10-31", "apple-aapl-q4-2024-earnings-call-transcript"),
    ],
    "MSFT": [
        ("2023-01-24", "microsoft-msft-q2-2023-earnings-call-transcript"),
        ("2023-04-25", "microsoft-msft-q3-2023-earnings-call-transcript"),
        ("2023-07-25", "microsoft-msft-q4-2023-earnings-call-transcript"),
        ("2023-10-24", "microsoft-msft-q1-2024-earnings-call-transcript"),
        ("2024-01-30", "microsoft-msft-q2-2024-earnings-call-transcript"),
        ("2024-04-25", "microsoft-msft-q3-2024-earnings-call-transcript"),
        ("2024-07-30", "microsoft-msft-q4-2024-earnings-call-transcript"),
        ("2024-10-30", "microsoft-msft-q1-2025-earnings-call-transcript"),
    ],
    "NVDA": [
        ("2023-02-22", "nvidia-nvda-q4-2023-earnings-call-transcript"),
        ("2023-05-24", "nvidia-nvda-q1-2024-earnings-call-transcript"),
        ("2023-08-23", "nvidia-nvda-q2-2024-earnings-call-transcript"),
        ("2023-11-21", "nvidia-nvda-q3-2024-earnings-call-transcript"),
        ("2024-02-21", "nvidia-nvda-q4-2024-earnings-call-transcript"),
        ("2024-05-22", "nvidia-nvda-q1-2025-earnings-call-transcript"),
        ("2024-08-28", "nvidia-nvda-q2-2025-earnings-call-transcript"),
        ("2024-11-20", "nvidia-nvda-q3-2025-earnings-call-transcript"),
    ],
    "HD": [
        ("2023-02-21", "home-depot-hd-q4-2022-earnings-call-transcript"),
        ("2023-05-16", "home-depot-hd-q1-2023-earnings-call-transcript"),
        ("2023-08-15", "home-depot-hd-q2-2023-earnings-call-transcript"),
        ("2023-11-14", "home-depot-hd-q3-2023-earnings-call-transcript"),
        ("2024-02-20", "home-depot-hd-q4-2023-earnings-call-transcript"),
        ("2024-05-14", "home-depot-hd-q1-2024-earnings-call-transcript"),
        ("2024-08-13", "home-depot-hd-q2-2024-earnings-call-transcript"),
        ("2024-11-12", "home-depot-hd-q3-2024-earnings-call-transcript"),
    ],
    "MA": [
        ("2023-01-26", "mastercard-ma-q4-2022-earnings-call-transcript"),
        ("2023-04-27", "mastercard-ma-q1-2023-earnings-call-transcript"),
        ("2023-07-27", "mastercard-ma-q2-2023-earnings-call-transcript"),
        ("2023-10-26", "mastercard-ma-q3-2023-earnings-call-transcript"),
        ("2024-01-31", "mastercard-ma-q4-2023-earnings-call-transcript"),
        ("2024-05-01", "mastercard-ma-q1-2024-earnings-call-transcript"),
        ("2024-07-31", "mastercard-ma-q2-2024-earnings-call-transcript"),
        ("2024-10-31", "mastercard-ma-q3-2024-earnings-call-transcript"),
    ],
}

def fetch_transcript(date: str, slug: str) -> str | None:
    """Fetch one transcript by its Motley Fool URL slug."""
    yyyy, mm, dd = date.split("-")
    url = f"https://www.fool.com/earnings/call-transcripts/{yyyy}/{mm}/{dd}/{slug}/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"    ✗ HTTP {resp.status_code} — {slug}")
            return None
    except Exception as e:
        print(f"    ✗ Request failed: {e}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Extract article body
    body = (
        soup.find("div", class_=re.compile(r"article-body", re.I))
        or soup.find("span", id=re.compile(r"article-body", re.I))
        or soup.find("div", id="article-body")
    )
    if not body:
        # Fallback: find largest <div> containing "Operator"
        for div in soup.find_all("div"):
            t = div.get_text()
            if len(t) > 5000 and "Operator" in t:
                body = div
                break

    if not body:
        print(f"    ✗ Body not found — {slug}")
        return None

    text = body.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Confirm it's a transcript
    t = text.lower()
    signals = [
        bool(re.search(r"\boperator\b", t)),
        bool(re.search(r"q&a|question.and.answer", t)),
        bool(re.search(r"your (next |first )?question (comes from|is from)", t)),
    ]
    if sum(signals) < 1:
        print(f"    ✗ No transcript signals — {slug}")
        return None

    return text


def fetch_transcripts_for_ticker(ticker: str, start: str = "2020-01-01",
                                  end: str = "2024-12-31") -> int:
    entries = EARNINGS_DATES.get(ticker, [])
    if not entries:
        print(f"  ⚠  No dates configured for {ticker}")
        return 0

    out_dir = RAW_DIR / ticker
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  {ticker}: {len(entries)} dates configured")
    saved = 0
    for date, slug in entries:
        if not (start <= date <= end):
            continue
        out_path = out_dir / f"{date}.txt"
        if out_path.exists():
            continue
        time.sleep(0.5)
        text = fetch_transcript(date, slug)
        if text is None:
            continue
        out_path.write_text(text, encoding="utf-8")
        print(f"    ✓ {date}")
        saved += 1

    print(f"  → {ticker}: {saved} transcript(s) saved")
    return saved


def fetch_all_transcripts(tickers: list[str] | None = None,
                           **kwargs) -> dict[str, int]:
    if tickers is None:
        tickers = list(EARNINGS_DATES.keys())
    return {t: fetch_transcripts_for_ticker(t, **kwargs) for t in tickers}




if __name__ == "__main__":
    fetch_transcripts_for_ticker("AAPL")