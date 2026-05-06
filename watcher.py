"""
watcher.py — EDGAR filing watcher
==================================

Given a built universe (output/universe.json), find all filings of specified
form types within a recency window for each ticker, and write a watch queue
(output/watch_queue.json) to be processed by batch.py.

Run:
    python watcher.py                              # default: 10-K, last 90 days
    python watcher.py --form 10-K --days 365       # one full year
    python watcher.py --form 10-K --days 90 --limit 50    # cap the queue size
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from universe import load_universe, UniverseEntry

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USER_AGENT = "Carson R carson.research@gmail.com"

OUTPUT_DIR = Path(__file__).parent / "output"
WATCH_QUEUE_PATH = OUTPUT_DIR / "watch_queue.json"

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# SEC requests no more than 10 req/s. We use a conservative ~6 req/s.
REQUEST_DELAY_SECONDS = 0.15

DEFAULT_FORM = "10-K"
DEFAULT_DAYS = 90


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class WatchedFiling:
    ticker: str
    cik: str
    company_name: str
    form: str
    filing_date: str           # YYYY-MM-DD
    accession: str             # e.g. "0000320193-25-000079"
    primary_document: str      # e.g. "aapl-20250927.htm"


# ---------------------------------------------------------------------------
# Rate-limited HTTP wrapper
# ---------------------------------------------------------------------------

_last_request_at = 0.0


def _http_get(url: str, headers: dict) -> requests.Response:
    """GET with global rate-limiter and basic 429 backoff."""
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < REQUEST_DELAY_SECONDS:
        time.sleep(REQUEST_DELAY_SECONDS - elapsed)

    for attempt in range(3):
        resp = requests.get(url, headers=headers, timeout=30)
        _last_request_at = time.monotonic()
        if resp.status_code == 200:
            return resp
        if resp.status_code == 429:
            wait = 2 ** attempt
            print(f"  [rate-limited] sleeping {wait}s before retry...", file=sys.stderr)
            time.sleep(wait)
            continue
        if resp.status_code == 404:
            return resp   # caller decides whether 404 is fatal
        resp.raise_for_status()
    raise RuntimeError(f"GET {url} failed after retries")


# ---------------------------------------------------------------------------
# Filing lookup
# ---------------------------------------------------------------------------

def _filings_for_ticker(entry: UniverseEntry, form: str, since: datetime) -> list[WatchedFiling]:
    """
    Return all filings of `form` type filed since `since` for the given ticker.
    SEC's submissions endpoint returns the most-recent ~1000 filings; that's
    way more than enough for a 365-day window even on the busiest filers.
    """
    url = SUBMISSIONS_URL.format(cik=entry.cik)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }
    resp = _http_get(url, headers)
    if resp.status_code == 404:
        # Some CIKs in the ticker map don't have submissions endpoints
        # (e.g., entity merged or deregistered). Skip silently.
        return []

    data = resp.json()
    recent = data["filings"]["recent"]
    forms = recent["form"]
    dates = recent["filingDate"]
    accessions = recent["accessionNumber"]
    primary_docs = recent["primaryDocument"]

    out: list[WatchedFiling] = []
    for f, d, acc, doc in zip(forms, dates, accessions, primary_docs):
        if f != form:
            continue
        try:
            filed_dt = datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            continue
        if filed_dt < since:
            continue
        out.append(WatchedFiling(
            ticker=entry.ticker,
            cik=entry.cik,
            company_name=entry.company_name,
            form=f,
            filing_date=d,
            accession=acc,
            primary_document=doc,
        ))
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_watch_queue(form: str = DEFAULT_FORM,
                      days: int = DEFAULT_DAYS,
                      limit: Optional[int] = None) -> list[WatchedFiling]:
    """
    Walk the universe, find all matching filings, return the watch queue.

    `limit` caps the total queue size (oldest filings dropped first). Useful
    for testing without committing to a full run.
    """
    universe = load_universe()
    print(f"[watch] Universe has {len(universe)} tickers")

    since = datetime.now() - timedelta(days=days)
    print(f"[watch] Looking for {form} filings since {since:%Y-%m-%d} ({days} days)")

    all_filings: list[WatchedFiling] = []
    found_count = 0
    not_found_count = 0
    error_count = 0
    start = time.monotonic()

    for i, entry in enumerate(universe, 1):
        try:
            filings = _filings_for_ticker(entry, form, since)
            all_filings.extend(filings)
            if filings:
                found_count += len(filings)
            else:
                not_found_count += 1
        except Exception as e:
            error_count += 1
            print(f"  [{entry.ticker}] error: {e}", file=sys.stderr)

        if i % 50 == 0 or i == len(universe):
            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(universe) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(universe)} processed "
                  f"({rate:.1f}/s, ~{eta:.0f}s remaining, "
                  f"{found_count} filings found)")

    print()
    print(f"[watch] Done. {found_count} filings across {len(universe) - not_found_count - error_count} tickers")
    print(f"        ({not_found_count} no recent filings, {error_count} errors)")

    # Sort by filing date descending (newest first), then by ticker.
    all_filings.sort(key=lambda f: (f.filing_date, f.ticker), reverse=True)

    if limit is not None and len(all_filings) > limit:
        print(f"[watch] Capping queue at {limit} (dropping {len(all_filings) - limit} older filings)")
        all_filings = all_filings[:limit]

    return all_filings


def save_watch_queue(filings: list[WatchedFiling], path: Path = WATCH_QUEUE_PATH) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": len(filings),
        "filings": [asdict(f) for f in filings],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"[save]  Watch queue written to {path} ({len(filings)} filings)")


def load_watch_queue(path: Path = WATCH_QUEUE_PATH) -> list[WatchedFiling]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run watcher.py first")
    payload = json.loads(path.read_text())
    return [WatchedFiling(**f) for f in payload["filings"]]


def main():
    parser = argparse.ArgumentParser(description="Build a queue of recent filings to fetch.")
    parser.add_argument("--form", default=DEFAULT_FORM,
                        help=f"Form type to watch for (default: {DEFAULT_FORM})")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"Filing recency window in days (default: {DEFAULT_DAYS})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap queue size (newest filings kept)")
    args = parser.parse_args()

    filings = build_watch_queue(form=args.form, days=args.days, limit=args.limit)
    save_watch_queue(filings)

    if filings:
        print()
        print("Newest 10 filings in queue:")
        for f in filings[:10]:
            print(f"  {f.filing_date}  {f.ticker:<6}  {f.form:<6}  {f.company_name[:50]}")


if __name__ == "__main__":
    main()
