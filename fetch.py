"""
sec_bot — fetch
===============

Single-ticker end-to-end EDGAR fetch and section split.

Given a ticker, this script:
  1. Resolves the ticker to a CIK via SEC's ticker mapping.
  2. Pulls the company's filing history.
  3. Finds the most recent 10-K.
  4. Downloads the filing document.
  5. Splits it into canonical Items using the edgartools HTMLParser.
  6. Saves the sectioned output to disk under ./output/<TICKER>/<accession>/

Run:
    python fetch.py AAPL

This is the single-ticker entry point. For batch processing across a whole
universe of tickers, see batch.py.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from edgar.documents import HTMLParser as EdgarHTMLParser, ParserConfig as EdgarParserConfig

# Modern 10-Ks are inline XBRL (XML embedded in HTML). BS4 warns; ignore it.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USER_AGENT = "Carson R carson.research@gmail.com"

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

# Headers for data.sec.gov endpoints (different Host header)
DATA_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dashes}/{filename}"

# SEC asks for <= 10 req/s. We use a conservative 0.15s delay (~6 req/s).
REQUEST_DELAY_SECONDS = 0.15

OUTPUT_DIR = Path(__file__).parent / "output"

# Per-filing data (raw HTML, sections, manifests) lives under output/filings/
# rather than directly in output/, so the top-level output directory stays
# clean for the human-facing artifacts (universe.json, run_manifest.json,
# extractions.json, ranking.json, digest.md).
FILINGS_DIR = OUTPUT_DIR / "filings"

# Sections we extract and keep. Items 10-16 are detected as terminators only.
SECTIONS_TO_KEEP = {
    "item_1", "item_1a", "item_1b", "item_1c",
    "item_2", "item_3", "item_4",
    "item_5", "item_6",
    "item_7", "item_7a", "item_8",
    "item_9", "item_9a", "item_9b", "item_9c",
}

# Heading regex: "Item N" or "Item NA" at line start, with optional leading
# punctuation/page-numbers. Doesn't require the title to be on the same line.
ITEM_HEADING_RE = re.compile(
    r"^[ \t\>\|\.\-\u2013\u2014\u2022\u00b7\*\d]{0,12}"
    r"item\s+(\d+[a-z]?)\b",
    re.IGNORECASE | re.MULTILINE,
)

# PART heading regex: "PART I", "PART II", etc. Roman numerals only.
PART_HEADING_RE = re.compile(
    r"^[ \t\>\|\.\-\u2013\u2014\u2022\u00b7\*]{0,6}"
    r"part\s+(i{1,3}|iv|v)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Used by the legacy split_into_sections caller — kept as an empty dict so
# existing imports / tests don't break.
ITEM_PATTERNS: dict[str, str] = {}


def _line_at(text: str, pos: int) -> str:
    """Return the line of `text` containing position `pos`, stripped."""
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].strip()


# ---------------------------------------------------------------------------
# Tiny rate-limited HTTP wrapper
# ---------------------------------------------------------------------------

_last_request_at = 0.0


def _http_get(url: str, headers: dict) -> requests.Response:
    """GET with a global rate-limiter and basic 429 backoff."""
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
            print(f"  [rate limited] sleeping {wait}s before retry...", file=sys.stderr)
            time.sleep(wait)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"GET {url} failed after retries")


# ---------------------------------------------------------------------------
# 1. Ticker -> CIK
# ---------------------------------------------------------------------------

def resolve_ticker_to_cik(ticker: str) -> tuple[str, str]:
    """
    Returns (cik_padded_10, company_name). The padded form is what the
    submissions endpoint expects.
    """
    print(f"[1/5] Resolving {ticker} -> CIK...")
    resp = _http_get(TICKER_MAP_URL, HEADERS)
    data = resp.json()
    # The map is keyed by stringified integers; values are dicts with
    # 'cik_str' (int), 'ticker' (str), 'title' (str).
    target = ticker.upper()
    for _, row in data.items():
        if row["ticker"].upper() == target:
            cik_padded = str(row["cik_str"]).zfill(10)
            print(f"      {ticker} = CIK {cik_padded} ({row['title']})")
            return cik_padded, row["title"]
    raise ValueError(f"Ticker {ticker!r} not found in SEC ticker map")


# ---------------------------------------------------------------------------
# 2. Filing history
# ---------------------------------------------------------------------------

@dataclass
class FilingRef:
    accession: str            # e.g. "0000320193-24-000123"
    accession_no_dashes: str  # e.g. "000032019324000123"
    form: str                 # e.g. "10-K"
    filing_date: str          # YYYY-MM-DD
    primary_document: str     # e.g. "aapl-20240928.htm"
    cik_int: int


def get_recent_10k(cik_padded: str) -> FilingRef:
    """Fetch the company's submissions index and return the most recent 10-K."""
    print(f"[2/5] Fetching filing history for CIK {cik_padded}...")
    url = SUBMISSIONS_URL.format(cik=cik_padded)
    resp = _http_get(url, DATA_HEADERS)
    data = resp.json()

    recent = data["filings"]["recent"]
    # 'recent' is column-oriented: parallel arrays for each field
    forms = recent["form"]
    dates = recent["filingDate"]
    accessions = recent["accessionNumber"]
    primary_docs = recent["primaryDocument"]

    for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form == "10-K":
            cik_int = int(cik_padded)
            ref = FilingRef(
                accession=acc,
                accession_no_dashes=acc.replace("-", ""),
                form=form,
                filing_date=date,
                primary_document=doc,
                cik_int=cik_int,
            )
            print(f"      Most recent 10-K: filed {date}, accession {acc}")
            print(f"      Primary document: {doc}")
            return ref

    raise RuntimeError("No 10-K found in recent filings")


# ---------------------------------------------------------------------------
# 3. Download filing
# ---------------------------------------------------------------------------

def download_filing(ref: FilingRef, dest_path: Path) -> str:
    """Download the primary 10-K document. Returns the raw HTML text."""
    print(f"[3/5] Downloading 10-K HTML...")
    url = ARCHIVE_URL.format(
        cik_int=ref.cik_int,
        accession_no_dashes=ref.accession_no_dashes,
        filename=ref.primary_document,
    )
    resp = _http_get(url, HEADERS)
    text = resp.text
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(text, encoding="utf-8")
    print(f"      Saved raw HTML: {dest_path} ({len(text):,} chars)")
    return text


# ---------------------------------------------------------------------------
# 4. HTML -> plain text
# ---------------------------------------------------------------------------

def html_to_text(html: str) -> str:
    """
    Convert filing HTML to plain text suitable for section splitting.

    Modern 10-Ks are inline XBRL — lots of <ix:nonNumeric>, <ix:nonFraction>
    tags wrapping ordinary text. BeautifulSoup handles these fine; we just
    want the visible text.
    """
    print("[4/5] Stripping HTML to plain text...")
    soup = BeautifulSoup(html, "lxml")

    # Drop tags that never carry filing prose.
    for tag in soup(["script", "style", "head", "meta", "link"]):
        tag.decompose()

    # Tables in 10-Ks contain real data (financials, schedules). We keep them
    # but flatten — get_text with a separator preserves structure adequately
    # for section detection. The extractor LLM will get the cleaner per-section
    # text downstream.
    text = soup.get_text(separator="\n")

    # Normalize whitespace: collapse runs of blank lines, trim trailing space.
    lines = [ln.rstrip() for ln in text.splitlines()]
    # Collapse 3+ consecutive blank lines to 2.
    cleaned: list[str] = []
    blank_run = 0
    for ln in lines:
        if not ln.strip():
            blank_run += 1
            if blank_run <= 2:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(ln)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# 5. Section splitting
# ---------------------------------------------------------------------------

@dataclass
class SectionHit:
    item_id: str    # e.g. "1a", "9c" (no "item_" prefix)
    start: int      # char offset of line containing the heading
    line: str       # full heading line, stripped


def find_part_anchors(text: str) -> dict[str, int]:
    """
    Find structural PART I/II/III/IV anchors in the document body, using a
    gap-based heuristic.

    A document typically mentions each PART twice: once in the TOC (where all
    PARTs are clustered together with little content between them) and once
    as the body header (where each PART is followed by significant prose
    before the next PART). For each roman numeral, we take the occurrence
    whose distance to the next PART occurrence is largest — that's the body
    anchor.

    Returns: {"i": offset, "ii": offset, ...}
    """
    all_hits: list[tuple[str, int]] = []
    for m in PART_HEADING_RE.finditer(text):
        roman = m.group(1).lower()
        line = _line_at(text, m.start())
        # Real PART headings are short. Filters out prose mentions like
        # "as described in Part II of this report".
        if len(line) > 60:
            continue
        all_hits.append((roman, m.start()))

    if not all_hits:
        return {}

    by_roman: dict[str, tuple[int, int]] = {}   # roman -> (pos, gap_to_next)
    for i, (roman, pos) in enumerate(all_hits):
        next_pos = all_hits[i + 1][1] if i + 1 < len(all_hits) else len(text) + 10**9
        gap = next_pos - pos
        existing = by_roman.get(roman)
        if existing is None or existing[1] < gap:
            by_roman[roman] = (pos, gap)
    return {roman: pos for roman, (pos, _) in by_roman.items()}


def find_item_hits(text: str) -> list[SectionHit]:
    """
    Find every line that begins with `Item N` (relaxed). Identifies each by
    item number / suffix only — the title may be on the same line, the next
    line, or absent. Filters out long lines (prose mentions of "Item N").
    """
    hits: list[SectionHit] = []
    for m in ITEM_HEADING_RE.finditer(text):
        line = _line_at(text, m.start())
        if len(line) > 200:
            continue
        item_id = m.group(1).lower()
        line_start = text.rfind("\n", 0, m.start()) + 1
        hits.append(SectionHit(item_id=item_id, start=line_start, line=line))
    return hits


def _print_diagnostic(text: str) -> None:
    """When nothing matched, dump candidate lines so the user can paste back."""
    print("      Diagnostic — first 20 lines containing 'item' (case-insensitive):", file=sys.stderr)
    count = 0
    for ln in text.split("\n"):
        if "item" in ln.lower():
            preview = ln.strip()[:140]
            print(f"        {preview!r}", file=sys.stderr)
            count += 1
            if count >= 20:
                break
    if count == 0:
        print("        (no 'item' substrings found in cleaned text at all)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

# Heuristic ranges for sanity checks. Wide on purpose — only fires on extreme
# outliers so the user can spot probable detection bugs without false alarms.
SECTION_SIZE_EXPECTATIONS = {
    # key:        (warn_below, warn_above, description)
    "item_1":     (500,  20000, "Business — usually 1k-10k words"),
    "item_1a":    (500,  35000, "Risk Factors — usually 5k-25k words"),
    "item_7":     (500,  25000, "MD&A — usually 2k-15k words"),
    "item_8":     (500,  60000, "Financial Statements — usually 5k-40k words"),
    "item_6":     (None, 500,   "Item 6 should be 'Reserved' (<<500 words)"),
}

# Phrases that strongly suggest a short section is a "stub" — i.e., the
# company filed the actual content as a separate exhibit and is just
# cross-referencing. False-positive risk on these phrases is very low; they
# don't appear in genuine MD&A or Risk Factors prose.
STUB_PHRASES = (
    "reference is made to",
    "incorporated by reference",
    "incorporated herein by reference",      # variant with "herein" inserted
    "see the financial section",
    "see the consolidated financial statements",
    "see notes to consolidated",
    "see exhibit",
    "see item 8",                            # Item 7 stubs often cross-reference Item 8
    "see item 15",                           # exhibits index
    "index to consolidated financial statements",  # IFF, ITT pattern
    "page f-",                               # references to back-of-filing exhibit pages (F-1, F-15, etc.)
    "filed on the pages listed below",       # EXPD-style: lists schedule pages with "F-" prefix
)


def _is_stub(text: str) -> bool:
    """
    True if `text` contains explicit cross-reference language that suggests
    the section is intentionally empty (content is filed elsewhere).

    Should only be called on sections already known to be unusually short —
    this function does not check length itself.

    Whitespace is normalized before matching: SEC filings frequently use
    non-breaking spaces (U+00A0) between heading numbers and content (e.g.,
    "page\u00a0F-1" instead of "page F-1"), and our STUB_PHRASES use regular
    spaces. Without normalization, "page f-" would miss "page\u00a0f-1".
    """
    # Replace NBSP and other Unicode whitespace with regular spaces, then
    # collapse runs of whitespace. The result still preserves word boundaries.
    normalized = " ".join(text.split()).lower()
    return any(p in normalized for p in STUB_PHRASES)


@dataclass
class Anomaly:
    """
    A structured anomaly description. Downstream stages route on `kind` and
    filter on `severity` rather than parsing the human-readable `message`.

    Severity:
        "warn" — real issue; section data is likely unusable.
        "info" — known/expected pattern (e.g., stub); section is empty by
                 design but the rest of the filing is fine.

    Kind:
        "too_small" — below expected lower bound. Real splitter miss.
        "too_large" — above expected upper bound. TOC bleed or merge.
        "stub"      — too small AND contains cross-reference language.
                      Section is intentionally a pointer to an exhibit.
    """
    section: str                                       # e.g. "item_7"
    kind: str                                          # too_small | too_large | stub
    severity: str                                      # warn | info
    observed_value: int                                # word count
    expected_range: tuple[Optional[int], Optional[int]]  # (low, high), either may be None
    message: str                                       # human-readable for logs


def _detect_anomalies(sections: dict[str, str]) -> list[Anomaly]:
    """
    Inspect each section against `SECTION_SIZE_EXPECTATIONS` and return a
    list of Anomaly records. Empty list means the filing looks normal.
    """
    out: list[Anomaly] = []
    for key, (lo, hi, desc) in SECTION_SIZE_EXPECTATIONS.items():
        if key not in sections:
            continue
        text = sections[key]
        wc = len(text.split())

        if lo is not None and wc < lo:
            # Distinguish "splitter missed it" from "company filed it
            # elsewhere." Stub-language detection has very low false-positive
            # risk because the phrases are highly specific to cross-references.
            if _is_stub(text):
                out.append(Anomaly(
                    section=key,
                    kind="stub",
                    severity="info",
                    observed_value=wc,
                    expected_range=(lo, hi),
                    message=(f"{key} = {wc:,} words is a STUB "
                             f"(cross-references content elsewhere). "
                             f"Section intentionally empty here."),
                ))
            else:
                out.append(Anomaly(
                    section=key,
                    kind="too_small",
                    severity="warn",
                    observed_value=wc,
                    expected_range=(lo, hi),
                    message=(f"{key} = {wc:,} words is unusually SMALL "
                             f"({desc}). Possible miss."),
                ))
        if hi is not None and wc > hi:
            out.append(Anomaly(
                section=key,
                kind="too_large",
                severity="warn",
                observed_value=wc,
                expected_range=(lo, hi),
                message=(f"{key} = {wc:,} words is unusually LARGE "
                         f"({desc}). Possible TOC bleed."),
            ))
    return out


def split_into_sections(html: str) -> dict[str, str]:
    """
    Split a 10-K filing into canonical sections using edgartools' HTMLParser.

    Replaces the previous regex-based splitter, which was brittle on filings
    with non-standard structure (banks, conglomerates, oddballs). edgartools
    has been tested across thousands of filings and handles the structural
    edge cases — TOC vs body disambiguation, inline XBRL, multi-page TOCs,
    items broken across lines, etc.

    Input:  raw filing HTML (NOT the cleaned text — edgartools needs the
            structural HTML to identify section boundaries).
    Output: dict mapping canonical keys (item_1, item_1a, ...) to section
            text. Only sections in SECTIONS_TO_KEEP are returned.

    Sections edgartools missed are simply absent from the dict — preferable
    to the regex splitter's tendency to produce mangled bleed-through.
    """
    print("[5/5] Splitting into sections (edgartools)...")

    # edgartools' canonical section names for 10-Ks → our keys.
    EDGAR_NAME_TO_KEY = {
        "business":             "item_1",
        "risk_factors":         "item_1a",
        "properties":           "item_2",
        "legal_proceedings":    "item_3",
        "mda":                  "item_7",
        "market_risk":          "item_7a",
        "financial_statements": "item_8",
        "controls_procedures":  "item_9a",
    }

    config = EdgarParserConfig(form="10-K", detect_sections=True)
    parser = EdgarHTMLParser(config)
    doc = parser.parse(html)

    sections: dict[str, str] = {}
    seen_texts: dict[str, str] = {}   # text hash → key it was first assigned
    for name, section in doc.sections.items():
        # Prefer the explicit `item` identifier when edgartools provides one
        # — it covers items the named-pattern map doesn't (1B, 1C, 4, 5, 6,
        # 9, 9B, 9C). Fall back to the named-pattern mapping otherwise.
        key: Optional[str] = None
        if section.item:
            key = f"item_{section.item.lower()}"
        elif name in EDGAR_NAME_TO_KEY:
            key = EDGAR_NAME_TO_KEY[name]

        if key is None or key not in SECTIONS_TO_KEEP:
            continue

        try:
            text = section.text()
        except Exception as e:
            print(f"      WARN: could not extract text for {key}: {e}", file=sys.stderr)
            continue

        text = text.strip()
        if not text:
            continue

        # If we get duplicates (e.g., section appears under both name and item
        # identifier), keep the longer one — almost always the body section.
        if key in sections and len(sections[key]) >= len(text):
            continue

        # Detect cross-key duplicates (e.g., BRK-B's narrative style assigns
        # the same content to both "business" and "mda"). Skip if this exact
        # text was already mapped to a different key.
        text_hash = f"{len(text)}:{text[:200]}"
        if text_hash in seen_texts and seen_texts[text_hash] != key:
            print(f"      NOTE: skipping duplicate content for {key} (same as {seen_texts[text_hash]})", file=sys.stderr)
            continue
        seen_texts[text_hash] = key

        sections[key] = text

    if not sections:
        print("      WARNING: edgartools detected no sections", file=sys.stderr)
        return {}

    # Print in canonical 10-K order so the user sees a stable layout.
    canonical_order = [
        "item_1", "item_1a", "item_1b", "item_1c",
        "item_2", "item_3", "item_4",
        "item_5", "item_6",
        "item_7", "item_7a", "item_8",
        "item_9", "item_9a", "item_9b", "item_9c",
    ]
    for key in canonical_order:
        if key in sections:
            wc = len(sections[key].split())
            print(f"      {key:10s}  {wc:>7,} words")

    # Anomaly notes — purely advisory. The actual structured anomaly list is
    # reconstructed by callers (batch.py) via _detect_anomalies(); this loop
    # is purely for the operator's eyes when running fetch.py directly.
    anomalies = _detect_anomalies(sections)
    if anomalies:
        print(file=sys.stderr)
        print("      ANOMALY NOTES:", file=sys.stderr)
        for a in anomalies:
            tag = "INFO" if a.severity == "info" else "WARN"
            print(f"        [{tag}] {a.message}", file=sys.stderr)

    return sections


def find_toc_region(item_hits, text_len):
    """Deprecated stub — retained for legacy test imports."""
    return None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def fetch_filing(ticker: str) -> Path:
    """Run the full pipeline for a single ticker. Returns the output dir."""
    cik_padded, company_name = resolve_ticker_to_cik(ticker)
    ref = get_recent_10k(cik_padded)

    out_dir = FILINGS_DIR / ticker.upper() / ref.accession
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw.htm"
    if raw_path.exists():
        print(f"[3/5] Cached HTML found at {raw_path}, skipping download.")
        html = raw_path.read_text(encoding="utf-8")
    else:
        html = download_filing(ref, raw_path)

    text = html_to_text(html)
    text_path = out_dir / "cleaned.txt"
    text_path.write_text(text, encoding="utf-8")
    print(f"      Saved cleaned text: {text_path} ({len(text):,} chars)")

    # NOTE: split_into_sections operates on the raw HTML, not cleaned text.
    # edgartools relies on structural HTML cues (bold tags, headings, font
    # styles) to disambiguate TOC entries from body sections.
    sections = split_into_sections(html)
    anomalies = _detect_anomalies(sections)

    sections_dir = out_dir / "sections"
    sections_dir.mkdir(exist_ok=True)
    for key, content in sections.items():
        (sections_dir / f"{key}.txt").write_text(content, encoding="utf-8")

    # Manifest with metadata that downstream stages will read. Anomalies are
    # serialized as dicts so JSON round-trips cleanly; batch.py and any
    # other consumer can hydrate them back into Anomaly via _anomaly_from_dict.
    manifest = {
        "ticker": ticker.upper(),
        "company_name": company_name,
        "cik": cik_padded,
        "form": ref.form,
        "filing_date": ref.filing_date,
        "accession": ref.accession,
        "primary_document": ref.primary_document,
        "sections_found": list(sections.keys()),
        "sections_word_counts": {k: len(v.split()) for k, v in sections.items()},
        "anomalies": [asdict(a) for a in anomalies],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print(f"Done. Output: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Fetch one company's most recent 10-K from EDGAR.")
    parser.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    args = parser.parse_args()
    fetch_filing(args.ticker)


if __name__ == "__main__":
    main()
