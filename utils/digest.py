"""
digest.py — Phase 8: human-readable digest with LLM-written summaries
=======================================================================

Reads extractions.json (Phase 4) and ranking.json (Phase 7) and produces a
clean markdown report at output/digest.md. For each filing, calls Claude
Haiku 4.5 once to write a 2-3 paragraph synthesis from the structured facts
the extraction layer already captured.

The LLM is given ONLY the extracted JSON facts plus the ranker's triggers —
it does not get the raw filing text. This keeps it from hallucinating
beyond what extract.py already validated, while still letting it write
prose that connects facts (e.g., "the CFO change coincides with the major
impairment, suggesting the company is reckoning with M&A costs").

Cost: roughly $0.02-0.03 per filing using Haiku 4.5. 100 filings ≈ $2-3.

Caching: summaries are cached in output/digest_summaries.json keyed by
(ticker, accession). Re-running digest.py without changing the underlying
extraction data is free — only NEW or CHANGED filings hit the API again.
This means tweaking the digest's layout (which sections to show, ordering,
etc.) is free to iterate on.

Usage:
    python digest.py                           # generate digest, use cache where possible
    python digest.py --refresh-summaries       # force re-generate all LLM summaries
    python digest.py --no-llm                  # skip LLM, use template-only fallback
    python digest.py --out my_digest.md        # custom output path

Output: output/digest.md (markdown, renders nicely anywhere — also
readable as plain text).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from fetch import OUTPUT_DIR, DATA_DIR

EXTRACTIONS_PATH = DATA_DIR / "extractions.json"
RANKING_PATH = DATA_DIR / "ranking.json"
SUMMARIES_PATH = DATA_DIR / "digest_summaries.json"
RUN_MANIFEST_PATH = DATA_DIR / "run_manifest.json"
WATCH_QUEUE_PATH = DATA_DIR / "watch_queue.json"

# Markdown deliverable lives at the top of output/ — easy to find, while
# the JSON artifacts (above) stay tucked away in data/.
DEFAULT_DIGEST_PATH = OUTPUT_DIR / "sec_report.md"

# Load .env from the project root (one level up from utils/).
load_dotenv(Path(__file__).parent.parent / ".env")

SUMMARY_MODEL = "claude-haiku-4-5-20251001"
SUMMARY_MAX_TOKENS = 900    # raised from 600 — generous ceiling for the 4-paragraph prompt.
                            # Haiku writes what it needs and stops, so a higher cap doesn't
                            # cost more (you pay for tokens emitted, not tokens allocated).
                            # Prevents mid-sentence truncation on filings with rich extractions.
SUMMARY_RETRIES = 2


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_usd(amount: float | int | None) -> str:
    """Format a USD amount in human-friendly units. None -> '?'."""
    if amount is None:
        return "?"
    if amount >= 1e9:
        return f"${amount/1e9:.1f}B"
    if amount >= 1e6:
        return f"${amount/1e6:.0f}M"
    if amount >= 1e3:
        return f"${amount/1e3:.0f}K"
    return f"${amount:.0f}"


# ---------------------------------------------------------------------------
# LLM summary generation
# ---------------------------------------------------------------------------

# The prompt is written to lock the LLM down hard. The single most likely
# failure mode is the model adding stuff that's not in the extraction data —
# e.g., "given AAPL's strong margins" when nothing about margins is in the
# input. We forbid that explicitly. The model gets ONE job: prose synthesis
# of the structured facts already provided.
SUMMARY_PROMPT_TEMPLATE = """You are writing a short research note for an investor about a 10-K filing.

You will be given the structured facts our extractor pulled from the filing and the triggers that fired.

Use ONLY the facts provided. Do not invent context, prior history, industry trends, or any company name not given in the data.

Write a research note in 3-4 paragraphs covering, in this order:

BUSINESS — what the company does in plain language. Cover the segments (with brief descriptions if present), primary geographies, and how the company positions itself competitively. If financial_performance data is present in the extraction, open this paragraph with the top-line revenue figure and YoY direction written as natural prose (e.g., "Revenue declined 8% to $1.2B in fiscal 2025, with gross margin compressing from 38% to 31%"). Do not label these as "financial_performance" — weave them into the description naturally.

WHY THIS FILING MATTERS — synthesize the events, governance changes, impairments, litigation, and regulatory matters into prose. Connect related facts where the data supports it. Open with the substantive fact, not a label about the filing's importance. Do NOT use the words "critical", "notable", "quiet", or paraphrases like "warrants attention", "of note", "deserves scrutiny". Skip this paragraph only if the triggers list below is empty AND the events_and_catalysts fields are all empty.

STRUCTURAL POSITION — surface concrete details from the data that shape how this business actually operates and what risks it carries. Draw from customer/vendor concentrations, self-disclosed disadvantages, and named competitors. If operating margin or gross margin figures are present in financial_performance and are distinctive (unusually high, low, or changing materially), include them here as natural prose. Surface the items that are distinctive to this company; skip generic ones (every company faces "competition" and "regulatory risk"). Use whatever percentages, named parties, or descriptors the extraction provides verbatim — do not invent figures. Skip this paragraph entirely if the relevant fields have nothing substantive.

FORWARD — capex plans, named initiatives, and revenue or EPS guidance if provided. If guidance figures are present (revenue_guidance_usd, eps_guidance_value), state them specifically. Brief.

GENERAL RULES:
- Use specific dollar amounts, percentages, and dates from the data wherever they exist
- Factual and dispassionate. No "investors should..." statements. No recommendations.
- Roughly 300-500 words. Adjust length to match how much substance exists — filings with rich financial data warrant longer notes.
- Plain paragraphs separated by blank lines. No headings, bullets, or markdown formatting.
- Refer to the company by its name if provided in CONTEXT; otherwise use the ticker. Do NOT invent a company name from the ticker.
- Begin with prose, not a heading or label.

CONTEXT:
TICKER: {ticker}
COMPANY NAME: {company_name}

TRIGGERS THAT FIRED:
{triggers}

EXTRACTED FACTS:
{extraction_json}

Write the research note now."""


def _build_summary_prompt(ticker: str, ext: dict, ranking_entry: dict,
                          company_name: str = "") -> str:
    """Construct the prompt for the summary LLM call."""
    triggers = ranking_entry.get("triggers", [])
    triggers_text = "\n".join(f"- {t}" for t in triggers) if triggers else "(no triggers fired)"

    # Strip the metadata fields that aren't useful for the LLM. They just
    # take up tokens without helping the synthesis.
    ext_for_llm = {k: v for k, v in ext.items()
                   if k not in ("ticker", "accession", "extraction_version",
                                "model", "generated_at")}

    return SUMMARY_PROMPT_TEMPLATE.format(
        ticker=ticker,
        company_name=company_name or "(not provided — use ticker only)",
        triggers=triggers_text,
        extraction_json=json.dumps(ext_for_llm, indent=2),
    )


def _call_summary_llm(client, ticker: str, ext: dict, ranking_entry: dict,
                      company_name: str = "") -> str:
    """
    Make one Haiku call for the summary. Retries on transient errors. Returns
    the prose summary text, or raises on persistent failure.
    """
    prompt = _build_summary_prompt(ticker, ext, ranking_entry, company_name)

    last_err: Optional[Exception] = None
    for attempt in range(SUMMARY_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=SUMMARY_MODEL,
                max_tokens=SUMMARY_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            return text
        except Exception as e:
            last_err = e
            if attempt < SUMMARY_RETRIES:
                # Brief backoff before retry. We don't get fancy here because
                # most failures will be one-off (network blip, rate limit) and
                # the retry usually succeeds.
                time.sleep(1.5 * (attempt + 1))
                continue
    raise RuntimeError(f"Summary LLM failed for {ticker} after retries: {last_err}")


def _load_summary_cache() -> dict:
    """Load the summary cache. Returns empty dict on any error."""
    if not SUMMARIES_PATH.exists():
        return {}
    try:
        return json.loads(SUMMARIES_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_company_names() -> dict:
    """
    Build {ticker: company_name} lookup from watch_queue.json or run_manifest.json.

    Company names are NOT stored in extractions.json — they live in the
    watch queue (preferred, more complete) and the run manifest (fallback).
    Returns empty dict if neither file exists; the LLM prompt will then
    fall back to using the ticker only.
    """
    lookup: dict = {}

    # Prefer watch_queue.json (richer source)
    if WATCH_QUEUE_PATH.exists():
        try:
            wq = json.loads(WATCH_QUEUE_PATH.read_text(encoding="utf-8"))
            for f in wq.get("filings", []):
                name = f.get("company_name", "").strip()
                if name:
                    lookup[f["ticker"]] = name
        except Exception:
            pass

    # Fill in any missing tickers from run_manifest. The manifest doesn't
    # currently store company_name, but we check defensively in case the
    # schema gets that field later.
    if RUN_MANIFEST_PATH.exists():
        try:
            rm = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
            for f in rm.get("filings", []):
                ticker = f.get("ticker")
                name = f.get("company_name", "").strip() if f.get("company_name") else ""
                if ticker and name and ticker not in lookup:
                    lookup[ticker] = name
        except Exception:
            pass

    return lookup


def _save_summary_cache(cache: dict) -> None:
    """Persist the summary cache. Called incrementally so partial progress survives."""
    SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARIES_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _cache_key(ticker: str, accession: str, ext: dict | None = None) -> str:
    """
    Cache key for a filing summary. Includes a short hash of the extraction
    content so that re-extracted filings automatically get a fresh summary
    rather than silently reusing a stale one. Falls back to ticker::accession
    alone if no extraction is provided.
    """
    import hashlib
    if ext is not None:
        data = {k: v for k, v in ext.items()
                if k not in ("ticker", "accession", "extraction_version",
                             "model", "generated_at")}
        content_hash = hashlib.md5(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{ticker}::{accession}::{content_hash}"
    return f"{ticker}::{accession}"


# ---------------------------------------------------------------------------
# Template-based fallback (used when --no-llm or LLM fails)
# ---------------------------------------------------------------------------

def _template_summary(ticker: str, ext: dict, ranking_entry: dict) -> str:
    """
    Build a serviceable summary from extracted fields without calling the
    LLM. Less polished than the LLM version but always works and is free.
    Useful as a fallback when the API is unavailable or as a deliberate
    cheap option via --no-llm.
    """
    op = ext.get("operational", {})
    cl = ext.get("competitive_landscape", {})
    g = ext.get("governance", {})
    fq = ext.get("financial_quality", {})
    ev = ext.get("events_and_catalysts", {})
    fl = ext.get("forward_looking", {})

    paragraphs: list[str] = []

    # Business paragraph
    biz_parts = []
    segs = [s.get("name", "") for s in op.get("segments", []) if s.get("name")]
    if segs:
        if len(segs) == 1:
            biz_parts.append(f"{ticker} operates as a single-segment business in {segs[0]}.")
        else:
            biz_parts.append(f"{ticker} operates {len(segs)} segments: {', '.join(segs[:4])}.")
    geos = [g_.get("region", "") for g_ in op.get("geographies", []) if g_.get("materiality") == "primary"]
    if geos:
        biz_parts.append(f"Primary geographic exposure is in {', '.join(geos[:3])}.")
    named = cl.get("named_competitors") or []
    if named:
        biz_parts.append(f"Named competitors include {', '.join(named[:5])}"
                         + (f" and {len(named)-5} others." if len(named) > 5 else "."))
    if biz_parts:
        paragraphs.append(" ".join(biz_parts))

    # Why-it-matters paragraph
    matters: list[str] = []
    if fq.get("going_concern_language"):
        matters.append("the auditor raised going-concern language")
    if fq.get("material_weakness_disclosed"):
        matters.append("a material weakness in internal controls was disclosed")
    if fq.get("restatement_mentioned"):
        matters.append("prior-period financials were restated")
    if g.get("ceo_changed"):
        matters.append("the CEO changed during the fiscal year")
    if g.get("cfo_changed"):
        matters.append("the CFO changed during the fiscal year")
    if g.get("auditor_changed"):
        matters.append("the auditor changed")

    big_imp = sorted(
        [i for i in ev.get("recent_impairments", []) if (i.get("amount_usd") or 0) >= 100_000_000],
        key=lambda i: i["amount_usd"], reverse=True,
    )
    if big_imp:
        b = big_imp[0]
        matters.append(f"a {_fmt_usd(b['amount_usd'])} impairment was recorded ({(b.get('description') or '')[:80]})")

    big_lit = sorted(
        [l for l in ev.get("material_litigation", []) if l.get("financial_exposure_usd")],
        key=lambda l: l["financial_exposure_usd"], reverse=True,
    )
    if big_lit:
        l = big_lit[0]
        matters.append(f"litigation with disclosed exposure of {_fmt_usd(l['financial_exposure_usd'])} is pending")

    sig_actions = [a for a in ev.get("regulatory_actions", []) if a.get("agency")]
    if sig_actions:
        matters.append(f"a regulatory matter is pending with {sig_actions[0]['agency']}")

    big_ma = sorted(
        [m for m in ev.get("ma_activity", []) if (m.get("amount_usd") or 0) >= 1_000_000_000],
        key=lambda m: m["amount_usd"], reverse=True,
    )
    if big_ma:
        m = big_ma[0]
        matters.append(f"a {_fmt_usd(m['amount_usd'])} {m.get('type', 'deal')} with "
                       f"{(m.get('counterparty') or 'undisclosed counterparty')[:60]} is {m.get('status', 'pending')}")

    if matters:
        paragraphs.append(
            "Notable in this filing: " + "; ".join(matters[:5]) + "."
        )

    # Forward paragraph
    fwd_parts = []
    capex = fl.get("capex_planned_usd")
    if capex:
        fwd_parts.append(f"Planned capex of {_fmt_usd(capex)} for the next fiscal year.")
    inits = fl.get("key_initiatives_named") or []
    if inits:
        fwd_parts.append(f"Named initiatives include {'; '.join(inits[:3])}.")
    if fwd_parts:
        paragraphs.append(" ".join(fwd_parts))

    return "\n\n".join(paragraphs) if paragraphs else "_(no extractable summary content)_"


# ---------------------------------------------------------------------------
# Per-filing renderer
# ---------------------------------------------------------------------------

def _render_filing(filing_score: dict, ext: dict, summary: str) -> str:
    """
    Render one filing's full markdown entry: header + SEC link + triggers + summary.
    """
    ticker = filing_score["ticker"]
    triggers = filing_score.get("triggers", [])
    accession = filing_score.get("accession", "")
    cik = filing_score.get("cik", "")
    primary_doc = filing_score.get("primary_document", "")
    cp_score = filing_score.get("chokepoint_score", 0)

    header = f"### {ticker}"

    # Subheader: clickable SEC link to the actual filing on EDGAR. Build the
    # URL only if we have all three pieces; otherwise fall back to showing
    # the accession number alone.
    if cik and accession and primary_doc:
        # SEC's EDGAR archive URL format: /Archives/edgar/data/{cik_int}/{accession_no_dashes}/{primary_doc}
        cik_int = str(int(cik)) if cik.lstrip("0") else cik
        accession_no_dashes = accession.replace("-", "")
        sec_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_dashes}/{primary_doc}"
        subheader = f"_[View on SEC]({sec_url})_"
    else:
        subheader = f"_Accession {accession or 'unknown'}_"

    # Triggers line — shows what the bot's rules flagged for this filing.
    # The LLM-written prose below covers the same content in narrative form,
    # but the triggers line gives a quick scan of "why is this here".
    if triggers:
        triggers_line = f"_Triggers: {'; '.join(triggers)}._"
    else:
        triggers_line = "_Triggers: none._"

    # Chokepoint connectivity note — only shown when nonzero.
    cp_line = (f"_Chokepoints: {cp_score} cross-corpus connection{'s' if cp_score != 1 else ''} detected._"
               if cp_score > 0 else "")

    parts = [header, subheader, "", triggers_line]
    if cp_line:
        parts.append(cp_line)
    parts += ["", summary, ""]

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Chokepoints integration
# ---------------------------------------------------------------------------

def _build_chokepoints_section(extractions: dict) -> str:
    """
    Attempt to compute and format the chokepoints section. Returns an empty
    string if chokepoints.py is unavailable or computation fails — the rest
    of the digest still renders normally.
    """
    try:
        from chokepoints import compute_chokepoints, format_chokepoints_section
    except ImportError:
        return ""

    try:
        cp = compute_chokepoints(extractions)
        total = sum(len(v) for v in cp.values())
        print(f"[digest] Chokepoints computed: {total} entities "
              f"({len(cp['vendors'])} vendor · "
              f"{len(cp['customers'])} customer · "
              f"{len(cp['competitors'])} competitor)")
        return format_chokepoints_section(cp)
    except Exception as e:
        print(f"[digest] Chokepoint section failed (non-fatal): {e}")
        return ""


# ---------------------------------------------------------------------------
# Main digest assembly
# ---------------------------------------------------------------------------

def build_digest(extractions: dict, ranking: dict, *,
                 use_llm: bool = True,
                 refresh_summaries: bool = False) -> str:
    """
    Assemble the markdown digest. Looks up each filing's summary from cache,
    generates a new one via the LLM when missing or refresh_summaries=True,
    and falls back to template-based prose if use_llm=False or the LLM call
    fails.
    """
    summary_cache = {} if refresh_summaries else _load_summary_cache()
    company_names = _load_company_names()

    client = None
    if use_llm:
        try:
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("[digest] No ANTHROPIC_API_KEY found — falling back to template summaries.")
                use_llm = False
            else:
                client = Anthropic(api_key=api_key)
        except ImportError:
            print("[digest] anthropic SDK not installed — falling back to template summaries.")
            use_llm = False

    summary = ranking.get("summary", {})
    n_total = ranking.get("filing_count", 0)
    filings = ranking.get("filings", [])

    # Title block — kept minimal. The tier counts give a quick at-a-glance
    # for the run; everything else (timestamp, model, rules version) is
    # noise that bloats the file without serving the reader.
    sort_mode = ranking.get("sort_mode", "ctmg")
    lines = [
        "# SEC Report",
        "",
        f"{n_total} filings reviewed.",
        "",
        "---",
        "",
    ]

    # Chokepoints section — injected between the title block and the filing
    # entries so the reader gets the aggregate cross-corpus view before
    # diving into individual filings.
    chokepoints_md = _build_chokepoints_section(extractions)
    if chokepoints_md:
        lines.append(chokepoints_md)

    # Track LLM cost and cache hits so the user knows what they paid for
    cache_hits = 0
    llm_calls = 0
    llm_failures = 0
    template_fallbacks = 0

    for i, fs in enumerate(filings, 1):
        ticker = fs["ticker"]
        accession = fs.get("accession", "")
        ext = extractions.get(ticker)
        if ext is None:
            lines.append(f"### {ticker}")
            lines.append("_(extraction missing — re-run extract.py)_")
            lines.append("")
            continue

        key = _cache_key(ticker, accession, ext)

        # Resolve a summary for this filing in priority order:
        #   1. Cache hit (free, instant)
        #   2. LLM call (costs money, slow)
        #   3. Template fallback (free, instant, less polished)
        if key in summary_cache:
            summary_text = summary_cache[key]
            cache_hits += 1
        elif use_llm and client is not None:
            try:
                print(f"  [{i:>3}/{len(filings)}] {ticker:<6} generating summary...", flush=True)
                summary_text = _call_summary_llm(client, ticker, ext, fs,
                                                 company_names.get(ticker, ""))
                summary_cache[key] = summary_text
                # Save cache incrementally so we don't lose progress on crash
                _save_summary_cache(summary_cache)
                llm_calls += 1
            except Exception as e:
                print(f"  [{i:>3}/{len(filings)}] {ticker:<6} LLM failed: {e} — using template", flush=True)
                summary_text = _template_summary(ticker, ext, fs)
                llm_failures += 1
        else:
            summary_text = _template_summary(ticker, ext, fs)
            template_fallbacks += 1

        lines.append(_render_filing(fs, ext, summary_text))

    # Print run stats so the user knows what just happened
    print()
    print(f"[digest] Cache hits: {cache_hits}")
    print(f"[digest] New LLM summaries: {llm_calls}")
    if llm_failures:
        print(f"[digest] LLM failures (template fallback): {llm_failures}")
    if template_fallbacks:
        print(f"[digest] Template-only summaries: {template_fallbacks}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Produce a markdown digest of ranked filings, with LLM-written summaries.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_DIGEST_PATH,
                        help=f"Output path (default: {DEFAULT_DIGEST_PATH})")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM summaries entirely; use template-based fallback for all filings.")
    parser.add_argument("--refresh-summaries", action="store_true",
                        help="Force regenerate all LLM summaries (ignore cache).")
    args = parser.parse_args()

    if not EXTRACTIONS_PATH.exists():
        raise SystemExit(f"{EXTRACTIONS_PATH} not found. Run extract.py first.")
    if not RANKING_PATH.exists():
        raise SystemExit(f"{RANKING_PATH} not found. Run rank.py first.")

    extractions_payload = json.loads(EXTRACTIONS_PATH.read_text(encoding="utf-8"))
    extractions = extractions_payload.get("extractions", {})
    ranking = json.loads(RANKING_PATH.read_text(encoding="utf-8"))

    digest = build_digest(
        extractions, ranking,
        use_llm=not args.no_llm,
        refresh_summaries=args.refresh_summaries,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(digest, encoding="utf-8")

    n_filings = ranking.get("filing_count", 0)
    size_kb = args.out.stat().st_size / 1024
    print(f"[digest] Wrote {n_filings} filings to {args.out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()