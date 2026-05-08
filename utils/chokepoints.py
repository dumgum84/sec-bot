"""
chokepoints.py — Cross-filing entity aggregation
=================================================

Reads extractions.json and identifies named entities (companies) that appear
across multiple filings as competitors, vendors, or customers. The signal
lives in entities mentioned by 3+ filers — frequent enough to be a real
structural pattern, rare enough to be non-obvious.

Zero LLM calls. Pure Python aggregation over data already in extractions.json.
Free per run, gets richer with every monthly cycle as the corpus grows.

Exposes:
    compute_chokepoints(extractions, min_mentions) -> dict
        Returns {"vendors": [...], "customers": [...], "competitors": [...]}
        Each entry: {"name": str, "count": int, "tickers": [str, ...]}

    compute_chokepoint_scores(extractions, chokepoints) -> dict[str, int]
        Returns {ticker: score} where score = count of chokepoint entities
        the filing mentions across all three categories. Used by rank.py as a
        within-tier tiebreaker.

    format_chokepoints_section(chokepoints, min_mentions, ticker_tiers) -> str
        Renders the chokepoints data as a markdown section with GFM-compatible
        heading anchor links, for embedding in sec_report.md by digest.py.

Usage (standalone):
    python chokepoints.py              # default min=3, no upper cap
    python chokepoints.py --min 2      # catch rarer patterns
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from fetch import DATA_DIR

EXTRACTIONS_PATH = DATA_DIR / "extractions.json"

# ---------------------------------------------------------------------------
# Mention threshold
# ---------------------------------------------------------------------------
# Entities mentioned by fewer than MIN_MENTIONS filers: noise, too sparse.
# There is no upper cap — more mentions = stronger signal, not weaker.
# Universal infrastructure noise (Amazon, Google, Microsoft) is handled by
# _DENYLIST below rather than an arbitrary ceiling.

DEFAULT_MIN_MENTIONS = 3

# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

# Corporate suffixes that don't distinguish entities — strip before comparing.
_SUFFIX_RE = re.compile(
    r'\b(incorporated|corporation|corp|inc|llc|l\.l\.c|limited|ltd|company|companies|'
    r'co|plc|holdings|holding|group|international|industries|industry|'
    r'enterprises|solutions|technologies|technology|services|systems)\b\.?',
    re.IGNORECASE,
)

# Alias map: normalized_name -> canonical_normalized_name.
# Handles common corporate identity splits (subsidiary names, rebrandings).
# Add entries here whenever you see fragmented counts in output.
_ALIASES: dict[str, str] = {
    # Amazon variants
    "aws": "amazon",
    "amazon web services": "amazon",
    "amazon.com": "amazon",
    "amazon com": "amazon",
    # Google / Alphabet
    "alphabet": "google",
    "alphabet inc": "google",
    "google llc": "google",
    "google inc": "google",
    # Meta variants
    "meta platforms": "meta",
    "facebook": "meta",
    "instagram": "meta",
    "whatsapp": "meta",
    # Microsoft
    "microsoft corporation": "microsoft",
    "microsoft corp": "microsoft",
    # Apple
    "apple inc": "apple",
    "apple computer": "apple",
    # TSMC
    "tsmc": "taiwan semiconductor",
    "taiwan semiconductor manufacturing": "taiwan semiconductor",
    "taiwan semiconductor manufacturing company": "taiwan semiconductor",
    # Other common splits
    "jp morgan": "jpmorgan",
    "j.p. morgan": "jpmorgan",
    "jpmorgan chase": "jpmorgan",
    "bank of america merrill lynch": "bank of america",
    "bofa": "bank of america",
    "wells fargo bank": "wells fargo",
    "fedex corporation": "fedex",
    "united parcel service": "ups",
}

# Entities so universally present across all filings that their mention count
# is not informative signal. Keep this list short — over-filtering loses real
# chokepoints. Expand here only when a name dominates output without adding
# research value.
_DENYLIST: set[str] = {
    "microsoft",   # universal SaaS/cloud — present across essentially all tech-adjacent filers
    "google",      # universal advertising/cloud infra
    "amazon",      # cloud (AWS) + logistics + marketplace all roll up; too broad to be signal
}


def _normalize(raw: str) -> str:
    """
    Normalize an entity name for comparison purposes.
    Pipeline: lowercase → strip corporate suffixes → collapse whitespace → alias map.
    """
    n = raw.lower().strip()
    # Remove trailing punctuation
    n = re.sub(r'[.,;:!?\'"]+$', '', n).strip()
    # Strip corporate suffixes (replace with space, then clean up)
    n = _SUFFIX_RE.sub(' ', n).strip()
    # Collapse any internal whitespace introduced by the substitution
    n = re.sub(r'\s+', ' ', n).strip()
    # Resolve aliases
    n = _ALIASES.get(n, n)
    return n


# ---------------------------------------------------------------------------
# Entity extraction from concentration description strings
# ---------------------------------------------------------------------------

# Regex: captures a leading proper-noun sequence from a description.
# Handles "of", "and", "&" as joining words inside entity names.
# Examples:
#   "Foxconn manufactures 90% of our products"  → "Foxconn"
#   "Taiwan Semiconductor Manufacturing Company" → "Taiwan Semiconductor Manufacturing Company"
#   "Bank of America provides our revolving..."  → "Bank of America"
#   "Apple Inc. represents 22% of revenue"       → "Apple Inc."
#   "The Home Depot accounted for..."            → "The Home Depot"
_LEADING_ENTITY_RE = re.compile(
    r'^(?:The\s+)?'
    r'([A-Z][A-Za-z&.\'-]+'
    r'(?:[ \t]+(?:[A-Z][A-Za-z&.\'-]+|[Oo]f|[Aa]nd|&|[Tt]he))*)',
)

# Phrases (and single words) that start with a capital letter but aren't entity names.
# Single generic words like "Top" are caught here — they appear in phrases like
# "Top customers represent..." and would otherwise register as entity names.
_GENERIC_PHRASES: set[str] = {
    "no single", "one customer", "one vendor", "one supplier", "a single",
    "certain customers", "several customers", "various customers",
    "the company", "our company", "we do not", "we have",
    # Single-word false positives from superlative/ordinal/generic language
    "top", "first", "second", "third", "largest", "major", "primary",
    "significant", "key", "certain", "various", "several", "no",
    # Numbers spelled out ("Five customers accounted for..." etc.)
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    # Articles, prepositions, and generic sentence starters
    "the", "a", "an", "for", "this", "that", "our", "its",
    # Other common false positives
    "sales", "during", "customer", "approximately", "other",
}


def _extract_entity_from_description(desc: str) -> Optional[str]:
    """
    Extract a company name from a concentration description string.
    Returns None if no plausible entity name is found.
    """
    if not desc:
        return None
    desc = desc.strip()
    m = _LEADING_ENTITY_RE.match(desc)
    if not m:
        return None
    name = m.group(1).strip()
    # Discard generic phrases and single generic words
    if name.lower() in _GENERIC_PHRASES:
        return None
    # Discard names shorter than 3 characters (abbreviations, noise)
    if len(name) < 3:
        return None
    # Discard if digits crept into the "name" (percentage statements, etc.)
    if re.search(r'\d', name):
        return None
    return name


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def _collect_entity_mentions(
    extractions: dict,
    field_getter,
) -> tuple[Counter, dict, dict]:
    """
    Walk all filings, apply field_getter to get a list of raw entity name
    strings per filing, normalize, and count.

    Returns:
        counts   — Counter {normalized_name: mention_count (one per filing)}
        tickers  — dict {normalized_name: [ticker, ...]} ordered by first seen
        raw_names — dict {normalized_name: Counter of raw name variants}
                    Used to pick the best display name.

    A filing only contributes +1 per entity regardless of how many times the
    entity appears within that filing — we're counting filer breadth, not
    repetition within a single document.
    """
    counts: Counter = Counter()
    tickers: dict[str, list[str]] = defaultdict(list)
    raw_names: dict[str, Counter] = defaultdict(Counter)

    for ticker, ext in extractions.items():
        # Collect all normalized names this filing mentions, deduplicated.
        names_this_filing: set[str] = set()
        raw_this_filing: dict[str, str] = {}  # norm → best raw form seen here

        for raw_name in field_getter(ext):
            if not raw_name:
                continue
            norm = _normalize(raw_name)
            if not norm or len(norm) < 3 or norm in _DENYLIST:
                continue
            names_this_filing.add(norm)
            # Track the raw variant for display-name selection
            if norm not in raw_this_filing:
                raw_this_filing[norm] = raw_name.strip()
            raw_names[norm][raw_name.strip()] += 1

        for norm in names_this_filing:
            counts[norm] += 1
            tickers[norm].append(ticker)

    return counts, tickers, raw_names


def _best_display_name(norm: str, raw_counter: Counter) -> str:
    """
    Pick the best display name for a normalized entity.
    Uses the most commonly seen raw variant, falling back to title-casing the
    normalized form if no raw variants were collected.
    """
    if raw_counter:
        return raw_counter.most_common(1)[0][0]
    return norm.title()


def compute_chokepoints(
    extractions: dict,
    min_mentions: int = DEFAULT_MIN_MENTIONS,
) -> dict[str, list[dict]]:
    """
    Compute chokepoints across all filings in the extractions dict.

    Returns a dict with three keys, each a list of chokepoint entries sorted
    by mention count descending:

        {
          "vendors":     [{"name": str, "count": int, "tickers": [str, ...]}, ...],
          "customers":   [...],
          "competitors": [...],
        }

    Only entities with count >= min_mentions are included. There is no upper
    cap — more mentions means stronger signal, not weaker. Universal noise
    (Amazon, Google, Microsoft) is handled by _DENYLIST.

    Entities in the denylist and those that fail name extraction are excluded.
    """

    def _get_vendor_names(ext: dict) -> list[str]:
        names = []
        for vc in ext.get("operational", {}).get("vendor_concentrations", []):
            desc = vc.get("description", "") if isinstance(vc, dict) else str(vc)
            entity = _extract_entity_from_description(desc)
            if entity:
                names.append(entity)
        return names

    def _get_customer_names(ext: dict) -> list[str]:
        names = []
        for cc in ext.get("operational", {}).get("customer_concentrations", []):
            desc = cc.get("description", "") if isinstance(cc, dict) else str(cc)
            entity = _extract_entity_from_description(desc)
            if entity:
                names.append(entity)
        return names

    def _get_competitor_names(ext: dict) -> list[str]:
        named = ext.get("competitive_landscape", {}).get("named_competitors", [])
        return [n for n in named if isinstance(n, str) and n.strip()]

    result: dict[str, list[dict]] = {}

    for category, getter in [
        ("vendors", _get_vendor_names),
        ("customers", _get_customer_names),
        ("competitors", _get_competitor_names),
    ]:
        counts, tickers, raw_names = _collect_entity_mentions(extractions, getter)
        entries: list[dict] = []
        for norm, count in counts.most_common():
            if count >= min_mentions:
                entries.append({
                    "name": _best_display_name(norm, raw_names[norm]),
                    "count": count,
                    "tickers": tickers[norm],
                })
        result[category] = entries

    return result


def compute_chokepoint_scores(
    extractions: dict,
    chokepoints: dict,
) -> dict[str, int]:
    """
    Compute a per-filing chokepoint connectivity score.

    Score = number of distinct chokepoint entities (from any category) that
    this filing mentions. Used by rank.py as a within-tier tiebreaker:
    filings connected to more cross-corpus structural patterns surface first
    within their tier.

    Each entity counts at most once per filing regardless of how many
    categories it appears in (e.g., a company named as both a vendor and a
    competitor only contributes 1 to the score).
    """
    # Build lookup: normalized name → is a chokepoint?
    chokepoint_norms: set[str] = set()
    for category in ("vendors", "customers", "competitors"):
        for entry in chokepoints.get(category, []):
            chokepoint_norms.add(_normalize(entry["name"]))

    scores: dict[str, int] = {}

    for ticker, ext in extractions.items():
        score = 0
        seen: set[str] = set()

        # Check vendor mentions
        for vc in ext.get("operational", {}).get("vendor_concentrations", []):
            desc = vc.get("description", "") if isinstance(vc, dict) else str(vc)
            entity = _extract_entity_from_description(desc)
            if entity:
                norm = _normalize(entity)
                if norm in chokepoint_norms and norm not in seen:
                    score += 1
                    seen.add(norm)

        # Check customer mentions
        for cc in ext.get("operational", {}).get("customer_concentrations", []):
            desc = cc.get("description", "") if isinstance(cc, dict) else str(cc)
            entity = _extract_entity_from_description(desc)
            if entity:
                norm = _normalize(entity)
                if norm in chokepoint_norms and norm not in seen:
                    score += 1
                    seen.add(norm)

        # Check competitor mentions
        for name in ext.get("competitive_landscape", {}).get("named_competitors", []):
            if isinstance(name, str):
                norm = _normalize(name)
                if norm in chokepoint_norms and norm not in seen:
                    score += 1
                    seen.add(norm)

        scores[ticker] = score

    return scores


# ---------------------------------------------------------------------------
# Markdown formatters
# ---------------------------------------------------------------------------

def format_chokepoints_section(
    chokepoints: dict,
    min_mentions: int = DEFAULT_MIN_MENTIONS,
) -> str:
    """
    Render the chokepoints data as a markdown section.
    Uses plain text tickers so the output is PDF-friendly.
    Format: - Entity Name — N filers: TICK · TICK · TICK
    """
    lines = [
        "## Chokepoints across this run",
        "",
        f"_Entities named in {min_mentions}+ filings across vendor, customer, and "
        f"competitor disclosures. Structural patterns that only appear when filings "
        f"are read in aggregate._",
        "",
    ]

    LABELS: dict[str, str] = {
        "vendors": "Vendors named by multiple filers",
        "customers": "Customers named by multiple filers",
        "competitors": "Competitors named by multiple filers",
    }

    any_results = False
    for category in ("vendors", "customers", "competitors"):
        entries = chokepoints.get(category, [])
        if not entries:
            continue
        any_results = True
        lines.append(f"**{LABELS[category]}:**")
        for e in entries:
            ticker_list = " · ".join(e["tickers"])
            lines.append(f"- {e['name']} — {e['count']} filers: {ticker_list}")
        lines.append("")

    if not any_results:
        lines.append(
            "_No chokepoints found in the current corpus. "
            "Try a larger run or lower the --min threshold._"
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _format_chokepoints_report(
    chokepoints: dict,
    min_mentions: int,
) -> str:
    """Render chokepoints as plain terminal output (no anchor links)."""
    lines = [
        "## Chokepoints across this run",
        "",
        f"Minimum mentions: {min_mentions}",
        "",
    ]

    LABELS: dict[str, str] = {
        "vendors": "VENDORS named by multiple filers",
        "customers": "CUSTOMERS named by multiple filers",
        "competitors": "COMPETITORS named by multiple filers",
    }

    any_results = False
    for category in ("vendors", "customers", "competitors"):
        entries = chokepoints.get(category, [])
        if not entries:
            continue
        any_results = True
        lines.append(LABELS[category])
        for e in entries:
            ticker_list = ", ".join(e["tickers"])
            lines.append(f"  {e['name']:<40} {e['count']:>3} filers  ({ticker_list})")
        lines.append("")

    if not any_results:
        lines.append(
            "No chokepoints found. Try --min 2 or run with a larger corpus."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute cross-filing chokepoints from extractions.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python chokepoints.py              # default min=3\n"
            "  python chokepoints.py --min 2      # catch rarer patterns\n"
        ),
    )
    parser.add_argument(
        "--min", type=int, default=DEFAULT_MIN_MENTIONS,
        help=f"Minimum filer mention count (default: {DEFAULT_MIN_MENTIONS})",
    )
    args = parser.parse_args()

    if not EXTRACTIONS_PATH.exists():
        raise SystemExit(f"{EXTRACTIONS_PATH} not found. Run extract.py first.")

    payload = json.loads(EXTRACTIONS_PATH.read_text(encoding="utf-8"))
    extractions = payload.get("extractions", {})
    if not extractions:
        raise SystemExit("No extractions found in extractions.json.")

    print(f"[chokepoints] Scanning {len(extractions)} filings for cross-corpus entity patterns...")
    print(f"[chokepoints] Minimum mentions: {args.min} (no upper cap)")
    print()

    chokepoints = compute_chokepoints(extractions, min_mentions=args.min)

    n_vendor = len(chokepoints["vendors"])
    n_customer = len(chokepoints["customers"])
    n_competitor = len(chokepoints["competitors"])
    total = n_vendor + n_customer + n_competitor

    print(f"[chokepoints] Found {total} chokepoints  "
          f"({n_vendor} vendor · {n_customer} customer · {n_competitor} competitor)")
    print()

    print(_format_chokepoints_report(chokepoints, args.min))

    # Show per-filing connectivity scores (top 20 nonzero)
    scores = compute_chokepoint_scores(extractions, chokepoints)
    nonzero = {t: s for t, s in scores.items() if s > 0}
    if nonzero:
        print(f"[chokepoints] Chokepoint connectivity (top {min(20, len(nonzero))} filers):")
        for ticker, score in sorted(nonzero.items(), key=lambda x: -x[1])[:20]:
            bar = "█" * score
            print(f"  {ticker:<6}  {score:>2}  {bar}")


if __name__ == "__main__":
    main()