"""
rank.py — Phase 7: tier-based filing ranking
==============================================

Reads extractions.json (the consolidated Phase 4 output) and run_manifest.json
(for the CIK and primary_document needed to build SEC links downstream),
evaluates a small set of trigger conditions on each filing, and groups
results into three tiers: critical, notable, quiet.

Output:
    output/ranking.json   — full ranking with tier + triggers per filing
    stdout                — tier breakdown, filtered by --tier if passed

Usage:
    python rank.py                            # show all filings, all tiers
    python rank.py --tier critical            # show only critical filings
    python rank.py --tier critical,notable    # show critical and notable

The ranking.json file ALWAYS contains all filings. The --tier flag only
controls what gets printed to your terminal.

Why tiers and not numerical scores: any weighted score encodes arbitrary
judgments ("a CFO change is worth 1.5x a CEO change"). Tiers admit the
imprecision by being categorical. A filing is critical, notable, or quiet —
the bot doesn't pretend to rank within a category beyond a simple "more
triggers fired" tiebreaker. Use the digest's prose summaries for the
qualitative differences within a tier.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from fetch import OUTPUT_DIR

EXTRACTIONS_PATH = OUTPUT_DIR / "extractions.json"
RUN_MANIFEST_PATH = OUTPUT_DIR / "run_manifest.json"
RANKING_PATH = OUTPUT_DIR / "ranking.json"
RULES_VERSION = "v1-tiers"

VALID_TIERS = ("critical", "notable", "quiet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_recent_year(year_str: Optional[str]) -> bool:
    """
    Heuristic: 'recent' means fiscal year 2024+ for filings dated 2025-2026.
    Filters out historical M&A entries (e.g., a 2009 joint venture still
    listed in the current 10-K for context).
    """
    if not year_str:
        return False
    s = str(year_str).lower()
    return any(yr in s for yr in ("2024", "2025", "2026"))


def _ma_counts(ma: dict) -> bool:
    """An M&A entry counts if recent OR pending/announced (forward-looking)."""
    status = (ma.get("status") or "").lower()
    if status in ("pending", "announced"):
        return True
    return _is_recent_year(ma.get("year"))


def _fmt_usd(amount: float | int | None) -> str:
    """Format a USD amount in human-friendly units."""
    if not amount:
        return "?"
    if amount >= 1e9:
        return f"${amount/1e9:.1f}B"
    if amount >= 1e6:
        return f"${amount/1e6:.0f}M"
    return f"${amount:.0f}"


# ---------------------------------------------------------------------------
# Trigger detectors — each returns either None (no trigger) or a tuple
# (trigger_string, severity) where severity is "critical" or "notable".
#
# Severity determines tier placement: any critical-severity trigger puts a
# filing in the critical tier. Otherwise, any notable-severity trigger
# puts it in the notable tier. No triggers means quiet.
# ---------------------------------------------------------------------------

def _trigger_going_concern(ext: dict) -> Optional[tuple[str, str]]:
    if ext.get("financial_quality", {}).get("going_concern_language"):
        return ("going concern language present", "critical")
    return None


def _trigger_material_weakness(ext: dict) -> Optional[tuple[str, str]]:
    if ext.get("financial_quality", {}).get("material_weakness_disclosed"):
        return ("material weakness in internal controls disclosed", "critical")
    return None


def _trigger_restatement(ext: dict) -> Optional[tuple[str, str]]:
    if ext.get("financial_quality", {}).get("restatement_mentioned"):
        return ("restatement of prior-period financials mentioned", "critical")
    return None


def _trigger_major_impairment(ext: dict) -> Optional[tuple[str, str]]:
    """
    Any impairment $500M+ → critical. $100M-$500M → notable. Below $100M
    is treated as noise (the LLM occasionally pulls these despite the
    schema; easier to filter here than perfectly tune the prompt).
    """
    imps = ext.get("events_and_catalysts", {}).get("recent_impairments", [])
    if not imps:
        return None
    biggest = max((i.get("amount_usd") or 0) for i in imps)
    if biggest >= 500_000_000:
        return (f"{_fmt_usd(biggest)}+ impairment disclosed", "critical")
    if biggest >= 100_000_000:
        return (f"{_fmt_usd(biggest)} impairment disclosed", "notable")
    return None


def _trigger_major_litigation(ext: dict) -> Optional[tuple[str, str]]:
    """
    Litigation with disclosed exposure $100M+ → critical (real money at stake).
    Otherwise just count of pending matters → notable.
    """
    lit = ext.get("events_and_catalysts", {}).get("material_litigation", [])
    if not lit:
        return None
    big = [l for l in lit if (l.get("financial_exposure_usd") or 0) >= 100_000_000]
    if big:
        max_exp = max(l["financial_exposure_usd"] for l in big)
        return (f"litigation with disclosed exposure {_fmt_usd(max_exp)}+", "critical")
    return (f"{len(lit)} pending litigation matter(s) disclosed", "notable")


def _trigger_significant_regulatory(ext: dict) -> Optional[tuple[str, str]]:
    """
    Action by DOJ / FDA / FTC / FERC / SEC / EPA / Department of Commerce
    → critical. Action by any other agency → notable.
    """
    actions = ext.get("events_and_catalysts", {}).get("regulatory_actions", [])
    if not actions:
        return None
    SIGNIFICANT = (
        "doj", "department of justice",
        "fda", "food and drug",
        "ftc", "federal trade commission",
        "ferc",
        "sec", "securities and exchange",
        "epa", "environmental protection",
        "department of commerce",
    )
    sig_agencies = []
    for a in actions:
        agency = (a.get("agency") or "").lower()
        if any(s in agency for s in SIGNIFICANT):
            sig_agencies.append(a.get("agency"))
    if sig_agencies:
        named = list({a for a in sig_agencies if a})[:3]
        return (f"regulatory action: {', '.join(named)}", "critical")
    return (f"{len(actions)} regulatory action(s) disclosed", "notable")


def _trigger_cfo_change(ext: dict) -> Optional[tuple[str, str]]:
    if ext.get("governance", {}).get("cfo_changed"):
        return ("CFO changed during fiscal year", "notable")
    return None


def _trigger_ceo_change(ext: dict) -> Optional[tuple[str, str]]:
    if ext.get("governance", {}).get("ceo_changed"):
        return ("CEO changed during fiscal year", "notable")
    return None


def _trigger_auditor_change(ext: dict) -> Optional[tuple[str, str]]:
    if ext.get("governance", {}).get("auditor_changed"):
        return ("auditor changed during fiscal year", "notable")
    return None


def _trigger_major_ma(ext: dict) -> Optional[tuple[str, str]]:
    """M&A activity worth $1B+, recent or pending."""
    ma_list = ext.get("events_and_catalysts", {}).get("ma_activity", [])
    qualifying = [
        m for m in ma_list
        if _ma_counts(m) and (m.get("amount_usd") or 0) >= 1_000_000_000
    ]
    if not qualifying:
        return None
    biggest = max(qualifying, key=lambda m: m["amount_usd"])
    deal_type = biggest.get("type") or "deal"
    return (f"{deal_type} {_fmt_usd(biggest['amount_usd'])} disclosed", "notable")


def _trigger_large_capex(ext: dict) -> Optional[tuple[str, str]]:
    """Capex >$1B planned for next fiscal year."""
    capex = ext.get("forward_looking", {}).get("capex_planned_usd")
    if capex and capex >= 1_000_000_000:
        return (f"planned capex {_fmt_usd(capex)} next fiscal year", "notable")
    return None


# Order matters for triggers list display — most-severe signals first.
TRIGGER_DETECTORS = [
    _trigger_going_concern,
    _trigger_material_weakness,
    _trigger_restatement,
    _trigger_major_impairment,
    _trigger_major_litigation,
    _trigger_significant_regulatory,
    _trigger_cfo_change,
    _trigger_ceo_change,
    _trigger_auditor_change,
    _trigger_major_ma,
    _trigger_large_capex,
]


# ---------------------------------------------------------------------------
# Per-filing tier evaluation
# ---------------------------------------------------------------------------

@dataclass
class FilingResult:
    ticker: str
    accession: str
    cik: str
    primary_document: str
    tier: str                          # "critical" | "notable" | "quiet"
    triggers: list[str] = field(default_factory=list)


def _evaluate_filing(ticker: str, ext: dict, manifest_lookup: dict) -> FilingResult:
    """Run all trigger detectors and assign a tier."""
    triggers: list[str] = []
    has_critical = False
    has_notable = False

    for detector in TRIGGER_DETECTORS:
        result = detector(ext)
        if result is None:
            continue
        msg, severity = result
        triggers.append(msg)
        if severity == "critical":
            has_critical = True
        elif severity == "notable":
            has_notable = True

    if has_critical:
        tier = "critical"
    elif has_notable:
        tier = "notable"
    else:
        tier = "quiet"

    # Pull CIK and primary_document from the run manifest so digest.py can
    # build SEC links without re-reading run_manifest itself.
    accession = ext.get("accession", "")
    manifest_entry = manifest_lookup.get(ticker, {})
    cik = manifest_entry.get("cik", "")
    # The manifest doesn't store primary_document directly. Fall back to
    # an empty string; digest.py handles the missing-document case.
    primary_doc = manifest_entry.get("primary_document", "")

    return FilingResult(
        ticker=ticker,
        accession=accession,
        cik=cik,
        primary_document=primary_doc,
        tier=tier,
        triggers=triggers,
    )


# ---------------------------------------------------------------------------
# Manifest lookup
# ---------------------------------------------------------------------------

def _load_manifest_lookup() -> dict:
    """
    Build {ticker: {cik, primary_document}} from run_manifest.json.

    primary_document isn't actually in the run_manifest currently — we read
    watch_queue.json instead, which does have it. If watch_queue is missing
    we fall back to whatever's in run_manifest, which still gives us CIK.
    """
    lookup: dict = {}

    # Prefer watch_queue.json (has primary_document)
    watch_path = OUTPUT_DIR / "watch_queue.json"
    if watch_path.exists():
        try:
            wq = json.loads(watch_path.read_text(encoding="utf-8"))
            for f in wq.get("filings", []):
                lookup[f["ticker"]] = {
                    "cik": f.get("cik", ""),
                    "primary_document": f.get("primary_document", ""),
                }
        except Exception:
            pass

    # Fill in missing tickers from run_manifest (CIK only)
    if RUN_MANIFEST_PATH.exists():
        try:
            rm = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
            for f in rm.get("filings", []):
                if f["ticker"] not in lookup:
                    lookup[f["ticker"]] = {
                        "cik": f.get("cik", ""),
                        "primary_document": "",
                    }
        except Exception:
            pass

    return lookup


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

TIER_ORDER = {"critical": 0, "notable": 1, "quiet": 2}


def _sort_key(r: FilingResult):
    """
    Sort: tier first (critical → notable → quiet), then trigger count
    descending (more signals = read first), then alphabetical by ticker.
    """
    return (TIER_ORDER[r.tier], -len(r.triggers), r.ticker)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_tier_flag(value: str) -> set[str]:
    """
    Parse the --tier argument into a set of tier names.

    Accepts comma-separated values: "critical", "critical,notable", etc.
    Validates every entry. Empty string → all tiers (default behavior).
    """
    if not value:
        return set(VALID_TIERS)
    requested = {t.strip().lower() for t in value.split(",") if t.strip()}
    invalid = requested - set(VALID_TIERS)
    if invalid:
        raise SystemExit(
            f"[rank] Invalid tier(s): {', '.join(sorted(invalid))}. "
            f"Valid options: {', '.join(VALID_TIERS)}"
        )
    return requested


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def _print_tier(tier_name: str, results: list[FilingResult]) -> None:
    """Print one tier's filings to stdout."""
    label = tier_name.upper()
    print("=" * 78)
    print(f"  {label}  ({len(results)} filing{'s' if len(results) != 1 else ''})")
    print("=" * 78)
    print()

    if not results:
        print("  (no filings in this tier)")
        print()
        return

    # Quiet tier gets a one-line list — there's nothing to say about each.
    # Critical and notable get full breakdowns since the triggers are the
    # whole point.
    if tier_name == "quiet":
        tickers = ", ".join(r.ticker for r in results)
        print(f"  {tickers}")
        print()
        return

    for i, r in enumerate(results, 1):
        n = len(r.triggers)
        print(f"  {i:>3}. {r.ticker:<6}  {n} trigger{'s' if n != 1 else ''}")
        for t in r.triggers:
            print(f"        - {t}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rank filings into critical / notable / quiet tiers using deterministic rules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python rank.py                          # show all tiers\n"
            "  python rank.py --tier critical          # only critical\n"
            "  python rank.py --tier critical,notable  # critical + notable\n"
            "\n"
            "ranking.json always contains all filings regardless of --tier."
        ),
    )
    parser.add_argument(
        "--tier", default="",
        help=f"Comma-separated tier filter for terminal output. "
             f"Options: {','.join(VALID_TIERS)}. "
             f"Default: all three tiers.",
    )
    args = parser.parse_args()

    requested_tiers = _parse_tier_flag(args.tier)

    if not EXTRACTIONS_PATH.exists():
        raise SystemExit(f"{EXTRACTIONS_PATH} not found. Run extract.py first.")

    payload = json.loads(EXTRACTIONS_PATH.read_text(encoding="utf-8"))
    extractions = payload.get("extractions", {})
    if not extractions:
        raise SystemExit(f"No extractions in {EXTRACTIONS_PATH}.")

    manifest_lookup = _load_manifest_lookup()

    print(f"[rank] Evaluating {len(extractions)} filings using rules version {RULES_VERSION}")
    print()

    results = [_evaluate_filing(t, ext, manifest_lookup) for t, ext in extractions.items()]
    results.sort(key=_sort_key)

    # Persist full ranking to disk regardless of CLI filtering. This is the
    # contract: ranking.json is always complete, --tier only filters stdout.
    ranking_payload = {
        "ranked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rules_version": RULES_VERSION,
        "filing_count": len(results),
        "summary": {
            "critical_count": sum(1 for r in results if r.tier == "critical"),
            "notable_count": sum(1 for r in results if r.tier == "notable"),
            "quiet_count": sum(1 for r in results if r.tier == "quiet"),
        },
        "filings": [asdict(r) for r in results],
    }
    RANKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    RANKING_PATH.write_text(json.dumps(ranking_payload, indent=2), encoding="utf-8")

    # Print each requested tier in canonical order (critical, notable, quiet).
    for tier_name in VALID_TIERS:
        if tier_name in requested_tiers:
            tier_filings = [r for r in results if r.tier == tier_name]
            _print_tier(tier_name, tier_filings)

    summary = ranking_payload["summary"]
    print(f"[rank] Wrote {RANKING_PATH}")
    print(
        f"[rank] {summary['critical_count']} critical | "
        f"{summary['notable_count']} notable | "
        f"{summary['quiet_count']} quiet"
    )


if __name__ == "__main__":
    main()
