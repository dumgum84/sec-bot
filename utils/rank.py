"""
rank.py — Filing ranker
========================

Reads extractions.json and run_manifest.json, evaluates trigger conditions
on each filing, scores chokepoint connectivity and financial performance,
and produces a flat ranked list sorted by the chosen --sort mode.

Output:
    output/ranking.json   — full ranked list with triggers + scores per filing
    stdout                — ranked breakdown

Usage:
    python rank.py                  # default sort (ctmg)
    python rank.py --sort tcmg      # trigger-first sort

Sort codes: c=chokepoint, t=trigger count, m=operating margin, g=revenue growth
All metrics sort descending. ranking.json always contains all filings.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from fetch import OUTPUT_DIR, DATA_DIR

EXTRACTIONS_PATH = DATA_DIR / "extractions.json"
RUN_MANIFEST_PATH = DATA_DIR / "run_manifest.json"
RANKING_PATH = DATA_DIR / "ranking.json"
RULES_VERSION = "v2-flat"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_recent_year(year_str: Optional[str]) -> bool:
    """
    Heuristic: 'recent' means the current year or either of the two prior years.
    Filters out historical M&A entries (e.g., a 2009 joint venture still
    listed in the current 10-K for context). Computed dynamically so it
    doesn't need updating each calendar year.
    """
    if not year_str:
        return False
    from datetime import datetime
    current_year = datetime.now().year
    recent_years = tuple(str(current_year - i) for i in range(3))
    s = str(year_str).lower()
    return any(yr in s for yr in recent_years)


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
    triggers: list[str] = field(default_factory=list)
    chokepoint_score: int = 0           # set by apply_chokepoint_scores after evaluation
    operating_margin_pct: Optional[float] = None  # from financial_performance extraction, None if missing
    revenue_growth_pct: Optional[float] = None   # from financial_performance extraction, None if missing


def _evaluate_filing(ticker: str, ext: dict, manifest_lookup: dict) -> FilingResult:
    """Run all trigger detectors and collect fired triggers."""
    triggers: list[str] = []

    for detector in TRIGGER_DETECTORS:
        result = detector(ext)
        if result is None:
            continue
        msg, severity = result
        triggers.append(msg)

    # Pull CIK and primary_document from the run manifest so digest.py can
    # build SEC links without re-reading run_manifest itself.
    accession = ext.get("accession", "")
    manifest_entry = manifest_lookup.get(ticker, {})
    cik = manifest_entry.get("cik", "")
    primary_doc = manifest_entry.get("primary_document", "")

    # Pull financial performance scores for sorting. None if not extracted —
    # null is preserved rather than defaulting to 0 so rankings are not
    # distorted by missing data masquerading as neutral profitability.
    fp = ext.get("financial_performance", {})
    om = fp.get("operating_margin_pct")
    rg = fp.get("revenue_growth_pct")
    operating_margin = float(om) if om is not None else None
    revenue_growth = float(rg) if rg is not None else None

    return FilingResult(
        ticker=ticker,
        accession=accession,
        cik=cik,
        primary_document=primary_doc,
        triggers=triggers,
        operating_margin_pct=operating_margin,
        revenue_growth_pct=revenue_growth,
    )


# ---------------------------------------------------------------------------
# Chokepoint scoring
# ---------------------------------------------------------------------------

def apply_chokepoint_scores(results: list[FilingResult], extractions: dict) -> None:
    """
    Compute chokepoint connectivity scores for each FilingResult and set them
    in-place. Must be called after _evaluate_filing completes for all filings,
    and before the final sort, so scores affect reading order.

    Degrades gracefully if chokepoints.py is unavailable or errors: all scores
    remain at their default of 0 and the ranking proceeds normally.
    """
    try:
        from chokepoints import compute_chokepoints, compute_chokepoint_scores
    except ImportError:
        # chokepoints.py not present yet — not fatal, scores stay 0.
        return

    try:
        cp = compute_chokepoints(extractions)
        scores = compute_chokepoint_scores(extractions, cp)
        assigned = 0
        for r in results:
            r.chokepoint_score = scores.get(r.ticker, 0)
            if r.chokepoint_score > 0:
                assigned += 1
        if assigned:
            print(f"[rank] Chokepoint scores applied: {assigned} filings have connectivity > 0")
    except Exception as e:
        print(f"[rank] Chokepoint scoring failed (non-fatal): {e}")


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
    watch_path = DATA_DIR / "watch_queue.json"
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

def _make_sort_key(mode: str = "ctmg"):
    """
    Return a sort key function for the given mode. All metrics sort descending
    (higher = more interesting), with ticker alphabetical as the final tiebreaker.

    Mode letters: c=chokepoint score, t=trigger count, m=operating margin, g=revenue growth
    Valid modes: ctmg, ctgm, tcmg, tcgm
    """
    def key(r: FilingResult):
        c = -r.chokepoint_score
        t = -len(r.triggers)
        # None margins sort below measured values (treat unknown as slight
        # penalty, not neutral). -1.0 is below breakeven but above severe losses.
        m = -(r.operating_margin_pct if r.operating_margin_pct is not None else -1.0)
        g = -(r.revenue_growth_pct if r.revenue_growth_pct is not None else 0.0)
        if mode == "ctmg":
            return (c, t, m, g, r.ticker)
        elif mode == "ctgm":
            return (c, t, g, m, r.ticker)
        elif mode == "tcmg":
            return (t, c, m, g, r.ticker)
        else:  # tcgm
            return (t, c, g, m, r.ticker)
    return key


# Keep _sort_key as the default for any code that calls it directly
def _sort_key(r: FilingResult):
    return _make_sort_key("ctmg")(r)


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def _print_results(results: list[FilingResult], mode: str = "ctmg") -> None:
    """Print ranked filings to stdout."""
    print("=" * 78)
    print(f"  RANKED FILINGS  ({len(results)} total, sort: {mode})")
    print("=" * 78)
    print()

    for i, r in enumerate(results, 1):
        n = len(r.triggers)
        cp_note = f"  [cp={r.chokepoint_score}]" if r.chokepoint_score > 0 else ""
        margin_note = f"  [margin={r.operating_margin_pct:.1f}%]" if r.operating_margin_pct is not None else ""
        growth_note = f"  [growth={r.revenue_growth_pct:.1f}%]" if r.revenue_growth_pct is not None else ""
        print(f"  {i:>3}. {r.ticker:<6}  {n} trigger{'s' if n != 1 else ''}{cp_note}{margin_note}{growth_note}")
        for t in r.triggers:
            print(f"        - {t}")
        if r.triggers:
            print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Rank filings using deterministic rules, no tiers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python rank.py                  # default sort (ctmg)\n"
            "  python rank.py --sort tcmg      # trigger-first sort\n"
            "\n"
            "Sort codes: c=chokepoint, t=trigger, m=margin, g=growth\n"
            "ranking.json always contains all filings."
        ),
    )
    parser.add_argument(
        "--sort", default="ctmg",
        choices=("ctmg", "ctgm", "tcmg", "tcgm"),
        help="Sort order (default: ctmg — chokepoint, trigger, margin, growth)",
    )
    args = parser.parse_args()

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

    # Apply chokepoint connectivity scores as a within-tier tiebreaker.
    # This step is free (no LLM calls) and degrades gracefully if
    # chokepoints.py is unavailable.
    apply_chokepoint_scores(results, extractions)

    results.sort(key=_make_sort_key(args.sort))

    ranking_payload = {
        "ranked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rules_version": RULES_VERSION,
        "sort_mode": args.sort,
        "filing_count": len(results),
        "filings": [asdict(r) for r in results],
    }
    RANKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    RANKING_PATH.write_text(json.dumps(ranking_payload, indent=2), encoding="utf-8")

    _print_results(results, mode=args.sort)

    print(f"[rank] Wrote {RANKING_PATH}")
    print(f"[rank] {len(results)} filings ranked, sort={args.sort}")


if __name__ == "__main__":
    main()