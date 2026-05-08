"""
sec_report.py — single-command pipeline runner
================================================

Runs the full sec_bot pipeline end-to-end:

    1. Build universe        (utils/universe.py)
    2. Build watch queue     (utils/watcher.py)
    3. Process filings       (utils/batch.py)
    4. LLM extraction        (utils/extract.py)   <-- $$ Anthropic API
    5. Rank into tiers       (utils/rank.py)
    6. Generate report       (utils/digest.py)    <-- $$ Anthropic API

Caching is on by default. Already-fetched filings and already-extracted
tickers are reused — only NEW filings hit the API. Pass --refresh to
force re-extraction of everything (expensive, only useful when iterating
on the extract.py prompt or recovering from bad model output). The file
under output/sec_report.md is regenerated every run regardless.

Run:
    py sec_report.py                  # use sensible defaults (with cache)
    py sec_report.py --days 45        # only filings from past 45 days
    py sec_report.py --top-n 300      # smaller, cheaper run
    py sec_report.py --refresh        # ignore cache, re-pay all LLM calls

Defaults match the standard "find under-followed opportunities" preset:
    --seed sp1500 --min-mcap-b 0.3 --max-mcap-b 10
    --rank-by mcap-asc --top-n 500 --days 365

The 365-day default is sized for a fresh first run. For monthly cadence
re-runs after that, pass --days 45 (catches the past month plus a 15-day
safety margin for SEC indexing delays).

Output:
    output/sec_report.md          ← read this
    output/data/*.json            ← machine-readable artifacts
    output/filings/<TICKER>/...   ← raw 10-Ks and split sections
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Make utils/ importable so the worker scripts can import each other
# unchanged (e.g., `from fetch import OUTPUT_DIR` still works).
sys.path.insert(0, str(Path(__file__).parent / "utils"))


# ---------------------------------------------------------------------------
# Defaults — change here, not on the command line every time.
# ---------------------------------------------------------------------------

DEFAULT_SEED = "sp1500"
DEFAULT_MIN_MCAP_B = 0.3
DEFAULT_MAX_MCAP_B = 10.0
DEFAULT_RANK_BY = "mcap-asc"
DEFAULT_TOP_N = 500
DEFAULT_DAYS = 365


# ---------------------------------------------------------------------------
# Migration helper — moves files from old layout to new layout if needed.
# ---------------------------------------------------------------------------

def _migrate_old_layout(output_dir: Path, data_dir: Path) -> None:
    """
    The pre-refactor layout had JSON files directly under output/ and the
    markdown report named digest.md. New layout puts JSON under output/data/
    and renames the report to sec_report.md.

    Caches representing real money (extractions.json, digest_summaries.json)
    must survive the move so the user doesn't lose work. This runs once on
    first execution against the new layout.
    """
    legacy_json_files = [
        "universe.json",
        "watch_queue.json",
        "run_manifest.json",
        "extractions.json",
        "ranking.json",
        "digest_summaries.json",
    ]

    moves: list[tuple[Path, Path]] = []
    for name in legacy_json_files:
        src = output_dir / name
        dst = data_dir / name
        if src.exists() and not dst.exists():
            moves.append((src, dst))

    # Cache directory
    src_cache = output_dir / "_cache"
    dst_cache = data_dir / "_cache"
    if src_cache.exists() and not dst_cache.exists():
        moves.append((src_cache, dst_cache))

    # Old digest.md → new sec_report.md
    legacy_digest = output_dir / "digest.md"
    new_report = output_dir / "sec_report.md"
    rename_digest = legacy_digest.exists() and not new_report.exists()

    if not moves and not rename_digest:
        return  # already migrated

    print("[migrate] Detected pre-refactor layout. Moving files to new locations...")
    data_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f"           {src.name} → {dst.relative_to(output_dir)}")
    if rename_digest:
        shutil.move(str(legacy_digest), str(new_report))
        print(f"           digest.md → sec_report.md")
    print()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _banner(stage: int, total: int, title: str) -> None:
    """Print a section banner so each stage is visually distinct in the log."""
    print()
    print("=" * 70)
    print(f" STAGE {stage}/{total}: {title}")
    print("=" * 70)


def run_pipeline(args: argparse.Namespace) -> None:
    # Imports happen inside the function so the script doesn't pay the
    # import cost (yfinance, anthropic, edgartools) just to print --help.
    from fetch import OUTPUT_DIR, DATA_DIR
    from universe import build_universe, save_universe
    from watcher import build_watch_queue, save_watch_queue
    from batch import run_batch, save_run_manifest
    import extract
    import rank
    import digest

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Migrate any old-layout files before doing anything else, so caches
    # carry over rather than getting bypassed and re-paid for.
    _migrate_old_layout(OUTPUT_DIR, DATA_DIR)

    use_cache = not args.refresh
    total_stages = 6
    pipeline_start = time.monotonic()

    # ---------------------------------------------------------------------
    _banner(1, total_stages, "BUILD UNIVERSE")
    universe = build_universe(
        top_n=args.top_n,
        seed=args.seed,
        min_mcap_b=args.min_mcap_b,
        max_mcap_b=args.max_mcap_b,
        rank_by=args.rank_by,
    )
    save_universe(universe)

    # ---------------------------------------------------------------------
    _banner(2, total_stages, "BUILD WATCH QUEUE")
    queue = build_watch_queue(days=args.days)
    save_watch_queue(queue)

    # ---------------------------------------------------------------------
    _banner(3, total_stages, "PROCESS FILINGS")
    results = run_batch(skip_cached=use_cache)
    save_run_manifest(results)

    # ---------------------------------------------------------------------
    _banner(4, total_stages, "LLM EXTRACTION")
    # extract.py expects to be invoked from CLI. Drive it programmatically
    # by calling its internals with a tiny synthetic argparse Namespace.
    import json
    import os
    from anthropic import Anthropic
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY not found. Create a .env file in the project "
            "root (next to sec_report.py) containing: ANTHROPIC_API_KEY=sk-ant-..."
        )
    client = Anthropic(api_key=api_key)

    manifest_path = DATA_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    targets = [f["ticker"] for f in manifest.get("filings", []) if f.get("status") == "ok"]

    extractions = extract._load_extractions()
    if extractions:
        print(f"[extract] Loaded {len(extractions)} existing extractions from cache")

    print(f"[extract] {len(targets)} ok-status filings to process "
          f"({sum(1 for t in targets if t.upper() in extractions and use_cache)} cached, "
          f"{sum(1 for t in targets if t.upper() not in extractions or not use_cache)} new)")
    print(f"[extract] Each filing = 7 LLM calls (one per category)")
    print()

    for i, ticker in enumerate(targets, 1):
        ticker_upper = ticker.upper()
        if use_cache and ticker_upper in extractions:
            continue
        print(f"[{i}/{len(targets)}] {ticker_upper}")
        result = extract._extract_one(client, ticker)
        if result.success:
            extractions[ticker_upper] = result.extraction
            extract._save_extractions(extractions)
            print(f"      → saved ({result.duration_seconds:.1f}s)")
        else:
            print(f"      ✗ FAILED: {result.error}")
        print()

    print(f"[extract] Total extractions in file: {len(extractions)}")

    # ---------------------------------------------------------------------
    _banner(5, total_stages, "RANK FILINGS")
    payload = json.loads(rank.EXTRACTIONS_PATH.read_text(encoding="utf-8"))
    rank_extractions = payload.get("extractions", {})
    manifest_lookup = rank._load_manifest_lookup()
    print(f"[rank] Evaluating {len(rank_extractions)} filings using rules version {rank.RULES_VERSION}")
    rank_results = [
        rank._evaluate_filing(t, ext, manifest_lookup)
        for t, ext in rank_extractions.items()
    ]

    # Apply chokepoint connectivity scores as a within-tier tiebreaker.
    # Free step — pure aggregation over data already in extractions.json.
    rank.apply_chokepoint_scores(rank_results, rank_extractions)

    rank_results.sort(key=rank._make_sort_key(args.sort))

    from dataclasses import asdict
    ranking_payload = {
        "ranked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rules_version": rank.RULES_VERSION,
        "sort_mode": args.sort,
        "filing_count": len(rank_results),
        "filings": [asdict(r) for r in rank_results],
    }
    rank.RANKING_PATH.parent.mkdir(parents=True, exist_ok=True)
    rank.RANKING_PATH.write_text(json.dumps(ranking_payload, indent=2), encoding="utf-8")
    print(f"[rank] {len(rank_results)} filings ranked, sort={args.sort}")

    # ---------------------------------------------------------------------
    _banner(6, total_stages, "GENERATE SEC REPORT")
    extractions_payload = json.loads(digest.EXTRACTIONS_PATH.read_text(encoding="utf-8"))
    extractions = extractions_payload.get("extractions", {})
    ranking = json.loads(digest.RANKING_PATH.read_text(encoding="utf-8"))
    report_md = digest.build_digest(
        extractions, ranking,
        use_llm=True,
        refresh_summaries=args.refresh,
    )
    digest.DEFAULT_DIGEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    digest.DEFAULT_DIGEST_PATH.write_text(report_md, encoding="utf-8")
    size_kb = digest.DEFAULT_DIGEST_PATH.stat().st_size / 1024
    print(f"[digest] Wrote {len(rank_results)} filings to {digest.DEFAULT_DIGEST_PATH} ({size_kb:.1f} KB)")

    # ---------------------------------------------------------------------
    elapsed_min = (time.monotonic() - pipeline_start) / 60
    print()
    print("=" * 70)
    print(f" PIPELINE COMPLETE in {elapsed_min:.1f} minutes")
    print(f" Report: {digest.DEFAULT_DIGEST_PATH}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full sec_bot pipeline in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Defaults:\n"
            f"  --seed {DEFAULT_SEED} --min-mcap-b {DEFAULT_MIN_MCAP_B} "
            f"--max-mcap-b {DEFAULT_MAX_MCAP_B}\n"
            f"  --rank-by {DEFAULT_RANK_BY} --top-n {DEFAULT_TOP_N} "
            f"--days {DEFAULT_DAYS}\n"
            "\n"
            "Caching is ON by default for filings and extractions. Pass --refresh\n"
            "to force a full rebuild (re-fetches, re-extracts, re-summarizes — $$$)."
        ),
    )
    parser.add_argument("--seed", default=DEFAULT_SEED,
                        choices=("sp500", "sp1500", "all"),
                        help="Universe seed list")
    parser.add_argument("--min-mcap-b", type=float, default=DEFAULT_MIN_MCAP_B,
                        help="Minimum market cap (billions USD)")
    parser.add_argument("--max-mcap-b", type=float, default=DEFAULT_MAX_MCAP_B,
                        help="Maximum market cap (billions USD)")
    parser.add_argument("--rank-by", default=DEFAULT_RANK_BY,
                        choices=("mcap-asc", "mcap-desc"),
                        help="Sort direction within the band")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help="Maximum number of tickers to include")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help="Filing recency window")
    parser.add_argument("--sort", default="ctmg",
                        choices=("ctmg", "ctgm", "tcmg", "tcgm"),
                        help="Sort order: c=chokepoint, t=trigger, m=margin, g=growth (default: ctmg)")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore caches; rebuild from scratch (expensive).")
    args = parser.parse_args()

    run_pipeline(args)


if __name__ == "__main__":
    main()
