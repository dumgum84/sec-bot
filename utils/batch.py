"""
batch.py — Batch filing processor
==================================

Reads the watch queue (output/watch_queue.json) and runs the section-
extraction pipeline on each filing. Tracks status per filing so downstream
stages know which parses succeeded, which were flagged with anomalies, and
which failed outright.

Output: output/run_manifest.json

Usage:
    python batch.py                                  # process the full queue
    python batch.py --max 20                         # process only the first 20 entries
    python batch.py --skip-cached                    # skip filings already in cache
    python batch.py --pipeline                       # run universe → watcher → batch end-to-end

Pipeline mode also accepts universe-builder flags:
    python batch.py --pipeline --seed sp500 --max 5
    python batch.py --pipeline --min-mcap-b 0.5 --max-mcap-b 10 --top-n 100 --days 365
    python batch.py --pipeline --max-mcap-b 50 --rank-by mcap-asc --top-n 50

Status semantics:
    "ok"      — sections extracted, zero warn-level anomalies. May still
                contain info-level notes (e.g., stub sections that legitimately
                cross-reference exhibits). Safe for downstream LLM extraction.
    "flagged" — sections extracted, but at least one warn-level anomaly
                fired (too_small or too_large). Section data is likely
                unreliable; downstream should skip the affected sections.
    "failed"  — couldn't extract any sections, or download/parse threw an
                exception. Skip downstream entirely.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

# Re-use the splitter, downloader, and anomaly types we already built.
from fetch import (
    Anomaly,
    download_filing,
    html_to_text,
    split_into_sections,
    _detect_anomalies,
    FilingRef,
    OUTPUT_DIR,
    DATA_DIR,
    FILINGS_DIR,
)
from watcher import load_watch_queue, WatchedFiling

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUN_MANIFEST_PATH = DATA_DIR / "run_manifest.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FilingStatus:
    """Per-filing processing result. Downstream stages key off `status`."""
    ticker: str
    cik: str
    form: str
    filing_date: str
    accession: str
    output_dir: Optional[str]      # path under output/, None if processing failed
    status: str                    # "ok" | "flagged" | "failed"
    anomalies: list[Anomaly]       # structured anomaly records
    sections_found: list[str]      # which item_X keys were extracted
    error: Optional[str] = None    # if status=="failed"
    processed_at: str = ""         # ISO timestamp


def _anomaly_from_dict(d: dict) -> Anomaly:
    """Hydrate an Anomaly from its JSON dict form (e.g. from a cached manifest)."""
    er = d.get("expected_range")
    if isinstance(er, list):
        # JSON loses tuple-ness; restore for type consistency.
        er = tuple(er)
    return Anomaly(
        section=d["section"],
        kind=d["kind"],
        severity=d["severity"],
        observed_value=d["observed_value"],
        expected_range=er,
        message=d["message"],
    )


def _classify_status(sections: dict, anomalies: list[Anomaly]) -> str:
    """
    Determine filing-level status.

    Info-level anomalies (stubs) do NOT promote to 'flagged' — they describe
    expected/known patterns, not splitter problems. Only warn-level anomalies
    flag a filing as unreliable for downstream use.
    """
    if not sections:
        return "failed"
    if any(a.severity == "warn" for a in anomalies):
        return "flagged"
    return "ok"


def _format_anomaly_counts(anomalies: list[Anomaly]) -> str:
    """Render '(2 warn, 1 info)' or '(1 info)' or '' depending on what's there."""
    warn = sum(1 for a in anomalies if a.severity == "warn")
    info = sum(1 for a in anomalies if a.severity == "info")
    parts: list[str] = []
    if warn:
        parts.append(f"{warn} warn")
    if info:
        parts.append(f"{info} info")
    return f" ({', '.join(parts)})" if parts else ""


# ---------------------------------------------------------------------------
# Per-filing pipeline
# ---------------------------------------------------------------------------

def _process_one(filing: WatchedFiling, *, skip_cached: bool = False) -> FilingStatus:
    """
    Run the pipeline for a single filing. Returns a FilingStatus describing
    the outcome. Captures stdout from `fetch` helpers so the per-filing
    output doesn't drown the batch progress log.
    """
    out_dir = FILINGS_DIR / filing.ticker.upper() / filing.accession
    raw_path = out_dir / "raw.htm"
    sections_dir = out_dir / "sections"
    manifest_path = out_dir / "manifest.json"
    iso_now = time.strftime("%Y-%m-%dT%H:%M:%S")

    # If we've already processed this filing successfully, optionally skip.
    # Old caches (string-format anomalies) are rejected here to force a
    # reprocess; mixing string and structured formats downstream would break
    # everything.
    if skip_cached and manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
            raw_anomalies = existing.get("anomalies", [])
            if any(isinstance(a, str) for a in raw_anomalies):
                raise ValueError("legacy (string) anomaly format; reprocessing")
            anomalies = [_anomaly_from_dict(a) for a in raw_anomalies]
            return FilingStatus(
                ticker=filing.ticker,
                cik=filing.cik,
                form=filing.form,
                filing_date=filing.filing_date,
                accession=filing.accession,
                output_dir=str(out_dir),
                status=existing.get("status", "ok"),
                anomalies=anomalies,
                sections_found=existing.get("sections_found", []),
                processed_at=existing.get("processed_at", iso_now),
            )
        except Exception:
            # Cached manifest unreadable or in old format; reprocess.
            pass

    out_dir.mkdir(parents=True, exist_ok=True)

    ref = FilingRef(
        accession=filing.accession,
        accession_no_dashes=filing.accession.replace("-", ""),
        form=filing.form,
        filing_date=filing.filing_date,
        primary_document=filing.primary_document,
        cik_int=int(filing.cik),
    )

    # Capture stdout/stderr from helpers so per-filing chatter doesn't clobber
    # the batch progress line. Errors still propagate via exceptions.
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            if raw_path.exists():
                html = raw_path.read_text(encoding="utf-8")
            else:
                html = download_filing(ref, raw_path)

            text = html_to_text(html)
            (out_dir / "cleaned.txt").write_text(text, encoding="utf-8")

            sections = split_into_sections(html)

            sections_dir.mkdir(exist_ok=True)
            for key, content in sections.items():
                (sections_dir / f"{key}.txt").write_text(content, encoding="utf-8")

        anomalies = _detect_anomalies(sections)
        status = _classify_status(sections, anomalies)

        result = FilingStatus(
            ticker=filing.ticker,
            cik=filing.cik,
            form=filing.form,
            filing_date=filing.filing_date,
            accession=filing.accession,
            output_dir=str(out_dir),
            status=status,
            anomalies=anomalies,
            sections_found=list(sections.keys()),
            processed_at=iso_now,
        )

        # Per-filing manifest for fast cache lookup on next run. asdict()
        # recursively serializes the nested Anomaly dataclasses.
        manifest_path.write_text(json.dumps(asdict(result), indent=2))
        return result

    except Exception as e:
        return FilingStatus(
            ticker=filing.ticker,
            cik=filing.cik,
            form=filing.form,
            filing_date=filing.filing_date,
            accession=filing.accession,
            output_dir=None,
            status="failed",
            anomalies=[],
            sections_found=[],
            error=f"{type(e).__name__}: {e}",
            processed_at=iso_now,
        )


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------

def run_batch(*, max_filings: Optional[int] = None, skip_cached: bool = False) -> list[FilingStatus]:
    queue = load_watch_queue()
    if max_filings is not None:
        queue = queue[:max_filings]

    print(f"[batch] Processing {len(queue)} filings...")
    print()

    results: list[FilingStatus] = []
    counts = {"ok": 0, "flagged": 0, "failed": 0}
    start = time.monotonic()

    for i, filing in enumerate(queue, 1):
        result = _process_one(filing, skip_cached=skip_cached)
        results.append(result)
        counts[result.status] = counts.get(result.status, 0) + 1

        elapsed = time.monotonic() - start
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(queue) - i) / rate if rate > 0 else 0

        marker = {"ok": "✓", "flagged": "⚠", "failed": "✗"}[result.status]
        note = _format_anomaly_counts(result.anomalies)
        err_note = f" — {result.error}" if result.error else ""
        print(f"  [{i:>4}/{len(queue)}] {marker} {filing.filing_date}  {filing.ticker:<6} "
              f"{filing.form:<6}  {result.status}{note}{err_note}")

        # Progress summary every 25 filings or at the end.
        if i % 25 == 0 or i == len(queue):
            print(f"           — progress: ok={counts['ok']} flagged={counts['flagged']} "
                  f"failed={counts['failed']}  ({rate:.1f}/s, ~{eta:.0f}s remaining)")

    return results


def save_run_manifest(results: list[FilingStatus], path: Path = RUN_MANIFEST_PATH) -> None:
    counts = {"ok": 0, "flagged": 0, "failed": 0}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1

    # Aggregate anomaly counts across all filings — useful for understanding
    # how many filings have stubs vs real misses without scanning each entry.
    warn_total = sum(
        sum(1 for a in r.anomalies if a.severity == "warn") for r in results
    )
    info_total = sum(
        sum(1 for a in r.anomalies if a.severity == "info") for r in results
    )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": len(results),
        "counts": counts,
        "anomaly_totals": {"warn": warn_total, "info": info_total},
        "filings": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print()
    print(f"[save]  Run manifest written to {path}")
    print(f"        ok={counts['ok']}  flagged={counts['flagged']}  failed={counts['failed']}")
    if info_total or warn_total:
        print(f"        anomalies: {warn_total} warn-level, {info_total} info-level (e.g., stubs)")
    if counts["ok"]:
        clean_pct = counts["ok"] / len(results) * 100
        print(f"        clean rate: {clean_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Process the watch queue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python batch.py                                                  # process queue\n"
            "  python batch.py --pipeline                                       # universe→watcher→batch (defaults)\n"
            "  python batch.py --pipeline --seed sp500 --max 5                  # quick mega-cap test\n"
            "  python batch.py --pipeline --min-mcap-b 0.5 --max-mcap-b 10      # mid/small-cap band\n"
        ),
    )
    parser.add_argument("--max", type=int, default=None,
                        help="Process only the first N filings.")
    parser.add_argument("--skip-cached", action="store_true",
                        help="Skip filings already processed (re-uses existing manifests).")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run universe → watcher → batch end-to-end before processing.")

    # Pipeline-only flags. All forwarded to universe.build_universe / watcher.build_watch_queue.
    pipe = parser.add_argument_group("--pipeline options (ignored without --pipeline)")
    pipe.add_argument("--seed", default=None,
                      help="Universe seed: sp500 / sp1500 / all")
    pipe.add_argument("--top-n", type=int, default=None,
                      help="Universe size cap")
    pipe.add_argument("--min-mcap-b", type=float, default=None,
                      help="Universe: minimum market cap (billions USD)")
    pipe.add_argument("--max-mcap-b", type=float, default=None,
                      help="Universe: maximum market cap (billions USD)")
    pipe.add_argument("--rank-by", default=None,
                      help="Universe sort: mcap-desc / mcap-asc")
    pipe.add_argument("--days", type=int, default=None,
                      help="Watcher: filing recency window in days")

    args = parser.parse_args()

    if args.pipeline:
        # Convenience: full sequence in one command.
        from universe import (
            build_universe, save_universe,
            DEFAULT_TOP_N, DEFAULT_SEED, DEFAULT_RANK_BY,
        )
        from watcher import build_watch_queue, save_watch_queue, DEFAULT_DAYS

        # Apply defaults only where the user didn't override
        seed = args.seed or DEFAULT_SEED
        top_n = args.top_n if args.top_n is not None else DEFAULT_TOP_N
        rank_by = args.rank_by or DEFAULT_RANK_BY
        days = args.days if args.days is not None else DEFAULT_DAYS

        print("=" * 60)
        print(" PIPELINE STAGE 1: BUILD UNIVERSE")
        print("=" * 60)
        universe = build_universe(
            top_n=top_n,
            seed=seed,
            min_mcap_b=args.min_mcap_b,
            max_mcap_b=args.max_mcap_b,
            rank_by=rank_by,
        )
        save_universe(universe)

        print()
        print("=" * 60)
        print(" PIPELINE STAGE 2: BUILD WATCH QUEUE")
        print("=" * 60)
        queue = build_watch_queue(days=days)
        save_watch_queue(queue)

        print()
        print("=" * 60)
        print(" PIPELINE STAGE 3: PROCESS FILINGS")
        print("=" * 60)

    results = run_batch(max_filings=args.max, skip_cached=args.skip_cached)
    save_run_manifest(results)


if __name__ == "__main__":
    main()
