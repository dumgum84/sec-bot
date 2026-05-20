"""
universe.py — Universe builder
==============================

Pulls a list of US-listed tickers, optionally filtered by market-cap band,
and writes them to output/universe.json.

Why yfinance: free, no API key, good enough for our universe-selection needs.
yfinance scrapes Yahoo's endpoints, so it occasionally breaks when Yahoo
changes their HTML. For development that's fine; we can swap in a paid API
(Polygon, FMP) later if reliability matters.

Strategy:
    1. Pick a seed source:
         sp500   — S&P 500 (~503 tickers, mega/large cap only)
         sp1500  — S&P 500 + 400 + 600 (~1500 tickers, mega/mid/small)  [default]
         all     — every SEC-registered ticker (~10k, includes micro and OTC)
    2. Resolve each ticker to its CIK using the SEC mapping. Drop unresolved.
    3. Fetch market cap via yfinance.
    4. Optionally filter by market-cap band (--min-mcap-b / --max-mcap-b).
    5. Sort by market cap (desc by default; --rank-by mcap-asc reverses).
    6. Cap result at --top-n (filtered set may be smaller; that's fine).
    7. Save output/universe.json.

Why mcap band matters:
    Mega-caps (AAPL, NVDA, etc.) are exhaustively analyzed by Wall Street;
    a research bot reading their 10-K is unlikely to surface anything new.
    A $500M–$10B band targets companies large enough to file clean reports
    and trade liquidly, but small enough that careful reading might still
    yield real information. Tune to your appetite.

Run:
    python universe.py                                          # sp1500, no band, top 500 by mcap-desc
    python universe.py --seed sp500                             # mega-caps only
    python universe.py --seed all                               # everything that files with SEC
    python universe.py --min-mcap-b 0.5 --max-mcap-b 10         # 500M-10B band
    python universe.py --max-mcap-b 50 --rank-by mcap-asc       # under-$50B, smallest first
    python universe.py --refresh-cache                          # force re-fetch SEC ticker map
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from io import StringIO
from pathlib import Path
from typing import Optional

import requests

# Path constants are centralized in fetch.py — single source of truth.
from fetch import OUTPUT_DIR, DATA_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USER_AGENT = "Carson R carson.research@gmail.com"

UNIVERSE_PATH = DATA_DIR / "universe.json"
TICKER_CACHE_PATH = DATA_DIR / "_cache" / "sec_ticker_map.json"

DEFAULT_TOP_N = 500
DEFAULT_SEED = "sp1500"
DEFAULT_RANK_BY = "mcap-desc"

VALID_SEEDS = ("sp500", "sp1500", "all")
VALID_RANK_BY = ("mcap-desc", "mcap-asc")

# Wikipedia ticker tables. We always pull S&P 500 if seed is sp500/sp1500;
# the EXTRAs below get added on top for sp1500.
SP500_SOURCE = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

EXTRA_INDEX_SOURCES = {
    # name → (URL, table index on the page)
    "S&P_400": ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", 0),
    "S&P_600": ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", 0),
}

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class UniverseEntry:
    ticker: str           # e.g. "AAPL"
    cik: str              # zero-padded 10-digit CIK, e.g. "0000320193"
    company_name: str
    market_cap: Optional[float]   # in USD, None if unavailable
    source: str           # which seed list it came from


# ---------------------------------------------------------------------------
# Seed: ticker lists from Wikipedia
# ---------------------------------------------------------------------------

def _read_wikipedia_tickers(url: str, table_idx: int = 0) -> list[tuple[str, str]]:
    """
    Scrape ticker symbols from a Wikipedia index page.

    Returns: list of (ticker, company_name) pairs. We use pandas.read_html
    because Wikipedia's tables are well-structured and pandas handles them
    gracefully — way less fragile than handcrafted scraping.
    """
    import pandas as pd
    headers = {"User-Agent": USER_AGENT}
    # pandas can fetch but we want our user-agent on the request
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    # Newer pandas/lxml require a file-like object; passing a raw string
    # makes lxml try to interpret the entire HTML as a filename and fail
    # with an OSError that echoes the entire input back as the "filename".
    tables = pd.read_html(StringIO(resp.text))
    if table_idx >= len(tables):
        raise RuntimeError(f"{url} has only {len(tables)} tables, asked for index {table_idx}")
    df = tables[table_idx]

    # Wikipedia ticker columns are inconsistent across pages.
    # Try a sequence of likely column names.
    ticker_col = None
    name_col = None
    for col in df.columns:
        col_str = str(col).lower()
        if ticker_col is None and ("symbol" in col_str or "ticker" in col_str):
            ticker_col = col
        if name_col is None and ("company" in col_str or "security" in col_str):
            name_col = col

    if ticker_col is None:
        raise RuntimeError(f"Could not find ticker column in {url}; columns: {list(df.columns)}")

    pairs: list[tuple[str, str]] = []
    for _, row in df.iterrows():
        t = str(row[ticker_col]).strip().upper()
        n = str(row[name_col]).strip() if name_col else ""
        # Yahoo / SEC use BRK-B; Wikipedia uses BRK.B. Normalize to dash form
        # since that's what the SEC ticker map uses.
        t = t.replace(".", "-")
        if t and t != "NAN":
            pairs.append((t, n))
    return pairs


def _seed_tickers(seed_type: str, sec_map: dict[str, tuple[str, str]]) -> list[tuple[str, str, str]]:
    """
    Build the candidate ticker list based on seed type.

    Returns: list of (ticker, name, source) tuples.

    seed_type:
        "sp500"  — S&P 500 only (Wikipedia)
        "sp1500" — S&P 500 + S&P 400 + S&P 600 (Wikipedia, deduped)
        "all"    — every ticker in the SEC ticker map (no Wikipedia call)
    """
    if seed_type == "all":
        # Use the SEC ticker map directly. No Wikipedia call needed.
        # This covers everything that files with the SEC, including
        # mid/small caps, micro-caps, ADRs, and other under-followed names.
        # Note: also includes ETFs, mutual funds, and trusts — many of these
        # don't file 10-Ks so they'll show up as "no recent filings" in the
        # watcher stage. That's fine; the noise is cheap to filter downstream.
        print(f"[seed] Using SEC ticker map ({len(sec_map)} tickers)")
        return [(t, name, "SEC_all") for t, (cik, name) in sec_map.items()]

    by_ticker: dict[str, tuple[str, str]] = {}

    # S&P 500 is always included (smallest seed, used by both sp500 and sp1500)
    print(f"[seed] Pulling S&P 500 from Wikipedia...")
    sp500 = _read_wikipedia_tickers(SP500_SOURCE)
    print(f"       {len(sp500)} tickers from S&P 500")
    for t, n in sp500:
        by_ticker[t] = (n, "S&P_500")

    if seed_type == "sp1500":
        # Add mid-cap (S&P 400, ~$5B–$15B) and small-cap (S&P 600, ~$850M–$5B).
        # Together with S&P 500 this covers ~90% of US market cap and excludes
        # the truly weird shell-company / OTC names.
        for name, (url, idx) in EXTRA_INDEX_SOURCES.items():
            try:
                print(f"[seed] Pulling {name} from Wikipedia...")
                extra = _read_wikipedia_tickers(url, idx)
                added = 0
                for t, n in extra:
                    if t not in by_ticker:
                        by_ticker[t] = (n, name)
                        added += 1
                print(f"       +{added} new tickers from {name} (total: {len(by_ticker)})")
            except Exception as e:
                print(f"       WARN: {name} failed ({e}); skipping", file=sys.stderr)

    return [(t, n, src) for t, (n, src) in by_ticker.items()]


# ---------------------------------------------------------------------------
# Market cap enrichment via yfinance
# ---------------------------------------------------------------------------

def _fetch_market_caps(tickers: list[str]) -> dict[str, Optional[float]]:
    """
    Fetch market cap for each ticker using yfinance.

    yfinance Ticker.info is per-ticker and slow; for hundreds of tickers we
    use yfinance.Tickers (batch) which is meaningfully faster. Even so,
    expect ~2 minutes for 500 tickers, ~6 min for 1500, and ~40 min for 10k.

    Returns: {ticker: market_cap_usd or None}
    """
    import logging
    import yfinance as yf

    # yfinance prints HTTP 404s for unknown tickers (e.g., CWEN.A — class
    # shares Yahoo doesn't recognize under our ticker-format conversion)
    # straight to its logger. We already handle the missing data gracefully
    # via except → results[t] = None, so the noise just clutters the output.
    # Silencing yfinance and urllib3 loggers at ERROR level keeps real
    # network failures visible while suppressing per-ticker 404s.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    print(f"[mcap] Fetching market caps for {len(tickers)} tickers...")
    results: dict[str, Optional[float]] = {}

    # yfinance has internal rate limiting; we batch in groups of 50 to avoid
    # hammering its endpoints. Going bigger sometimes triggers throttling.
    BATCH = 50
    start = time.monotonic()
    for i in range(0, len(tickers), BATCH):
        chunk = tickers[i:i + BATCH]
        # yfinance wants dot-notation tickers (BRK.B), not dash (BRK-B).
        yf_chunk = [t.replace("-", ".") for t in chunk]
        try:
            data = yf.Tickers(" ".join(yf_chunk))
            for original, yf_t in zip(chunk, yf_chunk):
                try:
                    info = data.tickers[yf_t].info
                    mcap = info.get("marketCap")
                    results[original] = float(mcap) if mcap else None
                except Exception:
                    results[original] = None
        except Exception as e:
            print(f"       batch {i // BATCH + 1} failed: {e}", file=sys.stderr)
            for original in chunk:
                results[original] = None
        elapsed = time.monotonic() - start
        done = min(i + BATCH, len(tickers))
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(tickers) - done) / rate if rate > 0 else 0
        print(f"       {done}/{len(tickers)} done ({rate:.1f}/s, ~{eta:.0f}s remaining)")

    return results


# ---------------------------------------------------------------------------
# CIK resolution via SEC ticker map (cached)
# ---------------------------------------------------------------------------

def _load_sec_ticker_map(refresh: bool = False) -> dict[str, tuple[str, str]]:
    """
    Returns: {TICKER: (cik_padded_10, company_name)}
    Cached locally to avoid hitting SEC every run.
    """
    TICKER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    cache_age_days = float("inf")
    if TICKER_CACHE_PATH.exists() and not refresh:
        cache_age_days = (time.time() - TICKER_CACHE_PATH.stat().st_mtime) / 86400
        if cache_age_days < 7:
            print(f"[cik]  Using cached SEC ticker map ({cache_age_days:.1f} days old)")
            data = json.loads(TICKER_CACHE_PATH.read_text())
            return {t: (cik, name) for t, (cik, name) in data.items()}

    print(f"[cik]  Fetching SEC ticker map...")
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    resp = requests.get(SEC_TICKER_MAP_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    out: dict[str, tuple[str, str]] = {}
    for _, row in raw.items():
        t = str(row["ticker"]).upper()
        cik = str(row["cik_str"]).zfill(10)
        name = row["title"]
        out[t] = (cik, name)
    TICKER_CACHE_PATH.write_text(json.dumps(out, indent=2))
    print(f"       {len(out)} ticker→CIK mappings cached at {TICKER_CACHE_PATH}")
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_band(min_b: Optional[float], max_b: Optional[float]) -> str:
    """Pretty-print a market-cap band for log lines."""
    if min_b is not None and max_b is not None:
        return f"${min_b:.2f}B–${max_b:.2f}B band"
    if min_b is not None:
        return f"≥${min_b:.2f}B"
    if max_b is not None:
        return f"≤${max_b:.2f}B"
    return "(no band)"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_universe(top_n: int = DEFAULT_TOP_N,
                   seed: str = DEFAULT_SEED,
                   min_mcap_b: Optional[float] = None,
                   max_mcap_b: Optional[float] = None,
                   rank_by: str = DEFAULT_RANK_BY,
                   refresh_cache: bool = False) -> list[UniverseEntry]:
    """
    Build the universe: seed list → CIK resolve → mcap fetch → optional band
    filter → sort → cap at top-n.

    Args:
        top_n: maximum number of tickers to return
        seed: "sp500", "sp1500", or "all"
        min_mcap_b: optional minimum market cap in billions of USD
        max_mcap_b: optional maximum market cap in billions of USD
        rank_by: "mcap-desc" (largest first) or "mcap-asc" (smallest first)
        refresh_cache: force re-fetch of cached SEC ticker map

    Returns: ordered list of UniverseEntry, length <= top_n
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if seed not in VALID_SEEDS:
        raise ValueError(f"seed must be one of {VALID_SEEDS}, got {seed!r}")
    if rank_by not in VALID_RANK_BY:
        raise ValueError(f"rank_by must be one of {VALID_RANK_BY}, got {rank_by!r}")
    if min_mcap_b is not None and max_mcap_b is not None and min_mcap_b > max_mcap_b:
        raise ValueError(f"min_mcap_b ({min_mcap_b}) > max_mcap_b ({max_mcap_b})")

    # 1. SEC ticker map (needed for CIK resolution and for seed=all)
    sec_map = _load_sec_ticker_map(refresh=refresh_cache)

    # 2. Seed list
    seed_list = _seed_tickers(seed, sec_map)
    print(f"[seed] Total candidate tickers: {len(seed_list)}")

    # 3. CIK resolution. Drop tickers we can't resolve (delisted, ADRs not in
    # SEC's main map, etc.). For seed=all this is a no-op since seed comes
    # from sec_map already.
    resolved: list[tuple[str, str, str, str]] = []  # (ticker, name, source, cik)
    unresolved = 0
    for t, name, source in seed_list:
        if t in sec_map:
            cik, sec_name = sec_map[t]
            # SEC's name is more authoritative than Wikipedia's
            resolved.append((t, sec_name, source, cik))
        else:
            unresolved += 1
    print(f"[cik]  Resolved {len(resolved)} tickers, dropped {unresolved} (no SEC CIK)")

    # 4. Market cap enrichment (only for resolved tickers)
    tickers = [r[0] for r in resolved]
    mcaps = _fetch_market_caps(tickers)

    # 5. Build entries
    entries: list[UniverseEntry] = []
    for ticker, sec_name, source, cik in resolved:
        entries.append(UniverseEntry(
            ticker=ticker,
            cik=cik,
            company_name=sec_name,
            market_cap=mcaps.get(ticker),
            source=source,
        ))

    # 6. Apply band filter (if any band specified). When a band is active, we
    # drop tickers without mcap data — we can't filter what we can't measure,
    # and silently keeping them would skew the result.
    band_active = (min_mcap_b is not None) or (max_mcap_b is not None)
    if band_active:
        before_null = len(entries)
        entries = [e for e in entries if e.market_cap is not None]
        null_dropped = before_null - len(entries)

        min_usd = (min_mcap_b * 1e9) if min_mcap_b is not None else 0.0
        max_usd = (max_mcap_b * 1e9) if max_mcap_b is not None else float("inf")
        before_band = len(entries)
        entries = [e for e in entries if min_usd <= e.market_cap <= max_usd]
        band_dropped = before_band - len(entries)

        print(f"[band] {_format_band(min_mcap_b, max_mcap_b)}: "
              f"{len(entries)} pass "
              f"(dropped {null_dropped} no-mcap, {band_dropped} out-of-band)")

    # 7. Sort. None-mcap entries always go to the end regardless of direction.
    if rank_by == "mcap-asc":
        entries.sort(key=lambda e: (e.market_cap is None, e.market_cap or 0))
    else:  # mcap-desc
        entries.sort(key=lambda e: (e.market_cap is None, -(e.market_cap or 0)))

    # 8. Cap at top_n
    selected = entries[:top_n]

    with_mcap = sum(1 for e in selected if e.market_cap is not None)
    if band_active and len(selected) < top_n:
        print(f"[rank] Selected {len(selected)} tickers ({rank_by}); "
              f"requested top {top_n} but band only had {len(selected)}")
    else:
        print(f"[rank] Selected top {len(selected)} by {rank_by} "
              f"({with_mcap} with mcap data, {len(selected) - with_mcap} without)")

    return selected



def build_universe_from_tickers(tickers: list[str],
                                refresh_cache: bool = False) -> list[UniverseEntry]:
    """
    Build a universe from a specific list of tickers, bypassing all seed
    lists, mcap filters, and top-n caps. Used for targeted runs on specific
    companies that may not be in the normal S&P 1500 universe.

    Tickers that cannot be resolved to a CIK are skipped with a warning.
    Market cap is fetched best-effort; missing mcap does not drop the ticker.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sec_map = _load_sec_ticker_map(refresh=refresh_cache)

    resolved: list[tuple[str, str, str]] = []  # (ticker, name, cik)
    unresolved: list[str] = []
    for raw in tickers:
        t = raw.upper().strip()
        if t in sec_map:
            cik, name = sec_map[t]
            resolved.append((t, name, cik))
        else:
            unresolved.append(t)

    if unresolved:
        print(f"[tickers] WARNING — could not resolve: {', '.join(unresolved)}")
    print(f"[tickers] Resolved {len(resolved)} ticker(s): {', '.join(r[0] for r in resolved)}")

    mcaps = _fetch_market_caps([r[0] for r in resolved])

    entries: list[UniverseEntry] = []
    for ticker, name, cik in resolved:
        mcap = mcaps.get(ticker)
        mcap_str = f"${mcap/1e9:.2f}B" if mcap else "(no mcap)"
        print(f"  {ticker:<8} {mcap_str}  {name}")
        entries.append(UniverseEntry(
            ticker=ticker,
            cik=cik,
            company_name=name,
            market_cap=mcap,
            source="manual",
        ))

    return entries


def save_universe(entries: list[UniverseEntry], path: Path = UNIVERSE_PATH) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "count": len(entries),
        "entries": [asdict(e) for e in entries],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"[save] Universe written to {path} ({len(entries)} entries)")


def load_universe(path: Path = UNIVERSE_PATH) -> list[UniverseEntry]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run universe.py first")
    payload = json.loads(path.read_text())
    return [UniverseEntry(**e) for e in payload["entries"]]


def main():
    parser = argparse.ArgumentParser(
        description="Build the ticker universe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python universe.py                                       # default: sp1500, top 500\n"
            "  python universe.py --seed sp500                          # mega-caps only\n"
            "  python universe.py --seed all                            # all SEC filers\n"
            "  python universe.py --min-mcap-b 0.5 --max-mcap-b 10      # 500M-10B band\n"
            "  python universe.py --max-mcap-b 50 --rank-by mcap-asc    # under-50B, smallest first\n"
        ),
    )
    parser.add_argument("--seed", choices=VALID_SEEDS, default=DEFAULT_SEED,
                        help=f"Seed source (default: {DEFAULT_SEED})")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"Max number of tickers to keep (default: {DEFAULT_TOP_N})")
    parser.add_argument("--min-mcap-b", type=float, default=None,
                        help="Minimum market cap in billions USD (e.g. 0.5)")
    parser.add_argument("--max-mcap-b", type=float, default=None,
                        help="Maximum market cap in billions USD (e.g. 10)")
    parser.add_argument("--rank-by", choices=VALID_RANK_BY, default=DEFAULT_RANK_BY,
                        help=f"Sort direction (default: {DEFAULT_RANK_BY})")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Force re-fetch of cached SEC ticker map.")
    args = parser.parse_args()

    entries = build_universe(
        top_n=args.top_n,
        seed=args.seed,
        min_mcap_b=args.min_mcap_b,
        max_mcap_b=args.max_mcap_b,
        rank_by=args.rank_by,
        refresh_cache=args.refresh_cache,
    )
    save_universe(entries)

    # Print a quick preview so the user knows it worked.
    direction = "smallest" if args.rank_by == "mcap-asc" else "largest"
    print()
    print(f"Top 10 ({direction} by market cap):")
    for i, e in enumerate(entries[:10], 1):
        mcap_b = (e.market_cap / 1e9) if e.market_cap else None
        mcap_str = f"${mcap_b:,.2f}B" if mcap_b else "(no mcap)"
        print(f"  {i:>2}. {e.ticker:<6} {mcap_str:>12}  {e.company_name}")


if __name__ == "__main__":
    main()