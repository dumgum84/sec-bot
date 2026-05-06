# sec_bot

A research bot that fetches SEC 10-K filings, extracts structured facts via
LLM, groups them into priority tiers using deterministic rules, and produces
a human-readable markdown report with research-note summaries per filing.

## Quick start

```
py sec_report.py
```

That's it. The pipeline runs all six stages (universe → watcher → batch →
extract → rank → digest), uses caches where possible, and writes the final
report to `output/sec_report.md`.

Default settings target under-followed small/mid-caps:

| Flag | Default | Meaning |
|---|---|---|
| `--seed` | `sp1500` | S&P 500 + 400 + 600 |
| `--min-mcap-b` | `0.5` | Minimum market cap ($B) |
| `--max-mcap-b` | `5.0` | Maximum market cap ($B) |
| `--rank-by` | `mcap-asc` | Sort direction within band |
| `--top-n` | `500` | Maximum tickers in universe |
| `--days` | `45` | Filing recency window |

Override anything from the command line. Pass `--refresh` to ignore caches
and rebuild from scratch (expensive — re-pays for all extraction/summary
LLM calls).

## One-time setup

```
pip install requests beautifulsoup4 lxml edgartools yfinance pandas anthropic python-dotenv pydantic
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Project layout

```
secbot_0.1.0/
├── sec_report.py            # one-command pipeline runner
├── README.md
├── .env                     # API key
├── utils/                   # worker modules — usually not run directly
│   ├── fetch.py             # fetch + section split for one filing
│   ├── universe.py          # ticker universe builder
│   ├── watcher.py           # find recent 10-K filings
│   ├── batch.py             # batch process the watch queue
│   ├── extract.py           # LLM extraction (6 calls per filing)
│   ├── rank.py              # tier-based ranking (rules in this file)
│   └── digest.py            # markdown report generator
└── output/
    ├── sec_report.md        # ← read this
    ├── data/                # machine-readable artifacts
    │   ├── universe.json
    │   ├── watch_queue.json
    │   ├── run_manifest.json
    │   ├── extractions.json
    │   ├── ranking.json
    │   ├── digest_summaries.json
    │   └── _cache/
    └── filings/             # raw 10-Ks and split sections per ticker
```

## How the pipeline works

```
sec_report.py
   ↓
1. universe.py    builds output/data/universe.json
                  (ticker list filtered by mcap band)
   ↓
2. watcher.py     builds output/data/watch_queue.json
                  (10-K filings filed within --days)
   ↓
3. batch.py       fetches each filing's HTML, splits into sections,
                  writes output/filings/<TICKER>/<accession>/
                  + output/data/run_manifest.json
   ↓
4. extract.py     runs Claude Haiku 4.5 6× per filing,
                  writes output/data/extractions.json   ← $$ API calls
   ↓
5. rank.py        evaluates triggers, assigns critical/notable/quiet,
                  writes output/data/ranking.json
   ↓
6. digest.py      writes one research note per filing,
                  writes output/sec_report.md           ← $$ API calls
```

Cost per full run from scratch (~500 filings): ~$50 in API spend, ~120
minutes wall time. Repeat runs are much cheaper because filings and
extractions are cached — only new tickers since the last run actually hit
the API.

## Tier definitions

A filing lands in **critical** if any of these triggers fire:

- Going concern language
- Material weakness in internal controls
- Restatement of prior-period financials
- Impairment of $500M or more
- Litigation with disclosed exposure of $100M or more
- Regulatory action by DOJ / FDA / FTC / FERC / SEC / EPA / Department of Commerce

A filing lands in **notable** if (it's not already critical and) any of
these fire:

- CEO change during fiscal year
- CFO change during fiscal year
- Auditor change during fiscal year
- Impairment of $100M to $500M
- Other regulatory action (non-significant agency)
- Pending litigation without disclosed exposure
- M&A activity worth $1B or more
- Planned capex of $1B or more for next fiscal year

A filing lands in **quiet** otherwise.

To tune triggers, edit the detector functions in `utils/rank.py`. After
editing, re-run `py sec_report.py` — re-ranking is free, and the digest
will regenerate using cached extractions and summaries.

## Cadence

Re-run monthly with the default `--days 45`. The 45-day window catches any
filing from the past month plus a 15-day safety margin for SEC indexing
delays. Caching means each monthly run only does work on filings that are
actually new.

## Iteration is cheap

Once you've done one full run, three things are essentially free to iterate
on:

- **Tier rules.** Edit `utils/rank.py`. Re-running re-ranks instantly.
- **Digest layout / prompt.** Edit `utils/digest.py`. Add `--refresh` to
  regenerate all summaries (~$2-3 for a 500-filing run); without `--refresh`
  only new filings hit the API.
- **Universe parameters.** Different `--min-mcap-b`, `--max-mcap-b`,
  `--rank-by` produce different universes; cached data is reused for any
  filings that overlap.

## Output meanings

`output/data/run_manifest.json` per-filing status:

- **ok** — sections extracted cleanly, safe for downstream LLM extraction
- **flagged** — at least one warn-level anomaly fired; section data may be
  unreliable
- **failed** — couldn't extract sections or download/parse threw an
  exception

Every filing's `output/filings/<TICKER>/<accession>/manifest.json` includes
a structured `anomalies` array describing any size-based issues with the
splitter's output.

## Performance reference

| Stage | Time | Cost |
|---|---|---|
| 1. Universe (sp1500) | ~6 min | — |
| 2. Watcher | ~1 min per 100 tickers | — |
| 3. Batch | ~5-10 sec per filing | — |
| 4. Extract | ~17 sec per filing | ~$0.10/filing |
| 5. Rank | <1 sec total | — |
| 6. Digest (initial) | ~3-5 sec per filing | ~$0.025/filing |
| 6. Digest (cached) | <1 sec total | — |

## Notes

- Section detection uses [edgartools](https://github.com/dgunning/edgartools)
  — well-tested across thousands of filings, handles inline XBRL and TOC vs
  body disambiguation cleanly.
- The current splitter is wired for **10-K** filings only.
- All stages cache aggressively. Migration from pre-refactor layouts is
  automatic on first run — old `output/*.json` files move to `output/data/`,
  and `output/digest.md` is renamed to `output/sec_report.md`.
