# sec_bot

A research bot that fetches SEC 10-K filings, extracts structured facts via
LLM, ranks filings using deterministic rules, and produces a human-readable
markdown report with research-note summaries per filing.

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
| `--min-mcap-b` | `0.3` | Minimum market cap ($B) |
| `--max-mcap-b` | `10.0` | Maximum market cap ($B) |
| `--rank-by` | `mcap-asc` | Sort direction within band |
| `--top-n` | `500` | Maximum tickers in universe |
| `--days` | `365` | Filing recency window |
| `--sort` | `ctmg` | Sort order: c=chokepoint, t=trigger, m=margin, g=growth |
| `--tickers` | _(none)_ | Comma-separated tickers to run, bypassing universe filters |
| `--chokepoints` | `cross` | Chokepoints mode: `cross` = 3+ filers, `full` = all named entities |

Override anything from the command line. Pass `--refresh` to ignore caches
and rebuild from scratch (expensive — re-pays for all extraction/summary
LLM calls).

To run on specific tickers outside the S&P 1500 universe:

```
py sec_report.py --tickers RDW,ASTS
```

## One-time setup

```
pip install anthropic python-dotenv pydantic requests beautifulsoup4 lxml edgartools yfinance pandas
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
│   ├── universe.py          # ticker universe builder (supports --tickers for ad-hoc runs)
│   ├── watcher.py           # find recent 10-K filings
│   ├── batch.py             # batch process the watch queue
│   ├── extract.py           # LLM extraction (7 calls per filing)
│   ├── rank.py              # deterministic ranking (trigger rules in this file)
│   ├── digest.py            # markdown report generator
│   └── chokepoints.py       # cross-filing entity aggregation (no LLM cost)
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
                  (ticker list filtered by mcap band, or specific tickers via --tickers)
   ↓
2. watcher.py     builds output/data/watch_queue.json
                  (10-K filings filed within --days)
   ↓
3. batch.py       fetches each filing's HTML, splits into sections,
                  writes output/filings/<TICKER>/<accession>/
                  + output/data/run_manifest.json
   ↓
4. extract.py     runs Claude Haiku 4.5 7× per filing,
                  writes output/data/extractions.json   ← $$ API calls
   ↓
5. rank.py        evaluates trigger conditions, scores chokepoint connectivity
                  and financial performance, sorts flat by --sort mode,
                  writes output/data/ranking.json
   ↓
6. digest.py      writes one research note per filing,
                  prepends chokepoints section to report,
                  writes output/sec_report.md           ← $$ API calls
```

Cost per full run from scratch (~500 filings): ~$60 in API spend, ~120
minutes wall time. Repeat runs are much cheaper because filings and
extractions are cached — only new tickers since the last run actually hit
the API.

## Chokepoints

`utils/chokepoints.py` runs automatically as part of every pipeline run.
It reads the already-extracted data and identifies named entities (companies)
across vendor, customer, and competitor disclosures. Two modes are available
via `--chokepoints`:

**`cross` (default)** — entities appearing in 3+ filings. Best for large runs
(500 filings) where the signal is structural patterns across the corpus.
Universal noise (Amazon, Google, Microsoft) is filtered by a denylist.

```
## Chokepoints across this run

Customers named by multiple filers:
- Walmart — 9 filers: FLO · FRPT · BRBR · SMPL · JBSS · CALM · ELF · PBH

Competitors named by multiple filers:
- Medtronic — 8 filers: LIVN · MMSI · TNDM · ICUI · AORT · CNMD · INSP · HAE
```

**`full`** — every named entity from every filing, no threshold. Best for
small targeted runs (`--tickers`) where you want to see the complete
competitive landscape and shared relationships between specific companies.

```
py sec_report.py --tickers RDW,ASTS --chokepoints full
```

```
## Entity index for this run

Competitors:
- Airbus — RDW
- Iridium — ASTS
- Rocket Lab — RDW · ASTS
- SpaceX — RDW · ASTS
```

Entities appearing in multiple tickers (like SpaceX above) reveal shared
competitive dynamics between the companies you are researching.

Chokepoints also influence reading order: filings connected to more
cross-corpus patterns surface first in the ranked list.

**No LLM calls. Free per run.** The analysis only aggregates data already
paid for during extraction.

Run standalone to probe the data interactively:

```
python utils/chokepoints.py              # default min=3
python utils/chokepoints.py --min 2     # catch rarer patterns
```

To tune which entities get filtered as noise, edit `_DENYLIST` and
`_ALIASES` at the top of `utils/chokepoints.py`.

## Triggers

Each filing is evaluated against a set of deterministic trigger conditions.
Triggers fire when the filing crosses specific factual thresholds — no
interpretation, no scoring weights.

**Major triggers** (high-severity events):
- Going concern language
- Material weakness in internal controls
- Restatement of prior-period financials
- Impairment of $500M or more
- Litigation with disclosed exposure of $100M or more
- Regulatory action by DOJ / FDA / FTC / FERC / SEC / EPA / Department of Commerce

**Minor triggers** (notable events):
- CEO or CFO change during fiscal year
- Auditor change during fiscal year
- Impairment of $100M to $500M
- Other regulatory action
- Pending litigation without disclosed exposure
- M&A activity worth $1B or more
- Planned capex of $1B or more for next fiscal year

## Sort order

Filings are ranked flat (no tiers) using the `--sort` flag. Default is `ctmg`:

| Code | Order |
|---|---|
| `ctmg` | chokepoint → trigger → margin → growth (default) |
| `ctgm` | chokepoint → trigger → growth → margin |
| `tcmg` | trigger → chokepoint → margin → growth |
| `tcgm` | trigger → chokepoint → growth → margin |

All four metrics sort descending (higher = earlier in the report). Financial
metrics (`m` and `g`) come from the `financial_performance` extraction and
default to 0 for filings that haven't been re-extracted yet.

To tune triggers, edit the detector functions in `utils/rank.py`. After
editing, re-run `py sec_report.py` — re-ranking is free, and the digest
will regenerate using cached extractions and summaries.

## Cadence

Re-run monthly with the default `--days 45`. The 45-day window catches any
filing from the past month plus a 15-day safety margin for SEC indexing
delays. Caching means each monthly run only does work on filings that are
actually new. The chokepoints corpus grows automatically with each run.

## Iteration is cheap

Once you've done one full run, these things are essentially free to iterate
on:

- **Trigger rules.** Edit `utils/rank.py`. Re-running re-ranks instantly.
- **Sort order.** Pass `--sort ctmg/ctgm/tcmg/tcgm` to change ranking priority. Free to try.
- **Digest layout / prompt.** Edit `utils/digest.py`. Add `--refresh` to
  regenerate all summaries (~$2-3 for a 500-filing run); without `--refresh`
  only new filings hit the API.
- **Universe parameters.** Different `--min-mcap-b`, `--max-mcap-b`,
  `--rank-by` produce different universes; cached data is reused for any
  filings that overlap.
- **Chokepoint tuning.** Edit `_DENYLIST` and `_ALIASES` in
  `utils/chokepoints.py`, or adjust `DEFAULT_MIN_MENTIONS`. Re-running
  costs nothing — chokepoints reads the existing extractions.

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
| 4. Extract | ~20 sec per filing | ~$0.12/filing |
| 5. Rank + Chokepoints | <1 sec total | — |
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
