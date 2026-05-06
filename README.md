# sec_bot

A research bot that fetches SEC filings, extracts structured facts via LLM,
groups them into priority tiers using deterministic rules, and produces
a human-readable digest with research-note summaries. See the white paper
(`sec_filing_extraction_whitepaper.pdf`) for the full design rationale.

## Files

| File | Purpose |
|---|---|
| `fetch.py` | Fetches and parses a single ticker's most recent 10-K. The plumbing for one filing, end-to-end. Also defines the `Anomaly` dataclass and the section-size heuristics. |
| `universe.py` | Builds a ranked list of US-listed tickers from Wikipedia + yfinance + SEC ticker map. Supports seed selection (sp500 / sp1500 / all) and market-cap band filtering. |
| `watcher.py` | Given the universe, finds all 10-K filings within a recency window. |
| `batch.py` | Processes the watch queue: downloads each filing, parses it, classifies status, writes a run manifest. |
| `extract.py` | LLM extraction (Phase 4). Calls Claude Haiku 4.5 to extract structured facts (operational, competitive, financial quality, governance, forward-looking, events) from each filing. Writes `output/extractions.json`. |
| `rank.py` | Tier-based ranker (Phase 7). Reads `extractions.json`, evaluates trigger conditions, assigns each filing to **critical**, **notable**, or **quiet** tier. Writes `output/ranking.json`. |
| `digest.py` | Human-readable digest (Phase 8). Reads `extractions.json` + `ranking.json`, calls Haiku 4.5 once per filing for a 2-3 paragraph research note, writes `output/digest.md`. Summaries are cached so re-running is free. |

## One-time setup

```
pip install requests beautifulsoup4 lxml edgartools yfinance pandas anthropic python-dotenv pydantic
```

For the LLM stages (`extract.py` and `digest.py`), create a `.env` file
containing your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Edit the `USER_AGENT` at the top of `fetch.py`, `universe.py`, and
`watcher.py` if you want to use your own contact details (SEC requires a
real-looking User-Agent).

---

## Tutorial: how each script works and how to use them

The pipeline has six conceptual stages but they live in five scripts you
actually run (universe and watcher are folded into batch when you use
`--pipeline`). Each script reads the previous stage's output from disk
and writes its own output to disk, so you can re-run any stage in
isolation.

### Stage 1: `universe.py` — pick the candidate tickers

What it does: pulls a list of US-listed companies from Wikipedia
(S&P 500, 400, 600 indexes), resolves each one's SEC CIK number, fetches
their market caps from yfinance, and filters/ranks them based on your
flags.

Output: `output/universe.json`

```
python universe.py                        # default: sp1500, top 500 by mcap
python universe.py --seed sp500           # only S&P 500 mega-caps
python universe.py --min-mcap-b 0.5 --max-mcap-b 10 --top-n 200
                                          # mid-/small-cap band
python universe.py --rank-by mcap-asc --top-n 100
                                          # smallest first within band
```

Common flags:
- `--seed` — `sp500`, `sp1500` (default), or `all` (every SEC filer)
- `--top-n N` — cap on universe size
- `--min-mcap-b` / `--max-mcap-b` — market cap band in billions USD
- `--rank-by` — `mcap-desc` (default) or `mcap-asc`
- `--refresh-cache` — force re-fetch of the SEC ticker map

### Stage 2: `watcher.py` — find recent 10-K filings

What it does: walks the universe, hits SEC's submissions endpoint for each
ticker, and collects every 10-K filed within the recency window.

Output: `output/watch_queue.json`

```
python watcher.py                         # default: 10-K, last 90 days
python watcher.py --days 365              # one full year
python watcher.py --days 90 --limit 50    # cap the queue size
```

### Stage 3: `batch.py` — fetch and section-split filings

What it does: downloads each filing's HTML, splits it into canonical
sections (Item 1, Item 1A, Item 7, Item 8, Item 9A), runs anomaly
detection, and writes a per-filing manifest.

Output: `output/run_manifest.json` plus `output/filings/<TICKER>/<accession>/`
for each filing.

```
python batch.py                           # process the queue
python batch.py --max 20                  # only first 20 filings
python batch.py --skip-cached             # skip already-processed filings
python batch.py --pipeline                # run universe → watcher → batch in one shot
```

`--pipeline` is the most useful — it runs all three earlier stages with
one command. It accepts every flag from `universe.py` and `watcher.py`:

```
python batch.py --pipeline \
    --seed sp1500 \
    --min-mcap-b 0.3 --max-mcap-b 30 \
    --top-n 100 \
    --days 365
```

### Stage 4: `extract.py` — LLM-extract structured facts

What it does: for each ok-status filing in the run manifest, calls Claude
Haiku 4.5 six times (one per category) to extract structured facts about
operations, competitive landscape, financial quality, governance,
forward-looking statements, and events/catalysts.

Cost: ~$0.10 per filing, ~17 seconds per filing.

Output: `output/extractions.json`

```
python extract.py --all-ok                # extract from all ok-status filings
python extract.py STZ MKSI                # extract specific tickers
python extract.py --max 10 --all-ok       # cap at 10 filings
python extract.py --skip-cached --all-ok  # skip already-extracted tickers
```

### Stage 5: `rank.py` — tier-based ranking

What it does: reads `extractions.json` and assigns each filing to one of
three tiers based on which trigger conditions fired.

Output: `output/ranking.json`

```
python rank.py                            # show all filings, all tiers
python rank.py --tier critical            # only critical to terminal
python rank.py --tier critical,notable    # critical + notable to terminal
```

The `ranking.json` file always contains every filing regardless of
`--tier`. The flag only affects what's printed to your terminal.

**Tier definitions:**

A filing lands in **critical** if any of these triggers fired:
- Going concern language present
- Material weakness in internal controls disclosed
- Restatement of prior-period financials
- Impairment of $500M or more
- Litigation with disclosed exposure of $100M or more
- Regulatory action by DOJ / FDA / FTC / FERC / SEC / EPA / Department of Commerce

A filing lands in **notable** if (it's not already critical and) any of
these fired:
- CEO change during fiscal year
- CFO change during fiscal year
- Auditor change during fiscal year
- Impairment of $100M to $500M
- Other regulatory action (non-significant agency)
- Pending litigation without disclosed exposure
- M&A activity worth $1B or more
- Planned capex of $1B or more for next fiscal year

A filing lands in **quiet** otherwise.

Within a tier, filings are sorted by trigger count (more first), then
alphabetically by ticker.

To tune what triggers what tier, edit the trigger detector functions in
`rank.py`. Re-run `python rank.py` (1 second, free) and `python digest.py`
(seconds, free if cache is valid) to see the effect.

### Stage 6: `digest.py` — human-readable report

What it does: reads `extractions.json` + `ranking.json`, generates a
2-3 paragraph research note per filing using Haiku 4.5, and writes a
markdown report.

Cost: ~$0.02-0.03 per filing for the summary. ~$2-3 for 100 filings.

Output: `output/digest.md` plus `output/digest_summaries.json` (cache).

```
python digest.py                          # use cache, generate missing
python digest.py --refresh-summaries      # regenerate everything (paid)
python digest.py --no-llm                 # template-only, free, less polished
```

Each filing's digest entry contains:
- Header with tier label: `### DOW [CRITICAL]`
- Clickable SEC link straight to the 10-K on EDGAR
- Triggers line listing the rules that fired
- 2-3 paragraph LLM-written research note (business / why-it-matters / forward)

Summaries are cached by `(ticker, accession)`, so re-running digest.py
without changing the underlying extractions costs nothing — only NEW or
CHANGED filings hit the API.

---

## Full pipeline (typical real run)

```
python batch.py --pipeline --seed sp1500 --min-mcap-b 0.3 --max-mcap-b 30 --top-n 100 --days 365
python extract.py --all-ok
python rank.py
python digest.py
```

Expected wall time: ~50 minutes.
Expected cost: ~$12-13 in API spend (~$10 extract + ~$2-3 digest).

After the initial run, iterating on tier rules or digest layout is free:
- Edit `rank.py` rules → `python rank.py` → `python digest.py` (cached summaries, ~1 second)
- Edit digest layout → `python digest.py` (cached summaries, ~1 second)

## Output layout

```
output/
├── universe.json             # ticker list with CIK + market cap
├── watch_queue.json          # pending filings to fetch
├── run_manifest.json         # batch results: counts, anomalies, per-filing details
├── extractions.json          # LLM-extracted structured facts
├── ranking.json              # tier assignments + triggers per filing
├── digest.md                 # human-readable markdown report
├── digest_summaries.json     # cache of LLM-written summaries
├── _cache/
│   └── sec_ticker_map.json   # cached SEC ticker→CIK map
└── filings/                  # per-filing data tucked away
    └── <TICKER>/
        └── <accession>/
            ├── raw.htm       # original 10-K HTML
            ├── cleaned.txt   # plain text (sanity reference)
            ├── manifest.json # per-filing status + anomalies
            └── sections/     # split per-section text files
```

## Status meanings (in run_manifest.json)

- **ok** — sections extracted cleanly. Safe for downstream LLM extraction.
- **flagged** — sections extracted, but at least one warn-level anomaly
  fired. Section data is likely unreliable.
- **failed** — couldn't extract any sections, or download/parse threw an
  exception. Skip downstream.

## Anomalies

Every filing's `manifest.json` includes a structured `anomalies` array.
Each anomaly is a record:

```json
{
  "section": "item_8",
  "kind": "stub",
  "severity": "info",
  "observed_value": 27,
  "expected_range": [500, 60000],
  "message": "item_8 = 27 words is a STUB..."
}
```

### Anomaly kinds

| Kind | Severity | Meaning |
|---|---|---|
| `too_small` | warn | Section is below expected lower bound and contains no cross-reference language. Likely a real splitter miss. |
| `too_large` | warn | Section is above expected upper bound. Usually means the splitter walked off the end and merged content from the next section. |
| `stub` | info | Section is below expected lower bound BUT contains explicit cross-reference language. The company filed the actual content as a separate exhibit; the section being short is intentional. |

## Performance notes

| Stage | Time | Cost |
|---|---|---|
| `universe.py` (sp1500) | ~6 min | — |
| `watcher.py` | ~1 min per 100 tickers | — |
| `batch.py` | ~5-10 sec per filing | — |
| `extract.py` | ~17 sec per filing | ~$0.10/filing |
| `rank.py` | <1 sec total | — |
| `digest.py` (initial) | ~3-5 sec per filing | ~$0.025/filing |
| `digest.py` (cached) | <1 sec total | — |

## Notes

- Section detection uses [edgartools](https://github.com/dgunning/edgartools)
  under the hood. We just feed it raw HTML.
- The current splitter is wired for **10-K** filings only. 10-Q and 8-K
  support is intentionally deferred (different item layouts).
- All stages cache aggressively: re-running is cheap, parsed filings are
  reused. Extraction results are keyed by ticker; digest summaries by
  `(ticker, accession)`.
