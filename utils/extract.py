"""
extract.py — LLM extraction layer (Phase 4)
=============================================

For each ok-status filing, runs 6 LLM calls (one per category) against
Claude Haiku 4.5 to extract structured fact data, validates the JSON
returned by each call against a Pydantic schema, and writes ALL filings'
extractions into one consolidated file: output/extractions.json.

The consolidated file makes inspection and reset easy:
  - To see all extractions:   type output\\extractions.json
  - To force a clean re-run:  del output\\extractions.json

Usage:
    python extract.py DLTR                    # single filing
    python extract.py DLTR MRNA EWBC          # several filings
    python extract.py --all-ok                # all ok-status filings in run_manifest
    python extract.py --all-ok --max 8        # first 8 ok filings
    python extract.py --skip-cached --all-ok  # only extract tickers not yet done

Schema: see extraction_schema_v0.md for the full field-by-field spec.

Cost: roughly $0.10 per filing using Haiku 4.5 (6 calls per filing). 50
filings ≈ $5 of API spend.

Environment:
    ANTHROPIC_API_KEY — read from .env file in this directory, or from the
    process environment if not present in .env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from anthropic import Anthropic, APIError, BadRequestError
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError, Field

from fetch import OUTPUT_DIR, DATA_DIR, FILINGS_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 4096               # response budget
MAX_RETRIES_PER_CATEGORY = 2    # retry on JSON parse / validation failure
EXTRACTION_VERSION = "v0"

# Single consolidated output: one JSON file containing all extractions, keyed
# by ticker. Easier to inspect and reset than per-filing files scattered
# across subfolders. To force a clean re-run, just delete this file.
EXTRACTIONS_PATH = DATA_DIR / "extractions.json"

# Load .env from the project root (one level up from utils/).
load_dotenv(Path(__file__).parent.parent / ".env")


# ---------------------------------------------------------------------------
# Pydantic schemas — one per category
# ---------------------------------------------------------------------------
# These mirror extraction_schema_v0.md exactly. Pydantic gives us:
#   - Type validation on fields
#   - Default values when LLM omits a field
#   - Clear error messages when validation fails (used in retry loop)


class Segment(BaseModel):
    name: str
    description: str


class Geography(BaseModel):
    region: str
    materiality: str  # "primary" | "secondary" | "not_material"


class Concentration(BaseModel):
    description: str
    threshold_disclosed: Optional[float] = None


class OperationalSchema(BaseModel):
    segments: list[Segment] = Field(default_factory=list)
    product_categories: list[str] = Field(default_factory=list)
    geographies: list[Geography] = Field(default_factory=list)
    vendor_concentrations: list[Concentration] = Field(default_factory=list)
    customer_concentrations: list[Concentration] = Field(default_factory=list)
    supply_chain_geographic_dependencies: list[str] = Field(default_factory=list)


class CompetitiveLandscapeSchema(BaseModel):
    named_competitors: list[str] = Field(default_factory=list)
    competitor_categories: list[str] = Field(default_factory=list)
    stated_advantages: list[str] = Field(default_factory=list)
    stated_disadvantages: list[str] = Field(default_factory=list)


class FinancialQualitySchema(BaseModel):
    restatement_mentioned: bool = False
    going_concern_language: bool = False
    material_weakness_disclosed: bool = False
    disclosure_controls_effective: bool = True
    icfr_effective: bool = True
    non_gaap_measures_used: list[str] = Field(default_factory=list)


class GovernanceSchema(BaseModel):
    ceo_changed: bool = False
    cfo_changed: bool = False
    auditor_changed: bool = False
    auditor_name: Optional[str] = None
    board_changes_disclosed: list[str] = Field(default_factory=list)
    related_party_transactions_mentioned: bool = False


class ForwardLookingSchema(BaseModel):
    revenue_guidance_provided: bool = False
    revenue_guidance_usd: Optional[float] = None        # midpoint if range given, full dollars
    revenue_guidance_range_usd: Optional[list[float]] = None  # [low, high] full dollars
    eps_guidance_provided: bool = False
    eps_guidance_value: Optional[float] = None          # midpoint if range given, in dollars
    guidance_direction: Optional[str] = None            # "raised" | "lowered" | "initiated" | "maintained" | null
    capex_planned_usd: Optional[float] = None
    capex_range_usd: Optional[list[float]] = None  # [low, high]
    key_initiatives_named: list[str] = Field(default_factory=list)


class Litigation(BaseModel):
    description: str
    status: str  # "pending" | "settled" | "dismissed" | "appealed"
    financial_exposure_usd: Optional[float] = None


class RegulatoryAction(BaseModel):
    description: str
    agency: Optional[str] = None
    year: Optional[str] = None
    amount_usd: Optional[float] = None


class MAEvent(BaseModel):
    type: str  # acquisition|divestiture|merger|joint_venture|asset_purchase|asset_sale
    counterparty: Optional[str] = None
    amount_usd: Optional[float] = None
    status: str  # completed|pending|announced|terminated
    year: Optional[str] = None


class Impairment(BaseModel):
    description: str
    amount_usd: Optional[float] = None
    fiscal_year: Optional[str] = None


class EventsAndCatalystsSchema(BaseModel):
    material_litigation: list[Litigation] = Field(default_factory=list)
    regulatory_actions: list[RegulatoryAction] = Field(default_factory=list)
    ma_activity: list[MAEvent] = Field(default_factory=list)
    recent_impairments: list[Impairment] = Field(default_factory=list)
    significant_facility_events: list[str] = Field(default_factory=list)


class FinancialPerformanceSchema(BaseModel):
    revenue_current_usd: Optional[float] = None         # current fiscal year revenue, full dollars
    revenue_prior_usd: Optional[float] = None           # prior fiscal year revenue, full dollars
    revenue_growth_pct: Optional[float] = None          # computed or disclosed YoY growth %
    yoy_revenue_direction: Optional[str] = None         # "growth" | "decline" | "flat"
    gross_margin_pct: Optional[float] = None            # gross profit / revenue %
    operating_margin_pct: Optional[float] = None        # operating income / revenue %
    net_income_usd: Optional[float] = None              # full dollars, negative if loss
    free_cash_flow_usd: Optional[float] = None          # if explicitly disclosed, full dollars


# Maps the category name to:
#   - the Pydantic schema to validate the response
#   - which section files to feed the LLM
CATEGORY_CONFIG = {
    "operational":            (OperationalSchema,            ["item_1", "item_7"]),
    "competitive_landscape":  (CompetitiveLandscapeSchema,   ["item_1", "item_1a"]),
    "financial_quality":      (FinancialQualitySchema,       ["item_7", "item_9a"]),
    "governance":             (GovernanceSchema,             ["item_7", "item_9a", "item_1a"]),
    "forward_looking":        (ForwardLookingSchema,         ["item_1", "item_7"]),
    "events_and_catalysts":   (EventsAndCatalystsSchema,     ["item_1", "item_1a", "item_3", "item_7"]),
    "financial_performance":  (FinancialPerformanceSchema,   ["item_7"]),
}


# ---------------------------------------------------------------------------
# Per-category prompts
# ---------------------------------------------------------------------------
# Each prompt:
#   1. Frames the LLM as a parser, not a scorer
#   2. Specifies the exact field schema to fill
#   3. Tells the LLM to return ONLY JSON (no preamble, no markdown fences)
#   4. Specifies what counts as null/empty (the silence-is-signal rule)


def _section_block(sections_dir: Path, section_keys: list[str]) -> str:
    """Concatenate the requested section files into a single block for the prompt."""
    parts = []
    for key in section_keys:
        path = sections_dir / f"{key}.txt"
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            parts.append(f"=== {key} ===\n{text.strip()}\n")
    return "\n".join(parts) if parts else "(no sections available)"


PROMPTS = {
    "operational": """You are extracting structured operational facts from SEC 10-K sections. Return ONLY a JSON object with the exact shape below — no preamble, no markdown fences, no commentary.

EXTRACTION RULES:
- "segments": Reported business segments per the filing's segment disclosure. If filing says "we operate in one segment," return one entry. Do NOT invent segments from product categories.
- "product_categories": How the filing groups its own merchandise/services. Use the filing's exact category names where possible.
- "geographies": Where the company operates. materiality is one of: "primary", "secondary", "not_material" — based on filing's own characterization.
- "vendor_concentrations" / "customer_concentrations": threshold_disclosed is the % threshold the filing references (e.g. "no vendor over 10%") as a number, or null. Empty array if the filing is silent.
- "supply_chain_geographic_dependencies": Country-level sourcing concentrations EXPLICITLY stated. Sourcing/supply chain only — not sales geographies.

Output the JSON exactly in this shape:
{
  "segments": [{"name": "...", "description": "..."}],
  "product_categories": ["..."],
  "geographies": [{"region": "...", "materiality": "primary|secondary|not_material"}],
  "vendor_concentrations": [{"description": "...", "threshold_disclosed": 10}],
  "customer_concentrations": [{"description": "...", "threshold_disclosed": null}],
  "supply_chain_geographic_dependencies": ["..."]
}

When the filing is silent on a field, return [] (empty array). Do not paraphrase silence.""",

    "competitive_landscape": """You are extracting competitive landscape facts from SEC 10-K sections. Return ONLY a JSON object — no preamble, no markdown fences, no commentary.

EXTRACTION RULES:
- "named_competitors": Specific COMPANY NAMES that appear LITERALLY in the source text as competitors. The exact company name must be present in the text — search for it before adding it. Do NOT infer competitor names from category descriptions like "online retailers" or "e-commerce platforms." Do NOT add well-known industry players (Amazon, Walmart, etc.) unless they are named verbatim in the filing. Do NOT include companies named only as customers, suppliers, or partners. If you find yourself thinking "this filing competes with X" but X is not actually written in the text, leave it out. When in doubt, exclude. Empty array is the correct answer for most filings.
- "competitor_categories": Categories of competitors when no specific company names are given (this is most filings). Use the filing's own phrasing.
- "stated_advantages": Short phrases describing competitive advantages the filing claims for itself. Maximum ~10 words each. Capture substance, not marketing voice. Maximum 8 entries — pick the most distinctive.
- "stated_disadvantages": Same for stated competitive pressures or disadvantages the filing acknowledges. Maximum 8 entries.

Output exactly:
{
  "named_competitors": ["..."],
  "competitor_categories": ["..."],
  "stated_advantages": ["..."],
  "stated_disadvantages": ["..."]
}

Do not include statements that aren't explicitly competitive in nature. Do not infer competitor identities.""",

    "financial_quality": """You are extracting financial-reporting-quality signals from SEC 10-K sections. Return ONLY a JSON object — no preamble, no markdown fences, no commentary.

EXTRACTION RULES:
- "restatement_mentioned": true ONLY if the filing references a restatement of PRIOR-PERIOD financial statements. Adjustments and reclassifications during the normal course of reporting do NOT count.
- "going_concern_language": true ONLY if the filing contains explicit "substantial doubt about the entity's ability to continue as a going concern" or equivalent SEC-mandated language. Generic risk-factor concerns about liquidity do NOT count.
- "material_weakness_disclosed": true ONLY if Item 9A discloses a material weakness in internal control over financial reporting (PCAOB AS 2201 standard). "Significant deficiencies" do NOT count.
- "disclosure_controls_effective": Management's assessment per Item 9A. Default true. Set false only if filing affirmatively states controls were not effective.
- "icfr_effective": Internal Control over Financial Reporting effective per management's Item 9A assessment. Default true. Set false only if filing says otherwise.
- "non_gaap_measures_used": Names of non-GAAP measures the filing references in MD&A.

DEFINITION OF NON-GAAP: A non-GAAP measure ADJUSTS a GAAP figure (e.g., GAAP net income → "Adjusted Net Income"; GAAP operating income → "Adjusted Operating EBITDA"). Non-GAAP measures typically have words like "Adjusted," "Comparable," "Constant currency," "Free cash flow," or are operational metrics specific to the company (e.g., "Comparable store sales").

NOT NON-GAAP — DO NOT INCLUDE THESE: Standard ratios calculated from GAAP figures are NOT non-GAAP. Specifically EXCLUDE: gross profit, gross profit margin, gross margin, operating income, operating margin, operating income margin, net income, net income margin, plain EBITDA (when not explicitly labeled "Adjusted EBITDA"), revenue growth, return on equity, return on assets. These are GAAP-derived. If a measure does not contain "Adjusted," "Comparable," "Constant," "Free Cash Flow," or similar modifier indicating an adjustment to a GAAP figure, it is most likely GAAP and should be excluded.

When in doubt, exclude.

Output exactly:
{
  "restatement_mentioned": false,
  "going_concern_language": false,
  "material_weakness_disclosed": false,
  "disclosure_controls_effective": true,
  "icfr_effective": true,
  "non_gaap_measures_used": ["..."]
}""",

    "governance": """You are extracting governance facts from SEC 10-K sections. Return ONLY a JSON object — no preamble, no markdown fences, no commentary.

EXTRACTION RULES:
- "ceo_changed": true ONLY if a new individual was named CEO during the FISCAL YEAR COVERED BY THIS FILING. Promotions from within count. Announced future transitions (CEO has not yet started in the role) do NOT count for this filing. Past CEO transitions referenced for comparison purposes do NOT count.
- "cfo_changed": Same definition, for CFO.
- "auditor_changed": true ONLY if the audit firm differs from the prior fiscal year. Mergers between audit firms or name changes do NOT count.
- "auditor_name": Name of the audit firm signing Item 9A's attestation report. Examples: "KPMG LLP", "PricewaterhouseCoopers LLP", "Ernst & Young LLP", "Deloitte & Touche LLP". null if not stated.
- "board_changes_disclosed": Director departures, new appointments, or governance restructuring mentioned in the filing. Brief descriptions (~10-15 words each).
- "related_party_transactions_mentioned": true if the filing references material related-party transactions in the sections provided. Note: full disclosure typically lives in proxy statements, so this often is false even for filings that have such transactions.

Output exactly:
{
  "ceo_changed": false,
  "cfo_changed": false,
  "auditor_changed": false,
  "auditor_name": "...",
  "board_changes_disclosed": ["..."],
  "related_party_transactions_mentioned": false
}""",

    "forward_looking": """You are extracting forward-looking statement facts from SEC 10-K sections. Return ONLY a JSON object — no preamble, no markdown fences, no commentary.

EXTRACTION RULES:
- "revenue_guidance_provided": true ONLY if the filing provides explicit numerical revenue guidance for the upcoming fiscal year. Generic statements like "we expect modest growth" do NOT count.
- "revenue_guidance_usd": The midpoint of guidance range in full USD (e.g. 2500000000 for $2.5B). If a single figure is given, use it. null if revenue_guidance_provided is false.
- "revenue_guidance_range_usd": [low, high] in full USD if a range is disclosed (e.g. [2400000000, 2600000000]). null if no range or no guidance.
- "eps_guidance_provided": Same standard, for EPS or net income per share.
- "eps_guidance_value": Midpoint EPS guidance in dollars per share (e.g. 3.45). null if eps_guidance_provided is false.
- "guidance_direction": Whether the company raised, lowered, initiated, or maintained guidance versus the prior year's guidance or consensus. One of: "raised" | "lowered" | "initiated" | "maintained" | null. Only populate if the filing explicitly states the direction.
- "capex_planned_usd": Total planned capital expenditures for the upcoming fiscal year, in USD (not millions — full dollar number, e.g. 1100000000 for $1.1 billion). If a range is given, use the midpoint. Round to nearest million. null if no specific planned amount stated.
- "capex_range_usd": [low, high] in USD as full dollar numbers (e.g. [1100000000, 1200000000]). null if range not disclosed.
- "key_initiatives_named": Specific named strategic initiatives or projects the filing identifies for the coming fiscal year. Maximum ~12 words each. Generic phrases like "drive shareholder value" or "execute our strategy" do NOT count. Initiatives must be concrete enough to identify (e.g., "Phoenix distribution center opening spring 2026"). MAXIMUM 5 ENTRIES — pick the most important. Prioritize initiatives that are: (1) tied to a specific date or fiscal year, (2) tied to a specific dollar amount, or (3) tied to a specific named project/facility. If you have more than 5 candidates, drop the vague strategic language ones and keep the concrete ones.

Output exactly:
{
  "revenue_guidance_provided": false,
  "revenue_guidance_usd": null,
  "revenue_guidance_range_usd": null,
  "eps_guidance_provided": false,
  "eps_guidance_value": null,
  "guidance_direction": null,
  "capex_planned_usd": null,
  "capex_range_usd": null,
  "key_initiatives_named": ["..."]
}""",

    "events_and_catalysts": """You are extracting specific events and catalysts from SEC 10-K sections. Return ONLY a JSON object — no preamble, no markdown fences, no commentary.

EXTRACTION RULES:
- "material_litigation": Specific legal proceedings the filing identifies as potentially material. status: "pending"|"settled"|"dismissed"|"appealed". financial_exposure_usd: disclosed amount or null. Generic risk-factor language about "potential litigation" does NOT count.
- "regulatory_actions": Specific regulatory actions, investigations, or fines NAMING the agency involved. Routine compliance discussions do NOT count.
- "ma_activity": Mergers, acquisitions, divestitures, or significant joint ventures. type: "acquisition"|"divestiture"|"merger"|"joint_venture"|"asset_purchase"|"asset_sale". status: "completed"|"pending"|"announced"|"terminated". TEMPORAL SCOPE: only include events that (a) occurred in the most recent 3 fiscal years, or (b) are currently pending or announced. EXCLUDE historical structural relationships, joint ventures formed many years ago, or acquisitions older than 3 fiscal years that are only mentioned for context.
- "recent_impairments": MATERIAL impairments to goodwill, intangibles, or long-lived assets. INCLUDE impairments from prior fiscal years if mentioned in the sections provided (often referenced in MD&A comparisons). amount_usd in full dollar numbers. MATERIALITY: include only impairments that are EITHER (a) $50 million or larger in absolute amount, OR (b) explicitly identified as a goodwill impairment or major intangible asset impairment. EXCLUDE routine PP&E or store-level write-downs below $50M (these are operational noise).
- "significant_facility_events": Specific events affecting key facilities — fires, natural disasters, closures, major openings. Brief descriptions (~10-15 words each).

Monetary amounts: ALL amount_usd / financial_exposure_usd values are full USD numbers (e.g. 25000000 for $25 million). NOT millions, NOT abbreviated.

Output exactly:
{
  "material_litigation": [{"description": "...", "status": "pending|settled|dismissed|appealed", "financial_exposure_usd": null}],
  "regulatory_actions": [{"description": "...", "agency": "...", "year": "...", "amount_usd": null}],
  "ma_activity": [{"type": "acquisition|divestiture|merger|joint_venture|asset_purchase|asset_sale", "counterparty": "...", "amount_usd": null, "status": "completed|pending|announced|terminated", "year": "..."}],
  "recent_impairments": [{"description": "...", "amount_usd": null, "fiscal_year": "..."}],
  "significant_facility_events": ["..."]
}""",

    "financial_performance": """You are extracting financial performance figures from SEC 10-K MD&A sections. Return ONLY a JSON object — no preamble, no markdown fences, no commentary.

EXTRACTION RULES:
- "revenue_current_usd": Total revenue / net sales for the CURRENT fiscal year in full USD (e.g. 1250000000 for $1.25B). Use the top-line consolidated revenue figure. null if not stated.
- "revenue_prior_usd": Same figure for the PRIOR fiscal year (the comparison year in MD&A tables). null if not stated.
- "revenue_growth_pct": Year-over-year revenue change as a percentage. Positive = growth, negative = decline (e.g. -8.3 for an 8.3% decline). If the filing states the % directly, use it. Otherwise compute from current and prior: ((current - prior) / prior) * 100. Round to one decimal. null if either revenue figure is missing.
- "yoy_revenue_direction": "growth" if revenue_growth_pct > 1%, "decline" if < -1%, "flat" if between -1% and 1%. null if revenue_growth_pct is null.
- "gross_margin_pct": Gross profit divided by revenue, as a percentage. Use the current fiscal year figure. Round to one decimal. null if not determinable from the sections.
- "operating_margin_pct": Operating income divided by revenue, as a percentage. Current fiscal year. Round to one decimal. null if not determinable.
- "net_income_usd": Net income (or net loss, as a negative number) for the current fiscal year in full USD. null if not stated.
- "free_cash_flow_usd": Free cash flow ONLY if the filing explicitly discloses it by name or defines it (e.g., "We define free cash flow as operating cash flow less capex"). Do NOT compute it yourself. null if not explicitly disclosed.

IMPORTANT: Use CONSOLIDATED figures only. For multi-segment companies, use the total company line, not segment-level data. All amounts are full USD integers (e.g. 500000000, not 500 or 500M). Percentages are plain numbers (e.g. 34.2, not "34.2%").

Output exactly:
{
  "revenue_current_usd": null,
  "revenue_prior_usd": null,
  "revenue_growth_pct": null,
  "yoy_revenue_direction": null,
  "gross_margin_pct": null,
  "operating_margin_pct": null,
  "net_income_usd": null,
  "free_cash_flow_usd": null
}""",
}


# ---------------------------------------------------------------------------
# Anthropic API call with retry
# ---------------------------------------------------------------------------

def _strip_json_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` despite our instructions."""
    text = text.strip()
    if text.startswith("```"):
        # Strip first line (```json or ```) and last line (```)
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _call_category(client: Anthropic, category: str, sections_dir: Path) -> dict:
    """
    Run one LLM call for one category. Returns the validated dict (already
    schema-conformant). Raises on unrecoverable failure.
    """
    schema_class, section_keys = CATEGORY_CONFIG[category]
    system_prompt = PROMPTS[category]
    section_text = _section_block(sections_dir, section_keys)

    user_message = (
        f"Sections to extract from:\n\n{section_text}\n\n"
        f"Return the JSON object now. ONLY the JSON object, nothing else."
    )

    last_error = ""
    for attempt in range(MAX_RETRIES_PER_CATEGORY + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text
            cleaned = _strip_json_fences(raw)

            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError as e:
                last_error = f"JSON decode failed: {e}. First 200 chars: {cleaned[:200]!r}"
                if attempt < MAX_RETRIES_PER_CATEGORY:
                    # Tell the LLM what went wrong on retry
                    user_message = (
                        f"Sections to extract from:\n\n{section_text}\n\n"
                        f"Your previous response was not valid JSON. Error: {e}. "
                        f"Return the JSON object now. ONLY the JSON object, no markdown, no commentary."
                    )
                    continue
                raise RuntimeError(last_error)

            try:
                validated = schema_class(**parsed)
                # mode="json" gives us a JSON-safe dict (no Pydantic objects nested)
                return validated.model_dump(mode="json")
            except ValidationError as e:
                last_error = f"Schema validation failed: {e}"
                if attempt < MAX_RETRIES_PER_CATEGORY:
                    user_message = (
                        f"Sections to extract from:\n\n{section_text}\n\n"
                        f"Your previous response failed schema validation: {e}. "
                        f"Return the JSON object now. ONLY the JSON object."
                    )
                    continue
                raise RuntimeError(last_error)

        except (APIError, BadRequestError) as e:
            last_error = f"API error: {e}"
            if attempt < MAX_RETRIES_PER_CATEGORY:
                time.sleep(2 ** attempt)
                continue
            raise

    raise RuntimeError(f"Exhausted retries for {category}. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Per-filing orchestration
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    ticker: str
    accession: str
    success: bool
    categories_done: list[str]
    extraction: Optional[dict] = None  # the full JSON for this filing on success
    error: Optional[str] = None
    duration_seconds: float = 0.0


def _load_extractions() -> dict:
    """
    Load the consolidated extractions file. Returns an empty dict if the
    file doesn't exist or is malformed (so a corrupt save can't permanently
    block re-runs).
    """
    if not EXTRACTIONS_PATH.exists():
        return {}
    try:
        payload = json.loads(EXTRACTIONS_PATH.read_text(encoding="utf-8"))
        return payload.get("extractions", {})
    except (json.JSONDecodeError, OSError) as e:
        print(f"[warn] Could not read {EXTRACTIONS_PATH} ({e}); starting fresh.")
        return {}


def _save_extractions(extractions: dict) -> None:
    """
    Write the full extractions dict to a single consolidated JSON file.
    Called after each filing completes so partial progress is preserved
    if a long run is interrupted.
    """
    payload = {
        "extraction_version": EXTRACTION_VERSION,
        "model": MODEL,
        "last_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "filing_count": len(extractions),
        "extractions": extractions,
    }
    EXTRACTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXTRACTIONS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _find_filing_dir(ticker: str) -> Optional[Path]:
    """Locate the most-recent accession folder for a ticker under filings/."""
    ticker_dir = FILINGS_DIR / ticker.upper()
    if not ticker_dir.exists():
        return None
    # Each subdir is an accession. Return the lexically latest, which for
    # SEC accessions corresponds chronologically.
    accessions = sorted([p for p in ticker_dir.iterdir() if p.is_dir()])
    return accessions[-1] if accessions else None


def _extract_one(client: Anthropic, ticker: str) -> ExtractionResult:
    """
    Run all 6 category calls for a single filing. Returns an ExtractionResult
    with the full extraction dict on success. Does NOT write to disk —
    persistence is handled by main() so all extractions land in one file.
    """
    start = time.monotonic()
    filing_dir = _find_filing_dir(ticker)

    if filing_dir is None:
        return ExtractionResult(
            ticker=ticker.upper(),
            accession="",
            success=False,
            categories_done=[],
            error=f"No filing directory found at {FILINGS_DIR / ticker.upper()}",
        )

    sections_dir = filing_dir / "sections"
    if not sections_dir.exists() or not any(sections_dir.iterdir()):
        return ExtractionResult(
            ticker=ticker.upper(),
            accession=filing_dir.name,
            success=False,
            categories_done=[],
            error=f"No section files at {sections_dir}",
        )

    print(f"  [{ticker.upper()}] extracting from {filing_dir.name}...")
    categories_done: list[str] = []
    extraction = {
        "ticker": ticker.upper(),
        "accession": filing_dir.name,
        "extraction_version": EXTRACTION_VERSION,
        "model": MODEL,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        for category in CATEGORY_CONFIG:
            t0 = time.monotonic()
            result = _call_category(client, category, sections_dir)
            extraction[category] = result
            categories_done.append(category)
            elapsed = time.monotonic() - t0
            print(f"      ✓ {category:<24} ({elapsed:.1f}s)")
    except Exception as e:
        return ExtractionResult(
            ticker=ticker.upper(),
            accession=filing_dir.name,
            success=False,
            categories_done=categories_done,
            error=f"{type(e).__name__}: {e}",
            duration_seconds=time.monotonic() - start,
        )

    return ExtractionResult(
        ticker=ticker.upper(),
        accession=filing_dir.name,
        success=True,
        categories_done=categories_done,
        extraction=extraction,
        duration_seconds=time.monotonic() - start,
    )


# ---------------------------------------------------------------------------
# Top-level CLI
# ---------------------------------------------------------------------------

def _resolve_targets(args: argparse.Namespace) -> list[str]:
    """Figure out which tickers to process based on CLI args."""
    if args.all_ok:
        manifest_path = DATA_DIR / "run_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"--all-ok requires {manifest_path}; run batch.py first")
        manifest = json.loads(manifest_path.read_text())
        tickers = [
            f["ticker"]
            for f in manifest.get("filings", [])
            if f.get("status") == "ok"
        ]
        if args.max is not None:
            tickers = tickers[: args.max]
        return tickers
    if args.tickers:
        return args.tickers
    raise SystemExit("Provide tickers as positional args or use --all-ok")


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM extraction on filings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python extract.py DLTR\n"
            "  python extract.py DLTR MRNA EWBC\n"
            "  python extract.py --all-ok\n"
            "  python extract.py --all-ok --max 8\n"
        ),
    )
    parser.add_argument("tickers", nargs="*",
                        help="Ticker symbols to extract (positional)")
    parser.add_argument("--all-ok", action="store_true",
                        help="Process all ok-status filings from run_manifest.json")
    parser.add_argument("--max", type=int, default=None,
                        help="With --all-ok, cap at N filings")
    parser.add_argument("--skip-cached", action="store_true",
                        help="Skip tickers already present in extractions.json")

    args = parser.parse_args()
    targets = _resolve_targets(args)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "ANTHROPIC_API_KEY not found. Create a .env file in this directory "
            "containing: ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = Anthropic(api_key=api_key)

    # Load existing extractions (if any). New extractions get added to this
    # dict; the whole thing gets re-saved after each filing finishes so
    # partial progress survives an interrupted run.
    extractions = _load_extractions()
    if extractions:
        print(f"[extract] Loaded {len(extractions)} existing extractions from "
              f"{EXTRACTIONS_PATH.name}")

    print(f"[extract] Processing {len(targets)} filings with model {MODEL}")
    print(f"[extract] Each filing = 7 LLM calls (one per category)")
    print()

    results: list[ExtractionResult] = []
    for i, ticker in enumerate(targets, 1):
        ticker_upper = ticker.upper()
        print(f"[{i}/{len(targets)}] {ticker_upper}")

        if args.skip_cached and ticker_upper in extractions:
            print(f"      → already in extractions.json, skipping")
            print()
            results.append(ExtractionResult(
                ticker=ticker_upper,
                accession=extractions[ticker_upper].get("accession", ""),
                success=True,
                categories_done=list(CATEGORY_CONFIG.keys()),
                extraction=extractions[ticker_upper],
            ))
            continue

        result = _extract_one(client, ticker)
        results.append(result)
        if result.success:
            extractions[ticker_upper] = result.extraction
            _save_extractions(extractions)
            print(f"      → saved to {EXTRACTIONS_PATH.name} "
                  f"({result.duration_seconds:.1f}s total)")
        else:
            print(f"      ✗ FAILED: {result.error}")
        print()

    ok = sum(1 for r in results if r.success)
    failed = len(results) - ok
    print(f"[extract] Done. {ok} succeeded, {failed} failed.")
    print(f"[extract] Total extractions in file: {len(extractions)}")
    print(f"[extract] Output: {EXTRACTIONS_PATH}")
    if failed:
        print()
        print("Failures:")
        for r in results:
            if not r.success:
                print(f"  {r.ticker}: {r.error}")


if __name__ == "__main__":
    main()