"""Scheme Agent — Step 6

For each top-10 ranked park, this agent:
  1. Matches applicable Central government schemes (PLI, MSME, MUDRA, etc.)
  2. Matches applicable State government schemes for the park's state
  3. Fetches park-specific incentives scraped in Step 3
  4. Calculates the subsidy stack: gross investment - total subsidies = net cost
  5. Returns a full scheme breakdown per park
"""

import json
import os
from typing import Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Load static schemes dataset
with open(os.path.join(_DATA_DIR, 'schemes.json'), encoding='utf-8') as f:
    SCHEMES_DB: Dict = json.load(f)

_GEMINI_MODEL = "gemini-2.0-flash"

# State name normalisation (UI slugs → JSON keys in schemes.json)
_STATE_SLUG_MAP = {
    "GUJARAT":        "gujarat",
    "KARNATAKA":      "karnataka",
    "TAMIL NADU":     "tamil_nadu",
    "MAHARASHTRA":    "maharashtra",
    "TELANGANA":      "telangana",
    "UTTAR PRADESH":  "uttar_pradesh",
    "ANDHRA PRADESH": "andhra_pradesh",
    "RAJASTHAN":      "rajasthan",
    "HARYANA":        "haryana",
    "PUNJAB":         "punjab",
}


def _state_key(state_raw: str) -> str:
    normalised = state_raw.strip().upper()
    return _STATE_SLUG_MAP.get(normalised, normalised.lower().replace(" ", "_"))


# ─────────────────────────────────────────────────────────────────────────────
# Central scheme matching
# ─────────────────────────────────────────────────────────────────────────────

def _match_central_schemes(sector: str, investment_inr: float,
                            is_msme: bool, is_startup: bool) -> List[Dict]:
    """Filter central schemes from schemes.json that apply to this profile."""
    matched = []
    investment_cr = investment_inr / 1e7  # convert ₹ to ₹ Crores

    for scheme in SCHEMES_DB.get("central", []):
        # Sector check
        scheme_sectors = scheme.get("sector", "all")
        if scheme_sectors != "all":
            sector_match = any(
                s.lower() in sector.lower() or sector.lower() in s.lower()
                for s in scheme_sectors
            )
            if not sector_match:
                continue

        # Investment minimum check
        min_inv = float(scheme.get("min_investment_cr", 0) or 0)
        if investment_cr < min_inv:
            continue

        # MSME / startup eligibility hints
        eligibility = " ".join(scheme.get("eligibility", [])).lower()
        if "msme" in eligibility and not is_msme:
            pass   # still include, just note it
        if "startup" in eligibility and not is_startup:
            pass   # still include

        matched.append({
            "name":         scheme["name"],
            "benefit":      scheme.get("benefit", ""),
            "eligibility":  scheme.get("eligibility", []),
            "url":          scheme.get("url", ""),
            "level":        "Central",
        })

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# State scheme matching
# ─────────────────────────────────────────────────────────────────────────────

def _match_state_schemes(state_raw: str, sector: str) -> List[Dict]:
    """Filter state-level schemes from schemes.json."""
    key = _state_key(state_raw)
    state_schemes = SCHEMES_DB.get("state", {}).get(key, [])
    matched = []

    for scheme in state_schemes:
        scheme_sectors = scheme.get("sector", "all")
        if scheme_sectors != "all":
            sector_match = any(
                s.lower() in sector.lower() or sector.lower() in s.lower()
                for s in scheme_sectors
            )
            if not sector_match:
                continue
        matched.append({
            "name":         scheme["name"],
            "benefit":      scheme.get("benefit", ""),
            "eligibility":  scheme.get("eligibility", []),
            "url":          scheme.get("url", ""),
            "level":        "State",
        })

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Gemini: live scheme enrichment + subsidy calculation
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_scheme_research(
    park_name: str,
    state: str,
    sector: str,
    investment_inr: float,
    static_schemes: List[Dict],
    park_incentives: List[str],
) -> Dict:
    """
    Use Gemini with Google Search grounding to:
      - Find any recent/additional schemes for this state + sector
      - Estimate monetary value of the subsidy stack
      - Calculate net investment after subsidies
    """
    investment_cr = investment_inr / 1e7

    known_schemes_text = "\n".join(
        f"- {s['name']}: {s['benefit']}"
        for s in static_schemes
    ) or "None found in local DB"

    park_incentives_text = "\n".join(f"- {inc}" for inc in park_incentives) or "Unknown"

    prompt = f"""
You are an expert on Indian government industrial investment schemes, subsidies, and incentives.

INVESTOR PROFILE:
- Park: {park_name}
- State: {state}
- Sector: {sector}
- Gross Investment: ₹{investment_cr:.1f} Crore

KNOWN APPLICABLE SCHEMES (from our database):
{known_schemes_text}

PARK-SPECIFIC INCENTIVES (from park's own data):
{park_incentives_text}

Your tasks:
1. Confirm and expand the list of applicable Central + State schemes for this investor profile.
2. Search for any NEW or additional schemes active in {state} for the {sector} sector in 2024-2025.
3. Estimate the TOTAL monetary value of subsidies (in ₹ Crore) the investor can realistically claim.
4. Calculate NET investment after subsidies.

Return ONLY a valid JSON object:
{{
  "additional_schemes": [
    {{"name": "Scheme Name", "benefit": "What it provides", "estimated_value_cr": 0.0, "level": "Central/State"}}
  ],
  "total_subsidy_estimate_cr": 0.0,
  "net_investment_cr": {investment_cr:.1f},
  "subsidy_percentage": 0.0,
  "subsidy_breakdown": [
    {{"scheme": "Scheme Name", "estimated_value_cr": 0.0, "type": "Capital/Tax/Land/Power"}}
  ],
  "key_insight": "One sentence on the most impactful scheme for this investor"
}}

Use realistic estimates based on scheme rules. Return ONLY the JSON.
"""

    try:
        model = genai.GenerativeModel(
            model_name=_GEMINI_MODEL,
            tools="google_search_retrieval",
        )
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1500,
                response_mime_type="application/json",
            ),
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        # Fallback: rough estimate
        est = investment_cr * 0.15
        return {
            "additional_schemes": [],
            "total_subsidy_estimate_cr": round(est, 2),
            "net_investment_cr": round(investment_cr - est, 2),
            "subsidy_percentage": 15.0,
            "subsidy_breakdown": [],
            "key_insight": f"Estimated ~15% subsidy stack typical for {sector} in {state}.",
            "_error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main: process schemes for all top-10 parks
# ─────────────────────────────────────────────────────────────────────────────

def process_schemes_for_parks(
    top10_parks: List[Dict],
    requirements: Dict,
    progress_callback=None,
) -> List[Dict]:
    """
    For each park in top10_parks, attach a 'schemes' dict containing:
      - central_schemes: List[Dict]
      - state_schemes:   List[Dict]
      - park_incentives: List[str]
      - gemini_analysis: Dict (subsidy stack, net cost, breakdown)

    Returns the enriched top10 list.
    """
    sector       = requirements.get("sector", "manufacturing")
    investment   = float(requirements.get("investment_inr") or 0)
    is_msme      = bool(requirements.get("is_msme", False))
    is_startup   = bool(requirements.get("is_startup", False))

    for i, park in enumerate(top10_parks):
        name  = park.get("name", f"Park #{i+1}")
        state = park.get("state", "")

        if progress_callback:
            progress_callback(i, len(top10_parks), name)

        central_schemes = _match_central_schemes(sector, investment, is_msme, is_startup)
        state_schemes   = _match_state_schemes(state, sector)
        park_incentives = park.get("incentives") or []

        all_static = central_schemes + state_schemes

        gemini_result = _gemini_scheme_research(
            park_name      = name,
            state          = state,
            sector         = sector,
            investment_inr = investment,
            static_schemes = all_static,
            park_incentives= park_incentives,
        )

        park["schemes"] = {
            "central_schemes":  central_schemes,
            "state_schemes":    state_schemes,
            "park_incentives":  park_incentives,
            "gemini_analysis":  gemini_result,
            "total_subsidy_cr": gemini_result.get("total_subsidy_estimate_cr", 0),
            "net_investment_cr": gemini_result.get("net_investment_cr", investment / 1e7),
            "subsidy_pct":      gemini_result.get("subsidy_percentage", 0),
            "key_insight":      gemini_result.get("key_insight", ""),
        }

    return top10_parks
