"""Ranking Agent — Step 5

Phase A: Score all scraped parks using a weighted multi-criteria model.
Phase B: Deep-dive Gemini research on the top 10 ranked parks.
"""

import json
import os
from typing import Dict, List

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
genai.configure(api_key=GOOGLE_API_KEY)

_GEMINI_MODEL = "gemini-1.5-flash"

# ─────────────────────────────────────────────────────────────────────────────
# Phase A — Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_park(park: Dict, requirements: Dict) -> float:
    score = 0.0

    # Sector match (20 pts)
    user_sector = (requirements.get("sector") or "").lower()
    park_sector = (park.get("sector") or "").lower()
    if user_sector and user_sector in park_sector:
        score += 20
    elif user_sector and any(kw in park_sector for kw in ["mixed", "general", "industrial"]):
        score += 10

    # Land availability (20 pts)
    land_required = float(requirements.get("land_required_acres") or 0)
    available     = float(park.get("available_area_acres") or 0)
    if land_required > 0:
        if available >= land_required * 2:
            score += 20
        elif available >= land_required:
            score += 15
        elif available >= land_required * 0.5:
            score += 8
    else:
        score += 10

    # Logistics (20 pts)
    logistics_req = requirements.get("logistics_required", False)
    hw  = park.get("nearest_highway_km")
    rw  = park.get("nearest_railway_km")
    air = park.get("nearest_airport_km")
    port= park.get("nearest_port_km")

    logistics_score = 0
    if hw  is not None: logistics_score += max(0, 5 - hw  / 5)
    if rw  is not None: logistics_score += max(0, 5 - rw  / 10)
    if air is not None: logistics_score += max(0, 5 - air / 20)
    if port is not None: logistics_score += max(0, 5 - port / 50)
    logistics_score = min(logistics_score, 20)

    if logistics_req and logistics_score == 0:
        logistics_score = -5
    score += logistics_score

    # Water (10 pts)
    water_req = requirements.get("water_required", False)
    water     = park.get("water_availability")
    if water and water not in [None, "null", ""]:
        score += 10
    elif water_req:
        score += 0
    else:
        score += 5

    # Incentives richness (15 pts)
    incentives = park.get("incentives") or []
    score += min(len(incentives) * 3, 15)

    # Plug & play (5 pts)
    if park.get("plug_and_play"):
        score += 5

    # Raw materials (10 pts)
    raw = park.get("raw_materials_nearby") or []
    score += min(len(raw) * 2, 10)

    return round(score, 2)


def rank_parks(scraped_parks: List[Dict], requirements: Dict) -> List[Dict]:
    for park in scraped_parks:
        park["rank_score"] = _score_park(park, requirements)

    ranked = sorted(scraped_parks, key=lambda p: p["rank_score"], reverse=True)
    for i, park in enumerate(ranked):
        park["rank"] = i + 1

    return ranked


# ─────────────────────────────────────────────────────────────────────────────
# Phase B — Deep Research on Top 10
# ─────────────────────────────────────────────────────────────────────────────

def _deep_research_park(park: Dict, requirements: Dict) -> Dict:
    req_lines = []
    if requirements.get("sector"): req_lines.append(f"Sector: {requirements['sector']}")
    if requirements.get("investment_inr"): req_lines.append(f"Investment: ₹{requirements['investment_inr']:,}")
    if requirements.get("land_required_acres"): req_lines.append(f"Land needed: {requirements['land_required_acres']} acres")
    if requirements.get("logistics_required"): req_lines.append("Logistics connectivity: Required")
    if requirements.get("water_required"): req_lines.append("Water supply: Required")

    req_text = "\n".join(req_lines) if req_lines else "General manufacturing"

    park_info = f"""
Park: {park.get('name')}
State: {park.get('state')}, District: {park.get('district')}
Sector: {park.get('sector')}
Available Land: {park.get('available_area_acres')} acres
Water: {park.get('water_availability', 'Unknown')}
Logistics: Highway {park.get('nearest_highway_km','?')}km | Railway {park.get('nearest_railway_km','?')}km
Incentives: {', '.join(park.get('incentives') or []) or 'Unknown'}
"""

    prompt = f"""
You are an expert industrial location advisor for India.

USER REQUIREMENTS:
{req_text}

INDUSTRIAL PARK BEING ANALYZED:
{park_info}

Return ONLY a valid JSON object:
{{
  "why_suitable": "2-3 sentences explaining why this park suits the user requirements",
  "why_attractive": "2-3 sentences on what makes this park stand out",
  "specific_benefits": ["list", "of", "benefits"],
  "summary_paragraph": "One compelling paragraph summarizing why this park is a strong choice"
}}
"""

    try:
        model = genai.GenerativeModel(model_name=_GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                max_output_tokens=1000,
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
        return {
            "why_suitable": f"Aligns with {requirements.get('sector','your')} sector requirements.",
            "why_attractive": f"Located in {park.get('state')} with {park.get('available_area_acres')} acres.",
            "specific_benefits": park.get("incentives") or [],
            "summary_paragraph": f"{park.get('name')} is a well-positioned facility in {park.get('state')}.",
            "_error": str(e),
        }


def deep_research_top10(
    ranked_parks: List[Dict],
    requirements: Dict,
    progress_callback=None,
) -> List[Dict]:
    top10 = ranked_parks[:10]

    for i, park in enumerate(top10):
        name = park.get("name", f"Park #{i+1}")
        if progress_callback:
            progress_callback(i, len(top10), name)

        research = _deep_research_park(park, requirements)
        park["research"] = research

    return top10
