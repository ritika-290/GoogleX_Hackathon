"""Scraper Agent — Step 3

For each park matched in Step 2, this agent:
  1. Geocodes the park (lat/lng) via Google Maps Geocoding API
  2. Fetches nearby logistics via Google Maps Places API
  3. Uses Gemini (with Google Search grounding) to find:
       - Water availability
       - Nearest highway/railway/airport/port distances
       - Raw materials available nearby (relevant to the sector)
       - Park-specific incentives (tax breaks, free power, etc.)
  4. Returns an enriched park document ready for MongoDB storage (Step 4)
"""

import json
import os
import time
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

genai.configure(api_key=GOOGLE_API_KEY)

_GEMINI_MODEL = "gemini-2.0-flash"

# ─────────────────────────────────────────────────────────────────────────────
# Geocoding
# ─────────────────────────────────────────────────────────────────────────────

def _geocode_park(park_name: str, district: str, state: str) -> Dict:
    """Return lat/lng for a park using Google Maps Geocoding API."""
    if not GOOGLE_MAPS_API_KEY:
        return {}

    query = f"{park_name}, {district}, {state}, India"
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": query, "key": GOOGLE_MAPS_API_KEY, "region": "in"},
            timeout=10,
        )
        data = resp.json()
        if data.get("status") == "OK" and data["results"]:
            loc = data["results"][0]["geometry"]["location"]
            return {
                "lat": loc["lat"],
                "lng": loc["lng"],
                "formatted_address": data["results"][0].get("formatted_address", ""),
            }
    except Exception:
        pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Nearby logistics via Places API
# ─────────────────────────────────────────────────────────────────────────────

def _nearest_km(lat: float, lng: float, place_type: str, radius_m: int = 80000) -> Optional[float]:
    """Return distance in km to the nearest place of `place_type`."""
    if not GOOGLE_MAPS_API_KEY or not lat or not lng:
        return None
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params={
                "location": f"{lat},{lng}",
                "radius": radius_m,
                "type": place_type,
                "key": GOOGLE_MAPS_API_KEY,
            },
            timeout=10,
        )
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None

        # Use Distance Matrix for the first result
        dest = results[0]["geometry"]["location"]
        dm_resp = requests.get(
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            params={
                "origins": f"{lat},{lng}",
                "destinations": f"{dest['lat']},{dest['lng']}",
                "mode": "driving",
                "key": GOOGLE_MAPS_API_KEY,
            },
            timeout=10,
        )
        dm = dm_resp.json()
        rows = dm.get("rows", [])
        if rows and rows[0]["elements"][0]["status"] == "OK":
            dist_m = rows[0]["elements"][0]["distance"]["value"]
            return round(dist_m / 1000, 1)
    except Exception:
        pass
    return None


def _get_logistics(lat: float, lng: float) -> Dict:
    """Fetch distances to highway, railway, airport, and port."""
    if not lat or not lng:
        return {}
    return {
        "nearest_highway_km":  _nearest_km(lat, lng, "highway", 30000),
        "nearest_railway_km":  _nearest_km(lat, lng, "train_station", 50000),
        "nearest_airport_km":  _nearest_km(lat, lng, "airport", 100000),
        "nearest_port_km":     _nearest_km(lat, lng, "harbor", 200000),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Research (water, raw materials, incentives)
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_research(park_name: str, district: str, state: str, sector: str) -> Dict:
    """
    Use Gemini with Google Search grounding to extract:
      - water_availability
      - raw_materials_nearby
      - incentives (tax exemptions, free power, land subsidies, etc.)
      - brief description of the park
    """
    prompt = f"""
You are an industrial park research assistant for India.

Research the following industrial park and provide ONLY factual, verified information:

Park Name: {park_name}
District: {district}
State: {state}
Primary Sector: {sector}

Return your answer as a valid JSON object with EXACTLY these keys:
{{
  "water_availability": "Description of water source and reliability (string)",
  "raw_materials_nearby": ["list", "of", "relevant", "raw materials", "available nearby"],
  "incentives": ["list of specific benefits like tax exemptions, subsidies, free power duration"],
  "description": "2-3 sentence factual description of the park",
  "power_availability": "Description of power supply and capacity if known",
  "notable_tenants": ["list of known companies in the park, if any"]
}}

If you cannot find specific information for a field, use null for that field.
Return ONLY the JSON object, no markdown, no extra text.
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
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )
        text = response.text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        return {
            "water_availability": None,
            "raw_materials_nearby": [],
            "incentives": [],
            "description": f"Industrial park located in {district}, {state}.",
            "power_availability": None,
            "notable_tenants": [],
            "_error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main scrape function
# ─────────────────────────────────────────────────────────────────────────────

def scrape_park(park: Dict, user_sector: str = "") -> Dict:
    """
    Enrich a single park dict with:
      - lat/lng
      - logistics distances
      - water, raw materials, incentives (via Gemini)

    Args:
        park:        Raw park dict from Step 2 (iilb_parks.json entry).
        user_sector: The user's sector choice (for raw material relevance).

    Returns:
        Enriched park dict ready for MongoDB upsert.
    """
    name     = park.get("name", "").replace("&amp;", "&")
    district = park.get("district", "")
    state    = park.get("state", "")
    sector   = park.get("sector", user_sector)

    enriched = {**park, "name": name}

    # Step 3a — Geocode
    geo = _geocode_park(name, district, state)
    enriched.update(geo)

    lat = enriched.get("lat")
    lng = enriched.get("lng")

    # Step 3b — Logistics distances
    if lat and lng:
        logistics = _get_logistics(lat, lng)
        enriched.update(logistics)

    # Step 3c — Gemini research (water, raw materials, incentives)
    time.sleep(0.5)   # gentle rate limiting
    research = _gemini_research(name, district, state, sector)
    enriched["water_availability"]   = research.get("water_availability")
    enriched["raw_materials_nearby"] = research.get("raw_materials_nearby") or []
    enriched["incentives"]           = research.get("incentives") or []
    enriched["description"]          = research.get("description", "")
    enriched["power_availability"]   = research.get("power_availability")
    enriched["notable_tenants"]      = research.get("notable_tenants") or []

    return enriched


def scrape_parks_batch(
    parks: List[Dict],
    user_sector: str = "",
    progress_callback=None,
) -> List[Dict]:
    """
    Scrape a list of parks. Calls progress_callback(i, total, park_name)
    after each park if provided.

    Returns list of enriched park dicts.
    """
    enriched = []
    total = len(parks)

    for i, park in enumerate(parks):
        name = park.get("name", f"Park #{i+1}")
        if progress_callback:
            progress_callback(i, total, name)

        try:
            result = scrape_park(park, user_sector)
        except Exception as e:
            result = {**park, "_scrape_error": str(e)}

        enriched.append(result)

    return enriched
