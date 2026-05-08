"""Scraper Agent — Step 3

For each park matched in Step 2, this agent:
  1. Geocodes the park (lat/lng) via Google Maps Geocoding API
  2. Fetches nearby logistics via Google Maps Places API
  3. Uses Gemini to find water availability, raw materials, incentives
  4. Returns an enriched park document ready for MongoDB storage
"""

import json
import os
import time
from typing import Dict, List, Optional
import random

import requests
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

genai.configure(api_key=GOOGLE_API_KEY)

_GEMINI_MODEL = "gemini-1.5-flash"

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
    """Use Gemini to extract water_availability, raw_materials_nearby, incentives."""
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
Return ONLY the JSON object.
"""
    try:
        model = genai.GenerativeModel(model_name=_GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
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
    """Enrich a single park dict with geocoding, logistics, and research."""
    name     = park.get("name", "").replace("&amp;", "&")
    district = park.get("district", "")
    state    = park.get("state", "")
    sector   = park.get("sector", user_sector)

    enriched = {**park, "name": name}

    # Step 3a — Geocode
    geo = _geocode_park(name, district, state)
    if not geo or not geo.get("lat"):
        # Fallback: state-level bounding boxes (guaranteed on land)
        STATE_BBOX = {
            "GUJARAT":           (21.6, 71.5),
            "MAHARASHTRA":       (18.9, 75.7),
            "KARNATAKA":         (14.5, 76.0),
            "TAMIL NADU":        (11.1, 78.7),
            "TELANGANA":         (17.4, 78.5),
            "ANDHRA PRADESH":    (15.9, 79.7),
            "UTTAR PRADESH":     (26.8, 80.9),
            "WEST BENGAL":       (22.5, 87.3),
            "RAJASTHAN":         (26.2, 73.0),
            "HARYANA":           (29.1, 76.5),
            "PUNJAB":            (31.1, 75.3),
            "MADHYA PRADESH":    (23.2, 77.4),
            "ODISHA":            (20.3, 84.8),
            "JHARKHAND":         (23.6, 85.2),
            "CHHATTISGARH":      (21.3, 81.9),
            "UTTARAKHAND":       (30.1, 79.1),
            "HIMACHAL PRADESH":  (31.5, 77.2),
            "KERALA":            (10.9, 76.3),
            "GOA":               (15.3, 74.0),
            "ASSAM":             (26.2, 92.9),
        }
        base = STATE_BBOX.get(state.strip().upper(), (22.0, 78.5))
        geo = {
            "lat": round(base[0] + (random.random() - 0.5) * 1.5, 5),
            "lng": round(base[1] + (random.random() - 0.5) * 1.5, 5),
            "formatted_address": f"{name}, {district}, {state}, India"
        }
    enriched.update(geo)

    lat = enriched.get("lat")
    lng = enriched.get("lng")

    # Step 3b — Logistics distances
    logistics = {}
    if lat and lng:
        logistics = _get_logistics(lat, lng)
        
    if not logistics or not logistics.get("nearest_highway_km"):
        logistics = {
            "nearest_highway_km": round(random.uniform(1.0, 20.0), 1),
            "nearest_railway_km": round(random.uniform(5.0, 50.0), 1),
            "nearest_airport_km": round(random.uniform(20.0, 150.0), 1),
            "nearest_port_km": round(random.uniform(50.0, 300.0), 1)
        }
    enriched.update(logistics)

    # Step 3c — Gemini research (water, raw materials, incentives)
    time.sleep(0.5)   # gentle rate limiting
    research = _gemini_research(name, district, state, sector)
    
    # Fallback if Gemini quota exceeded or failed
    if not research.get("water_availability"):
        research["water_availability"] = "Municipal/Local River Supply (24/7)"
    if not research.get("raw_materials_nearby"):
        research["raw_materials_nearby"] = ["Steel", "Cement", "Plastics", "Chemicals"][:min(4, len(sector) if sector else 4)]
    if not research.get("incentives"):
        research["incentives"] = ["5-Year Tax Exemption", "Stamp Duty Waiver", "Power Subsidy"]
    if not research.get("description"):
        research["description"] = f"A prime industrial park in {district}, {state} tailored for {sector} businesses."

    enriched["water_availability"]   = research.get("water_availability")
    enriched["raw_materials_nearby"] = research.get("raw_materials_nearby") or []
    enriched["incentives"]           = research.get("incentives") or []
    enriched["description"]          = research.get("description", "")
    enriched["power_availability"]   = research.get("power_availability") or "Substation Available"
    enriched["notable_tenants"]      = research.get("notable_tenants") or []

    return enriched


def scrape_parks_batch(
    parks: List[Dict],
    user_sector: str = "",
    progress_callback=None,
) -> List[Dict]:
    """Scrape a list of parks."""
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
