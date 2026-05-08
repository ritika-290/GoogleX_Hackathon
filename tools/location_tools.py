"""Location & Query Tools — Industrial Park Finder

Step 2: Manual query engine against the full iilb_parks.json dataset (4200+ entries).
Filters by sector, land required, state preference, and logistics/labor/water flags.
"""

import json
import os
import re
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ── Load full dataset once at startup ──────────────────────────────────────
with open(os.path.join(_DATA_DIR, 'iilb_parks.json'), encoding='utf-8') as f:
    IILB_PARKS: List[Dict] = json.load(f)

# Legacy small dataset (kept for backward compat)
_legacy_path = os.path.join(_DATA_DIR, 'industrial_parks.json')
INDUSTRIAL_PARKS: List[Dict] = []
if os.path.exists(_legacy_path):
    with open(_legacy_path, encoding='utf-8') as f:
        INDUSTRIAL_PARKS = json.load(f)

STATE_SCORES = {}
try:
    with open(os.path.join(_DATA_DIR, 'state_scores.json'), encoding='utf-8') as f:
        STATE_SCORES = json.load(f)
except FileNotFoundError:
    pass

GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')

# ── Sector keyword mapping ──────────────────────────────────────────────────
SECTOR_KEYWORDS: Dict[str, List[str]] = {
    "electronics":        ["electronics", "electronic", "hardware", "semiconductor", "it and ites", "it & ites"],
    "automobile":         ["automobile", "auto", "automotive", "vehicle", "ev", "electric vehicle"],
    "pharma":             ["pharma", "pharmaceutical", "biotech", "life science", "medical"],
    "textile":            ["textile", "apparel", "garment", "handloom", "weaving", "hosiery"],
    "food_processing":    ["food", "agri", "agriculture", "cold chain", "dairy"],
    "IT":                 ["it", "ites", "software", "tech park", "technology"],
    "renewable_energy":   ["energy", "solar", "wind", "power"],
    "logistics":          ["logistics", "warehouse", "warehousing", "storage", "freight"],
    "aerospace":          ["aerospace", "defence", "defense", "aviation"],
    "semiconductor":      ["semiconductor", "fab", "chip", "microelectronics"],
    "chemicals":          ["chemical", "petro", "petroleum", "plastic", "polymer"],
    "manufacturing":      ["manufacturing", "industrial", "mixed", "general"],
}

# State name normalisation (dataset uses ALL-CAPS, UI uses Title Case / slugs)
_STATE_NORMALISE: Dict[str, str] = {
    "andhra pradesh": "ANDHRA PRADESH",
    "arunachal pradesh": "ARUNACHAL PRADESH",
    "assam": "ASSAM",
    "bihar": "BIHAR",
    "chhattisgarh": "CHHATTISGARH",
    "goa": "GOA",
    "gujarat": "GUJARAT",
    "haryana": "HARYANA",
    "himachal pradesh": "HIMACHAL PRADESH",
    "jharkhand": "JHARKHAND",
    "karnataka": "KARNATAKA",
    "kerala": "KERALA",
    "madhya pradesh": "MADHYA PRADESH",
    "maharashtra": "MAHARASHTRA",
    "manipur": "MANIPUR",
    "meghalaya": "MEGHALAYA",
    "mizoram": "MIZORAM",
    "nagaland": "NAGALAND",
    "odisha": "ODISHA",
    "punjab": "PUNJAB",
    "rajasthan": "RAJASTHAN",
    "sikkim": "SIKKIM",
    "tamil_nadu": "TAMIL NADU",
    "tamil nadu": "TAMIL NADU",
    "telangana": "TELANGANA",
    "tripura": "TRIPURA",
    "uttar pradesh": "UTTAR PRADESH",
    "uttar_pradesh": "UTTAR PRADESH",
    "uttarakhand": "UTTARAKHAND",
    "west bengal": "WEST BENGAL",
    "west_bengal": "WEST BENGAL",
    "delhi": "DELHI",
    "jammu & kashmir": "JAMMU & KASHMIR",
}


def _normalise_state(state: str) -> str:
    key = state.strip().lower().replace("_", " ")
    return _STATE_NORMALISE.get(key, state.upper())


def _sector_matches(park_sector: str, query_sector: str) -> bool:
    """Check if park sector matches the user's chosen sector using keyword map."""
    if not query_sector:
        return True
    park_sector_lower = (park_sector or "").lower()
    keywords = SECTOR_KEYWORDS.get(query_sector.lower(), [query_sector.lower()])
    return any(kw in park_sector_lower for kw in keywords)


# ── Step 2: Main Query Function ─────────────────────────────────────────────
def query_parks(
    sector: str,
    land_required_acres: float = 0.0,
    state: str = "",
    logistics_required: bool = False,
    labor_required: bool = False,
    water_required: bool = False,
    plug_and_play: bool = False,
    max_results: int = 50,
) -> Dict:
    """
    Step 2 — Manual query against the full IILB dataset.

    Returns up to `max_results` parks matching sector + land requirements.
    These are then sent to the scraper agent (Step 3).

    Args:
        sector:              User-selected sector string.
        land_required_acres: Minimum available land in acres.
        state:               Optional preferred state filter.
        logistics_required:  If True, boosts parks with road/rail proximity data.
        labor_required:      Placeholder for future labor-data filtering.
        water_required:      Placeholder for future water-data filtering.
        plug_and_play:       If True, only return plug-and-play parks.
        max_results:         Cap on returned parks before scraping.

    Returns:
        Dict with status, count, and list of matched parks.
    """
    results = []

    for park in IILB_PARKS:
        # ── Sector filter ──────────────────────────────────────────────────
        if not _sector_matches(park.get("sector", ""), sector):
            continue

        # ── Land filter ────────────────────────────────────────────────────
        available = park.get("available_area_acres", 0.0)
        if available is None:
            available = 0.0
        # Convert ha to acres if acres field missing
        if available == 0.0:
            avail_ha = park.get("available_area_ha", 0.0) or 0.0
            available = round(avail_ha * 2.471, 3)

        if land_required_acres > 0 and available < land_required_acres:
            continue

        # ── State filter ───────────────────────────────────────────────────
        if state:
            normalised = _normalise_state(state)
            if park.get("state", "").upper() != normalised:
                continue

        # ── Plug & play filter ─────────────────────────────────────────────
        if plug_and_play and not park.get("plug_and_play", False):
            continue

        # ── Clean HTML entities ────────────────────────────────────────────
        name = park.get("name", "").replace("&amp;", "&").strip()

        results.append({
            "id":                 str(park.get("id", "")),
            "name":               name,
            "state":              park.get("state", ""),
            "district":           park.get("district", ""),
            "sector":             park.get("sector", ""),
            "type":               park.get("type", ""),
            "total_area_acres":   round(park.get("total_area_acres", 0.0) or 0.0, 2),
            "available_area_acres": round(available, 2),
            "plug_and_play":      park.get("plug_and_play", False),
            "ownership":          park.get("ownership", ""),
        })

    # ── Sort: available land descending, then plug_and_play priority ────────
    results.sort(key=lambda p: (
        -int(p.get("plug_and_play", False)),
        -(p.get("available_area_acres") or 0)
    ))

    total_found = len(results)
    results = results[:max_results]

    return {
        "status":      "success",
        "total_found": total_found,
        "returned":    len(results),
        "parks":       results,
    }


# ── Legacy helpers (kept for backward compatibility) ────────────────────────
def search_industrial_parks(
    state: str,
    sector: str = "",
    min_area_acres: Optional[float] = None,
    max_budget_inr: Optional[float] = None
) -> Dict:
    """Original search using legacy 14-park dataset."""
    state_parks = [p for p in INDUSTRIAL_PARKS if p['state'].lower() == state.lower()]
    if sector:
        state_parks = [p for p in state_parks if sector.lower() in p.get('sector', '').lower()]
    if min_area_acres:
        state_parks = [p for p in state_parks if p.get('available_acres', 0) >= min_area_acres]
    state_parks.sort(key=lambda x: x.get('infrastructure_score', 0), reverse=True)
    return {"status": "success", "count": len(state_parks), "parks": state_parks[:5]}


def get_state_infrastructure_score(state: str) -> Dict:
    state_key = state.lower().replace(" ", "_")
    if state_key not in STATE_SCORES:
        return {"status": "error", "message": f"State '{state}' not found"}
    data = STATE_SCORES[state_key]
    composite = (
        data['power_reliability'] * 0.25 +
        data['logistics_score'] * 0.25 +
        data['labor_availability'] * 0.20 +
        data['ease_of_business'] * 0.30
    )
    tier = "A" if composite >= 88 else ("B" if composite >= 80 else "C")
    tier_label = {"A": "Highly Recommended", "B": "Recommended", "C": "Consider Alternatives"}[tier]
    return {"status": "success", "state": state_key, "composite_score": round(composite, 1),
            "tier": tier, "tier_label": tier_label, "metrics": data}


def geocode_location(address: str) -> Dict:
    if not GOOGLE_MAPS_API_KEY:
        return {"status": "error", "message": "Google Maps API key not configured"}
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": address + ", India", "key": GOOGLE_MAPS_API_KEY, "region": "in"},
            timeout=10
        )
        data = resp.json()
        if data['status'] != 'OK':
            return {"status": "error", "message": f"Geocoding failed: {data['status']}"}
        result = data['results'][0]
        return {
            "status": "success",
            "lat": result['geometry']['location']['lat'],
            "lng": result['geometry']['location']['lng'],
            "formatted_address": result['formatted_address'],
            "place_id": result['place_id']
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_nearby_logistics(lat: float, lng: float, radius_km: int = 50) -> Dict:
    if not GOOGLE_MAPS_API_KEY:
        return {"status": "error", "message": "Google Maps API key not configured"}
    categories = {
        "highways": "highway", "railway_stations": "train_station",
        "airports": "airport", "ports": "harbor"
    }
    results = {}
    for name, place_type in categories.items():
        try:
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                params={"location": f"{lat},{lng}", "radius": radius_km * 1000,
                        "type": place_type, "key": GOOGLE_MAPS_API_KEY},
                timeout=10
            )
            data = resp.json()
            results[name] = [
                {"name": p['name'], "rating": p.get('rating', 'N/A'), "vicinity": p.get('vicinity', '')}
                for p in data.get('results', [])[:3]
            ] if data.get('status') == 'OK' else []
        except Exception:
            results[name] = []
    return {"status": "success", "infrastructure": results}
