"""Flask Backend — Industrial Park Finder

Pipeline:
  POST /api/find-parks    → Step 2: Manual query (iilb_parks.json)
  POST /api/run-pipeline  → Steps 3-6: Scrape → Rank → Schemes (SSE stream)
  GET  /api/results/<id>  → Fetch final top-10 results from MongoDB
  POST /api/chat          → Direct Gemini Q&A
  GET  /api/health        → Health check

Legacy (kept for backward compat):
  POST /api/rank-states, /api/parks, /api/schemes, /api/subsidy, /api/analyze
"""

import json
import os
import uuid
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Tools ──────────────────────────────────────────────────────────────────
from tools.location_tools import (
    query_parks,
    search_industrial_parks,
    get_state_infrastructure_score,
    geocode_location,
    get_nearby_logistics,
)
from tools.scheme_tools import match_government_schemes, estimate_subsidy_value
from tools.scoring_tools import calculate_location_score, rank_states_for_business

# ── Agents ─────────────────────────────────────────────────────────────────
from agents.scraper_agent import scrape_parks_batch
from agents.ranking_agent import rank_parks, deep_research_top10
from agents.scheme_agent  import process_schemes_for_parks

# ── DB ─────────────────────────────────────────────────────────────────────
from db.mongo_client import (
    upsert_scraped_park,
    store_ranked_parks,
    get_parks_for_session,
    create_session,
    update_session_status,
    get_session,
    is_db_available,
)

# ── In-memory results cache (fallback when MongoDB unavailable) ─────────────
_RESULTS_CACHE: dict = {}

MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html', maps_api_key=MAPS_API_KEY)


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Manual Query
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/find-parks', methods=['POST'])
def find_parks():
    """Step 2: Filter iilb_parks.json by sector + land requirement.
    Returns matched parks (up to 50) for the scraping agent.
    """
    data = request.json or {}

    try:
        result = query_parks(
            sector              = data.get('sector', ''),
            land_required_acres = float(data.get('land_required_acres') or 0),
            state               = data.get('preferred_state', ''),
            logistics_required  = bool(data.get('logistics_required', False)),
            labor_required      = bool(data.get('labor_required', False)),
            water_required      = bool(data.get('water_required', False)),
            plug_and_play       = bool(data.get('plug_and_play', False)),
            max_results         = int(data.get('max_results', 50)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# STEPS 3-6 — Full Pipeline (Server-Sent Events)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline():
    """
    Steps 3→4→5→6 via SSE stream.

    Client receives:
        data: {"step": 3, "msg": "Scraping park 2/15: Sanand GIDC...", "pct": 20}
        data: {"step": 4, "msg": "Saving to database...", "pct": 60}
        data: {"step": 5, "msg": "Ranking parks...", "pct": 70}
        data: {"step": 6, "msg": "Fetching schemes for Sanand GIDC...", "pct": 90}
        data: {"step": 7, "msg": "Done", "pct": 100, "session_id": "...", "results": [...]}
    """
    data          = request.json or {}
    matched_parks = data.get('parks', [])
    requirements  = data.get('requirements', {})

    if not matched_parks:
        return jsonify({"status": "error", "message": "No parks to process"}), 400

    session_id = str(uuid.uuid4())

    # Limit to 20 parks max for scraping (performance)
    matched_parks = matched_parks[:20]

    def generate():
        def _sse(payload: dict) -> str:
            return f"data: {json.dumps(payload)}\n\n"

        # ── Save session ───────────────────────────────────────────────────
        try:
            create_session(session_id, requirements)
        except Exception:
            pass  # DB might be unavailable

        total_parks = len(matched_parks)

        # ────────────────────────────────────────────────────────────────
        # STEP 3 — Scraping
        # ────────────────────────────────────────────────────────────────
        yield _sse({"step": 3, "msg": f"Starting web scraping for {total_parks} parks...", "pct": 2})

        scraped = []

        def scrape_progress(i, total, name):
            pct = int(2 + (i / total) * 40)
            yield_val = _sse({"step": 3, "msg": f"Scraping {i+1}/{total}: {name}", "pct": pct})
            # We can't yield from a callback, so we use a list
            _progress_events.append(yield_val)

        _progress_events = []

        # Run scraping — emit progress events interleaved
        for i, park in enumerate(matched_parks):
            name = park.get("name", f"Park #{i+1}")
            pct  = int(2 + (i / total_parks) * 40)
            yield _sse({"step": 3, "msg": f"Scraping {i+1}/{total_parks}: {name}", "pct": pct})

            try:
                from agents.scraper_agent import scrape_park
                enriched = scrape_park(park, requirements.get('sector', ''))
            except Exception as e:
                enriched = {**park, "_scrape_error": str(e)}

            scraped.append(enriched)

        # ────────────────────────────────────────────────────────────────
        # STEP 4 — Store to MongoDB
        # ────────────────────────────────────────────────────────────────
        yield _sse({"step": 4, "msg": f"Saving {len(scraped)} parks to database...", "pct": 45})

        db_ok = is_db_available()
        if db_ok:
            for park in scraped:
                try:
                    upsert_scraped_park(park)
                except Exception:
                    pass
            update_session_status(session_id, "scraped", "scraping")
        else:
            _RESULTS_CACHE[f"{session_id}_scraped"] = scraped

        yield _sse({"step": 4, "msg": "Database storage complete.", "pct": 50})

        # ────────────────────────────────────────────────────────────────
        # STEP 5a — Ranking
        # ────────────────────────────────────────────────────────────────
        yield _sse({"step": 5, "msg": "Scoring and ranking all parks...", "pct": 52})
        ranked = rank_parks(scraped, requirements)
        top10  = ranked[:10]
        yield _sse({"step": 5, "msg": f"Top 10 selected. Starting deep research...", "pct": 55})

        # ────────────────────────────────────────────────────────────────
        # STEP 5b — Deep Research on Top 10
        # ────────────────────────────────────────────────────────────────
        for i, park in enumerate(top10):
            name = park.get("name", f"Park #{i+1}")
            pct  = int(55 + (i / 10) * 20)
            yield _sse({"step": 5, "msg": f"Deep research {i+1}/10: {name}", "pct": pct})

            try:
                from agents.ranking_agent import _deep_research_park
                research = _deep_research_park(park, requirements)
                park["research"] = research
            except Exception as e:
                park["research"] = {"_error": str(e)}

        yield _sse({"step": 5, "msg": "Research complete. Fetching government schemes...", "pct": 76})

        # ────────────────────────────────────────────────────────────────
        # STEP 6 — Scheme Agent
        # ────────────────────────────────────────────────────────────────
        for i, park in enumerate(top10):
            name = park.get("name", f"Park #{i+1}")
            pct  = int(76 + (i / 10) * 20)
            yield _sse({"step": 6, "msg": f"Fetching schemes for {name}", "pct": pct})

            try:
                from agents.scheme_agent import _gemini_fetch_schemes
                sector     = requirements.get('sector', '')
                investment = float(requirements.get('investment_inr') or 0)
                is_msme    = bool(requirements.get('is_msme', False))
                is_startup = bool(requirements.get('is_startup', False))
                state      = park.get('state', '')
                park_inc   = park.get('incentives') or []

                gemini_result = _gemini_fetch_schemes(
                    park_name      = park.get('name', ''),
                    state          = state,
                    sector         = sector,
                    investment_inr = investment,
                    is_msme        = is_msme,
                    is_startup     = is_startup,
                    park_incentives= park_inc
                )
                park["schemes"] = {
                    "central_schemes":  gemini_result.get("central_schemes", []),
                    "state_schemes":    gemini_result.get("state_schemes", []),
                    "park_incentives":  park_inc,
                    "gemini_analysis":  gemini_result,
                    "total_subsidy_cr": gemini_result.get("total_subsidy_estimate_cr", 0),
                    "net_investment_cr": gemini_result.get("net_investment_cr", investment / 1e7),
                    "subsidy_pct":      gemini_result.get("subsidy_percentage", 0),
                    "key_insight":      gemini_result.get("key_insight", ""),
                }
            except Exception as e:
                park["schemes"] = {"_error": str(e)}

        # ────────────────────────────────────────────────────────────────
        # Save final results
        # ────────────────────────────────────────────────────────────────
        if db_ok:
            try:
                store_ranked_parks(session_id, top10)
                update_session_status(session_id, "complete", "ranking")
            except Exception:
                pass

        _RESULTS_CACHE[session_id] = top10

        yield _sse({
            "step":       7,
            "msg":        "Analysis complete!",
            "pct":        100,
            "session_id": session_id,
            "results":    top10,
        })

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control':  'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# GET RESULTS
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/results/<session_id>', methods=['GET'])
def get_results(session_id):
    """Fetch cached top-10 results for a completed session."""
    # Try memory cache first
    if session_id in _RESULTS_CACHE:
        return jsonify({
            "status":  "success",
            "results": _RESULTS_CACHE[session_id],
        })

    # Try MongoDB
    try:
        parks = get_parks_for_session(session_id)
        if parks:
            return jsonify({"status": "success", "results": parks})
    except Exception:
        pass

    return jsonify({"status": "error", "message": "Session not found"}), 404


# ═══════════════════════════════════════════════════════════════════════════
# CHAT
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/chat', methods=['POST'])
def chat():
    """Direct Gemini Q&A for follow-up questions."""
    data    = request.json or {}
    message = data.get('message', '')
    context = data.get('context', '')  # optional park/session context

    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)

        system = """You are an expert Indian government scheme and industrial location consultant.
Answer questions about Indian states, industrial parks, government schemes, subsidies, and business feasibility.
Provide specific, actionable information with numbers where possible.
If asked about a specific park, mention its state, sector, and any known incentives."""

        prompt = f"{system}\n\n"
        if context:
            prompt += f"Context (current park being viewed): {context}\n\n"
        prompt += f"User: {message}\n\nAssistant:"

        model    = genai.GenerativeModel('gemini-2.0-flash', tools="google_search_retrieval")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=1024)
        )
        return jsonify({"status": "success", "response": response.text})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    return jsonify({
        "status":       "healthy",
        "db_available": is_db_available(),
        "version":      "2.0.0-pipeline",
        "dataset":      "iilb_parks.json",
    })


# ═══════════════════════════════════════════════════════════════════════════
# LEGACY ENDPOINTS (backward compat)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/rank-states', methods=['POST'])
def rank_states():
    data = request.json or {}
    try:
        result = rank_states_for_business(
            sector               = data.get('sector', 'manufacturing'),
            investment_size_inr  = float(data.get('investment', 0)),
            priority_power       = int(data.get('priority_power', 5)),
            priority_logistics   = int(data.get('priority_logistics', 5)),
            priority_labor       = int(data.get('priority_labor', 5)),
            priority_cost        = int(data.get('priority_cost', 5)),
            priority_schemes     = int(data.get('priority_schemes', 5)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/parks', methods=['POST'])
def parks():
    data = request.json or {}
    try:
        result = search_industrial_parks(
            state          = data.get('state', 'gujarat'),
            sector         = data.get('sector', ''),
            min_area_acres = data.get('min_area'),
            max_budget_inr = data.get('max_budget'),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/schemes', methods=['POST'])
def schemes():
    data = request.json or {}
    try:
        result = match_government_schemes(
            business_type      = data.get('business_type', 'manufacturing'),
            sector             = data.get('sector', 'electronics'),
            investment_size_inr= float(data.get('investment', 0)),
            state              = data.get('state', 'gujarat'),
            employment_target  = int(data.get('employment', 0)),
            is_startup         = bool(data.get('is_startup', False)),
            is_msme            = bool(data.get('is_msme', False)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/subsidy', methods=['POST'])
def subsidy():
    data = request.json or {}
    try:
        result = estimate_subsidy_value(
            scheme_name          = data.get('scheme_name', ''),
            investment_size_inr  = float(data.get('investment', 0)),
            projected_revenue_inr= float(data.get('revenue', 0)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/infrastructure', methods=['POST'])
def infrastructure():
    data = request.json or {}
    try:
        return jsonify(get_state_infrastructure_score(data.get('state', 'gujarat')))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/geocode', methods=['POST'])
def geocode():
    data = request.json or {}
    try:
        return jsonify(geocode_location(data.get('address', '')))
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
