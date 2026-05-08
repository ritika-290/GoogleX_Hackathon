# Project Context: Startup India Advisor / Industrial Finder Engine

## 1. Project Overview
The **Industrial Finder Multi-Agent Engine** is an AI-powered recommendation platform designed to help investors and manufacturing startups find the perfect industrial park in India. It automates the complex process of evaluating thousands of locations based on land availability, logistics, sector suitability, and government subsidies.

## 2. Core Architecture: The 7-Step Pipeline
The application executes a sequential multi-agent workflow:
1.  **Requirements Collection**: Multi-step wizard UI captures sector, investment, land needs, and business status (MSME/Startup).
2.  **Dataset Filtering**: Direct querying of the `iilb_parks.json` dataset (4,200+ parks).
3.  **Scraper Agent**: Enriches results via Google Maps (Geocoding/Places) and Gemini (Water/Incentives/Description).
4.  **Database Persistence**: Stores enriched data in MongoDB (or in-memory fallback) to avoid redundant API calls.
5.  **Ranking Agent**: Scores parks using a weighted 100-point model and performs deep qualitative research on the Top 10.
6.  **Scheme Agent**: Uses Gemini with **Google Search Grounding** to fetch LIVE Central/State schemes, official application URLs, and subsidy estimates.
7.  **Interactive Reporting**: Delivers a full interactive report via Server-Sent Events (SSE) for real-time progress updates.

## 3. Key Components & Features
-   **Dynamic Scheme Agent**: Replaced legacy `schemes.json` with a real-time scraping agent using the `google.genai` SDK. It fetches actual `.gov.in` links and calculates a monetary "Subsidy Stack."
-   **Geo-Resilience**: Implemented a state-level bounding box fallback system for geocoding to ensure map markers never land in the ocean even if API quotas are hit.
-   **Rich UI/UX**: Built with Vanilla CSS (Glassmorphism), dynamic progress bars, interactive Google Maps markers, and detailed comparison cards.
-   **Quota Management**: Logic includes robust fallbacks (mock data) for Geocoding, Distance Matrix, and Gemini Research to ensure the pipeline never breaks under API rate limits.

## 4. Technical Stack
-   **Backend**: Flask (Python), SSE for real-time streaming.
-   **AI Orchestration**: Google Gemini 1.5 Flash / 2.0 Flash (`google.genai` & `google-generativeai`).
-   **APIs**: Google Maps (Geocoding, Places, Distance Matrix), Google Search Grounding.
-   **Storage**: MongoDB (local/Atlas).
-   **Frontend**: HTML5, Vanilla CSS3, Javascript (ES6).

## 5. Important Project Context (from Development)
-   **Model Transition**: Switched to `gemini-1.5-flash` for high-volume scraping and `gemini-2.0-flash` for search-grounded scheme fetching.
-   **Bug Fixes**: 
    - Resolved `ImportError` and `NameError` issues caused by rapid refactoring of agent internal functions.
    - Fixed the "Marker in Ocean" bug by clamping random jitter to state-level centers.
    - Fixed UI rendering where subsidies/schemes were hidden due to duplicate `try` blocks in `app.py`.
-   **Legacy Cleanup**: Removed `schemes.json` and hardcoded state slug maps to make the system fully autonomous and dynamic.

## 6. Directory Structure
```text
/agents
  ├── scraper_agent.py  # Maps & Research (Step 3)
  ├── ranking_agent.py  # Scoring & Qualitative (Step 5)
  └── scheme_agent.py   # Live Govt. Schemes (Step 6)
/data
  └── iilb_parks.json   # Master dataset of 4,200+ parks
/db
  └── __init__.py       # MongoDB logic
/templates
  └── index.html        # Main interactive dashboard
/tools
  ├── location_tools.py # Geo utilities
  └── scheme_tools.py   # Legacy matching (partially deprecated)
app.py                  # Flask Backend & SSE Pipeline
```
