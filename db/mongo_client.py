"""MongoDB Client — Industrial Park Finder

Manages connection to MongoDB and all CRUD operations for:
  - scraped_parks     : enriched park data from web scraping agent
  - analysis_sessions : per-user query sessions with results
"""

import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database

load_dotenv()

# ─────────────────────────────────────────────
# Connection
# ─────────────────────────────────────────────
_MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
_DB_NAME = "industrial_park_finder"

_client: Optional[MongoClient] = None
_db: Optional[Database] = None


def get_db() -> Database:
    """Return (and lazily create) the singleton MongoDB database handle."""
    global _client, _db
    if _db is None:
        _client = MongoClient(_MONGO_URI, serverSelectionTimeoutMS=5000)
        _db = _client[_DB_NAME]
        _ensure_indexes(_db)
    return _db


def _ensure_indexes(db: Database) -> None:
    """Create indexes on first connection."""
    parks = db["scraped_parks"]
    parks.create_index([("park_id", ASCENDING)], unique=True, background=True)
    parks.create_index([("state", ASCENDING), ("sector", ASCENDING)], background=True)
    parks.create_index([("scraped_at", DESCENDING)], background=True)

    sessions = db["analysis_sessions"]
    sessions.create_index([("session_id", ASCENDING)], unique=True, background=True)
    sessions.create_index([("created_at", DESCENDING)], background=True)


# ─────────────────────────────────────────────
# Scraped Parks CRUD
# ─────────────────────────────────────────────
def upsert_scraped_park(park_data: dict) -> str:
    """Insert or update an enriched park document.

    Uses park_id as the unique key. Returns the park_id.
    """
    db = get_db()
    park_id = str(park_data.get("id") or park_data.get("park_id"))
    park_data["park_id"] = park_id
    park_data["scraped_at"] = datetime.now(timezone.utc)

    db["scraped_parks"].update_one(
        {"park_id": park_id},
        {"$set": park_data},
        upsert=True
    )
    return park_id


def get_scraped_park(park_id: str) -> Optional[dict]:
    """Fetch a single scraped park by ID."""
    db = get_db()
    doc = db["scraped_parks"].find_one({"park_id": str(park_id)}, {"_id": 0})
    return doc


def get_parks_for_session(session_id: str) -> list:
    """Fetch all enriched parks stored for a session."""
    db = get_db()
    docs = list(db["scraped_parks"].find(
        {"session_id": session_id},
        {"_id": 0}
    ).sort("rank_score", DESCENDING))
    return docs


def store_ranked_parks(session_id: str, ranked_parks: list) -> None:
    """Bulk-update rank scores and research for a session's parks."""
    db = get_db()
    for park in ranked_parks:
        park_id = str(park.get("park_id") or park.get("id"))
        db["scraped_parks"].update_one(
            {"park_id": park_id},
            {"$set": {
                "session_id": session_id,
                "rank_score": park.get("rank_score", 0),
                "rank": park.get("rank"),
                "research": park.get("research", {}),
                "ranked_at": datetime.now(timezone.utc)
            }}
        )


# ─────────────────────────────────────────────
# Session CRUD
# ─────────────────────────────────────────────
def create_session(session_id: str, user_requirements: dict) -> str:
    """Create a new analysis session document."""
    db = get_db()
    db["analysis_sessions"].insert_one({
        "session_id": session_id,
        "requirements": user_requirements,
        "status": "created",
        "created_at": datetime.now(timezone.utc),
        "steps_completed": []
    })
    return session_id


def update_session_status(session_id: str, status: str, step: Optional[str] = None) -> None:
    """Update session status and mark a step as completed."""
    db = get_db()
    update = {"$set": {"status": status, "updated_at": datetime.now(timezone.utc)}}
    if step:
        update["$addToSet"] = {"steps_completed": step}
    db["analysis_sessions"].update_one({"session_id": session_id}, update)


def get_session(session_id: str) -> Optional[dict]:
    """Fetch session metadata."""
    db = get_db()
    return db["analysis_sessions"].find_one({"session_id": session_id}, {"_id": 0})


def is_db_available() -> bool:
    """Quick ping to check MongoDB connectivity."""
    try:
        get_db().command("ping")
        return True
    except Exception:
        return False
