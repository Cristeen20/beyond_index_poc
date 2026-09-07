"""Thin persistence layer for completed itineraries.

Stores each assembled Itinerary in a `trips` table alongside the
session that produced it. All functions are no-ops when pool is None
(MemorySaver / local-dev without DATABASE_URL).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_models import Itinerary

logger = logging.getLogger("trips_store")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trips (
    trip_id    TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    session_id TEXT,
    title      TEXT,
    destination TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    data       JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS trips_user_id_idx ON trips (user_id);
"""


async def setup_trips_table(pool) -> None:
    if pool is None:
        return
    async with pool.connection() as conn:
        await conn.execute(_CREATE_TABLE)
    logger.info("trips table ready")


async def save_trip(pool, itinerary: Itinerary, session_id: str) -> None:
    if pool is None:
        return
    data = json.loads(itinerary.model_dump_json())
    destination = _destination_from(data)
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO trips (trip_id, user_id, session_id, title, destination, created_at, data)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (trip_id) DO UPDATE
              SET data       = EXCLUDED.data,
                  title      = EXCLUDED.title,
                  session_id = EXCLUDED.session_id
            """,
            (
                itinerary.trip_id,
                itinerary.user_id,
                session_id,
                itinerary.title,
                destination,
                itinerary.created_at,
                json.dumps(data),
            ),
        )
    logger.info("save_trip: saved trip_id=%s user_id=%s", itinerary.trip_id, itinerary.user_id)


def _destination_from(data: dict[str, Any]) -> str:
    """Extract destination from the serialised itinerary dict as a fallback."""
    title = data.get("title", "")
    if " to " in title:
        return title.split(" to ")[-1].strip()
    if "trip to " in title.lower():
        return title.lower().split("trip to ")[-1].strip().title()
    return title


async def list_trips(pool, user_id: str) -> list[dict[str, Any]]:
    if pool is None:
        return []
    async with pool.connection() as conn:
        rows = await conn.execute(
            """
            SELECT trip_id, title, destination, created_at,
                   (data->>'total_cost')::float AS total_cost
            FROM trips
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,),
        )
        results = await rows.fetchall()
    return [dict(r) for r in results]


async def delete_trip(pool, trip_id: str) -> bool:
    """Delete a trip by id. Returns True if a row was deleted."""
    if pool is None:
        return False
    async with pool.connection() as conn:
        result = await conn.execute(
            "DELETE FROM trips WHERE trip_id = %s",
            (trip_id,),
        )
        return result.rowcount > 0


async def get_trip(pool, trip_id: str) -> dict[str, Any] | None:
    if pool is None:
        return None
    async with pool.connection() as conn:
        row = await conn.execute(
            "SELECT data FROM trips WHERE trip_id = %s",
            (trip_id,),
        )
        result = await row.fetchone()
    if result is None:
        return None
    return result["data"] if isinstance(result, dict) else result[0]
