import logging
import os

from dotenv import load_dotenv

load_dotenv()  # load .env before anything else imports os.environ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # uvicorn installs its own root handlers; force= wins.
)
for _name in (
    "intake_router", "travel_orchestrator", "itinerary_agent", "sub_agents",
):
    logging.getLogger(_name).setLevel(logging.INFO)

from contextlib import asynccontextmanager

import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langfuse.decorators import langfuse_context

from pydantic import BaseModel

from agent_models import Itinerary, PlanRequest, PlanResponse
from graph import close_checkpointer, get_pool, init_checkpointer
from travel_orchestrator import plan as plan_handler
from trips_store import delete_trip, get_trip, list_trips, save_trip, setup_trips_table


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Opens the Postgres pool and creates the checkpoint + trips tables when
    # DATABASE_URL is set; no-op for the local MemorySaver path.
    await init_checkpointer()
    await setup_trips_table(get_pool())
    yield
    await close_checkpointer()
    # Drain any pending Langfuse spans before the process exits — the SDK
    # batches in the background and would otherwise drop the tail on a
    # serverless cold-stop. No-op when Langfuse env vars are unset.
    langfuse_context.flush()


app = FastAPI(title="Trip Itinerary Generator", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/plan", response_model=PlanResponse)
async def plan_endpoint(req: PlanRequest) -> PlanResponse:
    """Session-checkpointed LangGraph plan flow. First call for a session_id
    runs from START; subsequent calls resume at wait_for_next_message and
    re-enter intent_decision with the new message + persisted trip/itinerary
    context. See itinerary_langgraph_flow.md."""
    try:
        return await plan_handler(req)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Missing env var: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class SaveTripRequest(BaseModel):
    session_id: str
    user_id: str
    itinerary: dict


@app.post("/trips/save")
async def save_trip_endpoint(req: SaveTripRequest) -> dict:
    """Explicitly save an itinerary to the dashboard (user-initiated)."""
    try:
        itin = Itinerary.model_validate(req.itinerary)
        itin = itin.model_copy(update={"user_id": req.user_id})
        await save_trip(get_pool(), itin, req.session_id)
        return {"trip_id": itin.trip_id, "status": "saved"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/trips")
async def get_trips(user_id: str) -> list[dict]:
    """Return trip summaries for a given user_id, newest first."""
    return await list_trips(get_pool(), user_id)


@app.get("/trips/{trip_id}")
async def get_trip_detail(trip_id: str) -> dict:
    """Return the full itinerary JSON for a trip."""
    trip = await get_trip(get_pool(), trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    return trip


@app.get("/trips/{trip_id}/map")
async def get_trip_map(trip_id: str):
    """Proxy a Google Static Maps image with pins for every segment that has
    lat/lng. Key stays server-side so it never reaches the browser."""
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="Maps API key not configured")

    trip = await get_trip(get_pool(), trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail="Trip not found")

    # Collect unique (lat, lng) pairs from segments, deduplicated to 15 pins.
    seen: set[tuple[float, float]] = set()
    markers: list[str] = []
    colors = ["red", "blue", "green", "purple", "orange", "yellow"]
    for day in trip.get("days", []):
        day_idx = day.get("day_number", 1) - 1
        color = colors[day_idx % len(colors)]
        label = str(day.get("day_number", ""))
        for seg in day.get("segments", []):
            lat = seg.get("latitude")
            lng = seg.get("longitude")
            if lat and lng and (lat, lng) not in seen:
                seen.add((lat, lng))
                markers.append(f"color:{color}|label:{label}|{lat},{lng}")
            if len(markers) >= 15:
                break
        if len(markers) >= 15:
            break

    if not markers:
        raise HTTPException(status_code=404, detail="No coordinates found in itinerary")

    params = [("size", "480x520"), ("scale", "2"), ("maptype", "roadmap"), ("key", api_key)]
    for m in markers:
        params.append(("markers", m))

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get("https://maps.googleapis.com/maps/api/staticmap", params=params)
        r.raise_for_status()
        return StreamingResponse(
            content=r.aiter_bytes(),
            media_type=r.headers.get("content-type", "image/png"),
        )


@app.delete("/trips/{trip_id}")
async def delete_trip_endpoint(trip_id: str) -> dict:
    """Delete a saved trip."""
    deleted = await delete_trip(get_pool(), trip_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Trip not found")
    return {"status": "deleted"}
