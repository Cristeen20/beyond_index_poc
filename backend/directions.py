import math
import os
import httpx
from typing import Any


DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"

# Cap at 10 origins × 10 destinations = 100 elements to keep API cost low
_MAX_PLACES = 10


async def get_travel_times(
    places: list[dict[str, Any]],
) -> tuple[list[list[str]], list[str]]:
    """Return (NxN walking-time matrix, place names) for the top N places."""
    valid = [p for p in places[:_MAX_PLACES] if p.get("lat") and p.get("lng")]
    if len(valid) < 2:
        return [], []

    coords = [f"{p['lat']},{p['lng']}" for p in valid]
    names = [p["name"] for p in valid]
    api_key = os.environ["GOOGLE_MAPS_API_KEY"]

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            DISTANCE_MATRIX_URL,
            params={
                "origins": "|".join(coords),
                "destinations": "|".join(coords),
                "mode": "walking",
                "key": api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    matrix: list[list[str]] = []
    for i, row in enumerate(data.get("rows", [])):
        row_data: list[str] = []
        for j, element in enumerate(row.get("elements", [])):
            if element.get("status") == "OK":
                row_data.append(element["duration"]["text"])
            else:
                row_data.append(_haversine_label(valid[i], valid[j]))
        matrix.append(row_data)

    return matrix, names


def _haversine_label(place_a: dict[str, Any], place_b: dict[str, Any]) -> str:
    """Straight-line walking time estimate as a fallback."""
    try:
        R = 6371.0
        lat1 = math.radians(place_a["lat"])
        lat2 = math.radians(place_b["lat"])
        dlat = math.radians(place_b["lat"] - place_a["lat"])
        dlon = math.radians(place_b["lng"] - place_a["lng"])
        hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        km = 2 * R * math.asin(math.sqrt(hav))
        minutes = max(1, int(km / 5.0 * 60))
        return f"~{minutes} min walk"
    except (KeyError, TypeError, ValueError):
        return "unknown"
