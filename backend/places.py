import os
import httpx
from typing import Any


PLACES_URL = "https://places.googleapis.com/v1/places:searchText"

_FIELD_MASK = ",".join([
    "places.displayName",
    "places.formattedAddress",
    "places.location",
    "places.rating",
    "places.types",
    "places.editorialSummary",
    "places.regularOpeningHours",
    "places.priceLevel",
])


async def fetch_places(destination: str, interests: list[str]) -> list[dict[str, Any]]:
    api_key = os.environ["GOOGLE_MAPS_API_KEY"]

    queries: list[str] = []
    for interest in interests[:4]:
        queries.append(f"{interest} in {destination}")
    if not queries:
        queries = [
            f"top tourist attractions in {destination}",
            f"best restaurants in {destination}",
        ]

    all_places: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=15.0) as client:
        for query in queries:
            resp = await client.post(
                PLACES_URL,
                headers={
                    "X-Goog-Api-Key": api_key,
                    "X-Goog-FieldMask": _FIELD_MASK,
                    "Content-Type": "application/json",
                },
                json={"textQuery": query, "maxResultCount": 10},
            )
            resp.raise_for_status()
            for place in resp.json().get("places", []):
                all_places.append(_normalise(place))

    return _dedupe(all_places)[:30]


def _normalise(raw: dict[str, Any]) -> dict[str, Any]:
    loc = raw.get("location", {})
    hours = raw.get("regularOpeningHours", {})
    return {
        "name": raw.get("displayName", {}).get("text", ""),
        "address": raw.get("formattedAddress", ""),
        "lat": loc.get("latitude"),
        "lng": loc.get("longitude"),
        "rating": raw.get("rating"),
        "types": raw.get("types", [])[:3],
        "summary": raw.get("editorialSummary", {}).get("text", ""),
        "open_now": hours.get("openNow"),
        "price_level": raw.get("priceLevel"),
    }


def _dedupe(places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for place in places:
        key = place["name"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(place)
    return unique
