"""Sub-agent stubs — Route, Hotel, Restaurant, Event.

Each agent wraps the existing Google Places integration to produce
typed `Option` objects for the Itinerary Agent. This is Phase-1 grade:
real data where cheap, deterministic mocks where the POC doesn't yet
have a proper provider (flights, live availability, etc.).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import date, datetime, timedelta, time
from typing import Any

logger = logging.getLogger("sub_agents")

from agent_models import (
    EventOption,
    HotelOption,
    RestaurantOption,
    RouteOption,
    TransportSegment,
    TripRequest,
    UserPreferences,
)
from places import fetch_places


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _price_from_level(price_level: str | int | None, default: float) -> float:
    """Map Google's PRICE_LEVEL_* enum (or numeric) to a $ estimate."""
    mapping = {
        "PRICE_LEVEL_FREE": 0.0,
        "PRICE_LEVEL_INEXPENSIVE": 15.0,
        "PRICE_LEVEL_MODERATE": 35.0,
        "PRICE_LEVEL_EXPENSIVE": 75.0,
        "PRICE_LEVEL_VERY_EXPENSIVE": 150.0,
        0: 0.0,
        1: 15.0,
        2: 35.0,
        3: 75.0,
        4: 150.0,
    }
    if price_level in mapping:
        return mapping[price_level]
    return default


def _classify_event_type(types: list[str]) -> str:
    """Map Google place `types` to our EventOption.type enum."""
    tset = set(types or [])
    if tset & {"museum", "art_gallery"}:
        return "museum"
    if tset & {"tourist_attraction", "landmark", "place_of_worship", "church", "temple"}:
        return "landmark"
    if tset & {"park", "natural_feature", "national_park", "campground"}:
        return "nature"
    if tset & {"shopping_mall", "store", "market"}:
        return "shopping"
    if tset & {"concert_hall", "night_club", "stadium"}:
        return "concert"
    return "other"


def _classify_meal_type(types: list[str]) -> str:
    tset = set(types or [])
    if "cafe" in tset or "bakery" in tset:
        return "cafe"
    if "meal_delivery" in tset or "meal_takeaway" in tset:
        return "lunch"
    return "dinner"


def _parse_opening_hours(descriptions: list[str]) -> dict[str, str] | None:
    if not descriptions:
        return None
    out: dict[str, str] = {}
    for entry in descriptions:
        day, sep, hours = entry.partition(":")
        if sep and day.strip() and hours.strip():
            out[day.strip().lower()] = hours.strip()
    return out or None


def _guess_cuisine(name: str, types: list[str]) -> str:
    tset = set(types or [])
    n = name.lower()
    for cuisine in ("italian", "japanese", "chinese", "indian", "thai", "mexican", "french",
                    "korean", "spanish", "greek", "vietnamese", "american", "seafood", "sushi"):
        if cuisine in n:
            return cuisine.capitalize()
    if "cafe" in tset:
        return "Cafe"
    return "International"


# --------------------------------------------------------------------------- #
# Hotel Agent
# --------------------------------------------------------------------------- #


async def run_hotel_agent(
    trip: TripRequest,
    prefs: UserPreferences | None = None,
    limit: int = 8,
) -> list[HotelOption]:
    """Return ranked hotel options for the destination + dates."""
    nights = max(trip.num_days - 1, 1)
    interests = ["hotels"]
    if prefs and prefs.preferred_hotel_rating:
        interests = [f"{prefs.preferred_hotel_rating} star hotels"]

    raw = await fetch_places(trip.destination, interests)
    hotels: list[HotelOption] = []
    for r in raw[:limit]:
        if r["lat"] is None or r["lng"] is None:
            continue
        rating_star = max(1, min(5, round(r.get("rating") or 3)))
        price_per_night = _price_from_level(r.get("price_level"), default=120.0)
        hotels.append(
            HotelOption(
                hotel_id=str(uuid.uuid4()),
                name=r["name"],
                location=trip.destination,
                latitude=float(r["lat"]),
                longitude=float(r["lng"]),
                star_rating=rating_star,
                price_per_night=price_per_night,
                total_cost=price_per_night * nights,
                score=float(r.get("rating") or 0.0),
                address=r.get("address", ""),
                amenities=[],
            )
        )
    # Rank by star match first, then rating.
    prefer_stars = prefs.preferred_hotel_rating if prefs else None
    hotels.sort(
        key=lambda h: (
            -abs((prefer_stars or h.star_rating) - h.star_rating),
            -h.score,
        )
    )
    return hotels


# --------------------------------------------------------------------------- #
# Restaurant Agent
# --------------------------------------------------------------------------- #


async def run_restaurant_agent(
    trip: TripRequest,
    prefs: UserPreferences | None = None,
    limit: int = 12,
) -> list[RestaurantOption]:
    interests = list((prefs.preferred_foods if prefs else []) or []) + ["restaurants"]
    raw = await fetch_places(trip.destination, interests[:3])
    restaurants: list[RestaurantOption] = []
    for r in raw[:limit]:
        if r["lat"] is None or r["lng"] is None:
            continue
        restaurants.append(
            RestaurantOption(
                restaurant_id=str(uuid.uuid4()),
                name=r["name"],
                cuisine=_guess_cuisine(r["name"], r.get("types", [])),
                meal_type=_classify_meal_type(r.get("types", [])),
                location=trip.destination,
                latitude=float(r["lat"]),
                longitude=float(r["lng"]),
                avg_cost_per_person=_price_from_level(r.get("price_level"), default=25.0),
                rating=float(r.get("rating") or 0.0),
                opening_hours=_parse_opening_hours(r.get("weekday_hours", [])),
            )
        )
    restaurants.sort(key=lambda x: -x.rating)
    return restaurants


# --------------------------------------------------------------------------- #
# Event / Activity Agent
# --------------------------------------------------------------------------- #


async def run_event_agent(
    trip: TripRequest,
    prefs: UserPreferences | None = None,
    limit: int = 16,
) -> list[EventOption]:
    interests = list((prefs.activity_interests if prefs else []) or [])
    if not interests:
        interests = ["top attractions", "museums", "landmarks"]
    raw = await fetch_places(trip.destination, interests[:4])
    events: list[EventOption] = []
    for r in raw[:limit]:
        if r["lat"] is None or r["lng"] is None:
            continue
        events.append(
            EventOption(
                event_id=str(uuid.uuid4()),
                name=r["name"],
                type=_classify_event_type(r.get("types", [])),
                location=trip.destination,
                latitude=float(r["lat"]),
                longitude=float(r["lng"]),
                duration_minutes=90,
                cost=_price_from_level(r.get("price_level"), default=15.0),
                typical_hours=None,
                best_time_of_day="flexible",
                closed_days=[],
            )
        )
    events.sort(key=lambda e: -(float(getattr(e, "cost", 0)) == 0.0) or 0.0)
    return events


# --------------------------------------------------------------------------- #
# Route Agent (mock — no live flight/train integration in the POC yet)
# --------------------------------------------------------------------------- #


async def run_route_agent(
    trip: TripRequest,
    prefs: UserPreferences | None = None,
) -> list[RouteOption]:
    """Deterministic mock route options.

    Real flight / rail search is out of scope for the POC. We synthesise
    two plausible options (flight + train) so the Itinerary Agent has
    something to plan around and the budget breakdown includes transport.
    """
    dep = datetime.combine(trip.start_date, time(9, 0))
    arr_flight = dep + timedelta(hours=4)
    ret_dep = datetime.combine(trip.end_date, time(17, 0))
    ret_arr = ret_dep + timedelta(hours=4)

    per_traveler_flight = 350.0
    flight = RouteOption(
        route_id=str(uuid.uuid4()),
        mode="flight",
        total_cost=per_traveler_flight * max(trip.travelers, 1) * 2,
        total_duration_minutes=int((arr_flight - dep).total_seconds() // 60)
                              + int((ret_arr - ret_dep).total_seconds() // 60),
        segments=[
            TransportSegment(
                type="flight",
                departure_location=trip.origin,
                arrival_location=trip.destination,
                departure_time=dep,
                arrival_time=arr_flight,
                cost=per_traveler_flight * max(trip.travelers, 1),
            ),
            TransportSegment(
                type="flight",
                departure_location=trip.destination,
                arrival_location=trip.origin,
                departure_time=ret_dep,
                arrival_time=ret_arr,
                cost=per_traveler_flight * max(trip.travelers, 1),
            ),
        ],
    )

    per_traveler_train = 120.0
    train_dep = dep
    train_arr = dep + timedelta(hours=7)
    train_ret_dep = ret_dep
    train_ret_arr = ret_dep + timedelta(hours=7)
    train = RouteOption(
        route_id=str(uuid.uuid4()),
        mode="train",
        total_cost=per_traveler_train * max(trip.travelers, 1) * 2,
        total_duration_minutes=int((train_arr - train_dep).total_seconds() // 60)
                              + int((train_ret_arr - train_ret_dep).total_seconds() // 60),
        segments=[
            TransportSegment(
                type="train",
                departure_location=trip.origin,
                arrival_location=trip.destination,
                departure_time=train_dep,
                arrival_time=train_arr,
                cost=per_traveler_train * max(trip.travelers, 1),
            ),
            TransportSegment(
                type="train",
                departure_location=trip.destination,
                arrival_location=trip.origin,
                departure_time=train_ret_dep,
                arrival_time=train_ret_arr,
                cost=per_traveler_train * max(trip.travelers, 1),
            ),
        ],
    )

    return [flight, train]


# --------------------------------------------------------------------------- #
# Parallel dispatch
# --------------------------------------------------------------------------- #


AGENT_RUNNERS = {
    "route": run_route_agent,
    "hotel": run_hotel_agent,
    "restaurant": run_restaurant_agent,
    "event": run_event_agent,
}


async def dispatch_agents(
    target_agents: list[str],
    trip: TripRequest,
    prefs: UserPreferences | None = None,
) -> dict[str, Any]:
    """Fan out the selected sub-agents concurrently and return their outputs."""
    logger.info("dispatch_agents: agents=%s destination=%r",
                target_agents, trip.destination)
    coros = [AGENT_RUNNERS[a](trip, prefs) for a in target_agents]
    results = await asyncio.gather(*coros, return_exceptions=True)
    out: dict[str, Any] = {}
    for name, res in zip(target_agents, results):
        if isinstance(res, Exception):
            logger.warning("agent %s raised: %s", name, res)
            out[name] = []
        else:
            logger.info("agent %s → %d options", name, len(res))
            out[name] = res
    return out
