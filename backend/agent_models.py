"""Pydantic models for the Itinerary Agent architecture (§3 of itenary_agent.md).

Kept separate from the legacy models.py (which the /chat and /itinerary
endpoints still use) so the new dual-path flow can be introduced without
breaking the existing POC.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# User context
# --------------------------------------------------------------------------- #


class MealTiming(BaseModel):
    breakfast: str = "08:00"
    lunch: str = "12:30"
    dinner: str = "19:30"


class UserPreferences(BaseModel):
    preferred_foods: list[str] = []
    preferred_hotel_rating: int | None = None
    travel_style: Literal["relaxed", "balanced", "packed"] = "balanced"
    meal_timing: MealTiming | None = None
    walking_tolerance: Literal["low", "medium", "high"] = "medium"
    activity_interests: list[str] = []
    accessibility_needs: list[str] = []


class TripSummary(BaseModel):
    trip_id: str
    destination: str
    start_date: date
    end_date: date


class UserProfile(BaseModel):
    user_id: str
    name: str = ""
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    travel_history: list[TripSummary] = []


class TripRequest(BaseModel):
    origin: str
    destination: str
    start_date: date
    end_date: date
    travelers: int = 1
    total_budget: float = 0.0
    currency: str = "USD"
    must_include: list[str] = []
    must_exclude: list[str] = []
    special_occasion: str | None = None

    @property
    def num_days(self) -> int:
        return (self.end_date - self.start_date).days + 1


# --------------------------------------------------------------------------- #
# Sub-agent outputs (inputs to the Itinerary Agent)
# --------------------------------------------------------------------------- #


class TransportSegment(BaseModel):
    type: Literal["flight", "train", "bus", "car"]
    departure_location: str
    arrival_location: str
    departure_time: datetime
    arrival_time: datetime
    cost: float
    booking_ref: str | None = None


class RouteOption(BaseModel):
    route_id: str
    segments: list[TransportSegment]
    total_cost: float
    total_duration_minutes: int
    mode: Literal["flight", "train", "bus", "car"]


class HotelOption(BaseModel):
    hotel_id: str
    name: str
    location: str
    latitude: float
    longitude: float
    star_rating: int
    price_per_night: float
    total_cost: float
    score: float = 0.0
    address: str = ""
    amenities: list[str] = []
    check_in_time: time = time(15, 0)
    check_out_time: time = time(11, 0)


class RestaurantOption(BaseModel):
    restaurant_id: str
    name: str
    cuisine: str
    meal_type: Literal["breakfast", "lunch", "dinner", "cafe"]
    location: str
    latitude: float
    longitude: float
    avg_cost_per_person: float
    rating: float
    opening_hours: dict[str, str] | None = None


class EventOption(BaseModel):
    event_id: str
    name: str
    type: Literal["museum", "festival", "concert", "landmark", "nature", "shopping", "other"]
    location: str
    latitude: float
    longitude: float
    duration_minutes: int
    cost: float
    typical_hours: str | None = None
    best_time_of_day: Literal["morning", "afternoon", "evening", "flexible"] = "flexible"
    closed_days: list[str] = []


# --------------------------------------------------------------------------- #
# Itinerary (the planning engine's product)
# --------------------------------------------------------------------------- #


class TimeSegment(BaseModel):
    id: str
    start_time: time
    end_time: time
    type: Literal["travel", "activity", "meal", "rest", "free_time", "buffer"]
    title: str
    description: str = ""
    location: str = ""
    latitude: float | None = None
    longitude: float | None = None
    cost: float = 0.0
    item_ref: str | None = None
    booking_status: Literal["pending", "approved", "booked", "cancelled"] = "pending"


class DayPlan(BaseModel):
    day_number: int
    date: date
    day_name: str
    location: str
    accommodation: HotelOption | None = None
    segments: list[TimeSegment] = []
    total_cost: float = 0.0
    notes: list[str] = []

    @property
    def is_free_day(self) -> bool:
        return all(s.type == "free_time" for s in self.segments)


class BudgetBreakdown(BaseModel):
    transport: float = 0.0
    accommodation: float = 0.0
    food: float = 0.0
    activities: float = 0.0
    other: float = 0.0
    total: float = 0.0
    remaining: float = 0.0


class Itinerary(BaseModel):
    trip_id: str
    user_id: str
    title: str
    days: list[DayPlan]
    total_cost: float
    budget_breakdown: BudgetBreakdown
    notes: list[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1


# --------------------------------------------------------------------------- #
# Router contract (§3.4)
# --------------------------------------------------------------------------- #


AgentName = Literal["route", "hotel", "restaurant", "event"]


class IntentClassification(BaseModel):
    route: Literal["direct", "full"]
    target_agents: list[AgentName] = []
    extracted_slots: dict[str, str] = {}
    missing_required_slots: list[str] = []
    confidence: float = 0.0
    rationale: str | None = None


# Required-slot sets per agent — used by the direct flow's slot gate (§3.4).
REQUIRED_SLOTS: dict[str, set[str]] = {
    "hotel": {"destination", "dates"},
    "restaurant": {"destination"},
    "route": {"origin", "destination", "dates"},
    "event": {"destination"},
}


# --------------------------------------------------------------------------- #
# LangGraph-style state (§3.3)
# --------------------------------------------------------------------------- #


Phase = Literal[
    "routing",
    "intake",
    "agent_dispatch",
    "planning",
    "review",
    "revision",
    "approved",
    "booked",
    "direct_answer",
]


class PlanningState(BaseModel):
    # User context
    user_profile: UserProfile | None = None
    trip_request: TripRequest | None = None

    # Agent outputs
    route_options: list[RouteOption] = []
    hotel_options: list[HotelOption] = []
    restaurant_options: list[RestaurantOption] = []
    event_options: list[EventOption] = []

    # Itinerary (current state)
    itinerary: Itinerary | None = None
    revision_feedback: str | None = None

    # Routing
    intent: IntentClassification | None = None
    direct_result: list[dict] | None = None

    # Workflow control
    phase: Phase = "routing"
    agent_outputs_received: dict[str, bool] = {}


# --------------------------------------------------------------------------- #
# API request/response wrappers
# --------------------------------------------------------------------------- #


class PlanRequest(BaseModel):
    """Top-level request that enters the Travel Orchestrator."""

    message: str
    user_profile: UserProfile | None = None
    trip_request: TripRequest | None = None
    history: list[dict] = []


class PlanResponse(BaseModel):
    route: Literal["direct", "full"]
    intent: IntentClassification
    itinerary: Itinerary | None = None
    direct_result: list[dict] | None = None
    followup_question: str | None = None
    message: str = ""
