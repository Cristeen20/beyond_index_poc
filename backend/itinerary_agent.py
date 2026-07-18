"""Itinerary Agent — Planning Engine (§4 of itenary_agent.md).

Phase-1 implementation (§5): a straight-through Python pipeline
(Load → Budget → Schedule → Conflict → Generate) with an LLM tool-call
for the day-by-day scheduling step. This can later be lifted into a
LangGraph sub-graph without changing the step boundaries.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timedelta
from typing import Any

import openai

logger = logging.getLogger("itinerary_agent")

from agent_models import (
    BudgetBreakdown,
    DayPlan,
    EventOption,
    HotelOption,
    Itinerary,
    PlanningState,
    RestaurantOption,
    RouteOption,
    TimeSegment,
    TripRequest,
    UserProfile,
)


# --------------------------------------------------------------------------- #
# Budget allocation ratios (§4 Step 2)
# --------------------------------------------------------------------------- #

_STYLE_RATIOS = {
    "relaxed":  {"accommodation": 0.40, "food": 0.25, "activities": 0.20, "other": 0.15},
    "balanced": {"accommodation": 0.30, "food": 0.25, "activities": 0.30, "other": 0.15},
    "packed":   {"accommodation": 0.20, "food": 0.20, "activities": 0.45, "other": 0.15},
}


# --------------------------------------------------------------------------- #
# Structured output tool for the LLM scheduler (§4 Step 3+5)
# --------------------------------------------------------------------------- #


_SEGMENT_SCHEMA = {
    "type": "object",
    "required": ["start_time", "end_time", "type", "title"],
    "properties": {
        "start_time": {"type": "string", "description": "24h HH:MM"},
        "end_time":   {"type": "string", "description": "24h HH:MM"},
        "type": {"type": "string", "enum": ["travel", "activity", "meal", "rest", "free_time", "buffer"]},
        "title": {"type": "string"},
        "description": {"type": "string"},
        "location": {"type": "string"},
        "cost": {"type": "number"},
        "item_ref": {"type": "string", "description": "hotel_id / restaurant_id / event_id / route_id if applicable"},
    },
}

_DAY_SCHEMA = {
    "type": "object",
    "required": ["day_number", "date", "day_name", "location", "segments"],
    "properties": {
        "day_number": {"type": "integer"},
        "date":       {"type": "string", "description": "ISO date YYYY-MM-DD"},
        "day_name":   {"type": "string"},
        "location":   {"type": "string"},
        "accommodation_hotel_id": {"type": "string"},
        "segments": {"type": "array", "items": _SEGMENT_SCHEMA},
        "notes":    {"type": "array", "items": {"type": "string"}},
    },
}

_ITINERARY_TOOL = {
    "type": "function",
    "function": {
        "name": "return_itinerary",
        "description": "Return the complete day-by-day itinerary.",
        "parameters": {
            "type": "object",
            "required": ["title", "days"],
            "properties": {
                "title": {"type": "string"},
                "days":  {"type": "array", "items": _DAY_SCHEMA},
                "notes": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


_client: openai.AsyncOpenAI | None = None


def _get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI()
    return _client


# --------------------------------------------------------------------------- #
# Step 1 — Load Agent Data
# --------------------------------------------------------------------------- #


def load_agent_data(state: PlanningState) -> tuple[bool, list[str]]:
    """Validate agent outputs are present. Returns (ok, errors)."""
    errors: list[str] = []
    if not state.hotel_options:
        errors.append("Hotel Agent returned no options.")
    if not state.event_options:
        errors.append("Event Agent returned no options.")
    if not state.route_options:
        errors.append("Route Agent returned no options.")
    # Restaurant absence is soft — not fatal.
    return (len(errors) == 0, errors)


# --------------------------------------------------------------------------- #
# Step 2 — Budget Allocation
# --------------------------------------------------------------------------- #


def allocate_budget(trip: TripRequest, prefs_style: str, chosen_route: RouteOption | None) -> BudgetBreakdown:
    ratios = _STYLE_RATIOS.get(prefs_style, _STYLE_RATIOS["balanced"])
    total = float(trip.total_budget or 0.0)
    transport = float(chosen_route.total_cost) if chosen_route else 0.0
    remaining_after_transport = max(total - transport, 0.0)

    accommodation = remaining_after_transport * ratios["accommodation"]
    food          = remaining_after_transport * ratios["food"]
    activities    = remaining_after_transport * ratios["activities"]
    other         = remaining_after_transport * ratios["other"]

    allocated = transport + accommodation + food + activities + other
    return BudgetBreakdown(
        transport=transport,
        accommodation=accommodation,
        food=food,
        activities=activities,
        other=other,
        total=total,
        remaining=max(total - allocated, 0.0),
    )


# --------------------------------------------------------------------------- #
# Step 3+5 — Day-by-day scheduling + structured output (LLM Option A)
# --------------------------------------------------------------------------- #


def _serialise_hotels(hs: list[HotelOption]) -> list[dict[str, Any]]:
    return [
        {
            "hotel_id": h.hotel_id, "name": h.name, "star_rating": h.star_rating,
            "price_per_night": h.price_per_night, "total_cost": h.total_cost,
            "lat": h.latitude, "lng": h.longitude, "address": h.address,
        }
        for h in hs
    ]


def _serialise_events(es: list[EventOption]) -> list[dict[str, Any]]:
    return [
        {
            "event_id": e.event_id, "name": e.name, "type": e.type,
            "duration_minutes": e.duration_minutes, "cost": e.cost,
            "lat": e.latitude, "lng": e.longitude,
            "best_time_of_day": e.best_time_of_day,
        }
        for e in es
    ]


def _serialise_restaurants(rs: list[RestaurantOption]) -> list[dict[str, Any]]:
    return [
        {
            "restaurant_id": r.restaurant_id, "name": r.name, "cuisine": r.cuisine,
            "meal_type": r.meal_type, "avg_cost_per_person": r.avg_cost_per_person,
            "rating": r.rating, "lat": r.latitude, "lng": r.longitude,
        }
        for r in rs
    ]


def _serialise_route(r: RouteOption) -> dict[str, Any]:
    return {
        "route_id": r.route_id, "mode": r.mode,
        "total_cost": r.total_cost, "total_duration_minutes": r.total_duration_minutes,
        "segments": [
            {
                "type": s.type,
                "departure_location": s.departure_location,
                "arrival_location": s.arrival_location,
                "departure_time": s.departure_time.isoformat(),
                "arrival_time": s.arrival_time.isoformat(),
                "cost": s.cost,
            }
            for s in r.segments
        ],
    }


def _build_planner_prompt(
    trip: TripRequest,
    user: UserProfile | None,
    route: RouteOption,
    hotels: list[HotelOption],
    restaurants: list[RestaurantOption],
    events: list[EventOption],
    budget: BudgetBreakdown,
) -> str:
    prefs = user.preferences if user else None
    profile_block = {
        "travelers": trip.travelers,
        "travel_style": (prefs.travel_style if prefs else "balanced"),
        "walking_tolerance": (prefs.walking_tolerance if prefs else "medium"),
        "meal_timing": (prefs.meal_timing.model_dump() if prefs and prefs.meal_timing else None),
        "activity_interests": (prefs.activity_interests if prefs else []),
        "preferred_foods": (prefs.preferred_foods if prefs else []),
        "must_include": trip.must_include,
        "must_exclude": trip.must_exclude,
        "special_occasion": trip.special_occasion,
    }

    payload = {
        "trip": {
            "origin": trip.origin,
            "destination": trip.destination,
            "start_date": trip.start_date.isoformat(),
            "end_date":   trip.end_date.isoformat(),
            "num_days":   trip.num_days,
            "currency":   trip.currency,
        },
        "profile": profile_block,
        "budget": budget.model_dump(),
        "chosen_route": _serialise_route(route),
        "hotels":       _serialise_hotels(hotels),
        "restaurants":  _serialise_restaurants(restaurants),
        "events":       _serialise_events(events),
    }

    return (
        "Plan a coherent day-by-day itinerary using ONLY the provided options.\n"
        "Rules:\n"
        "  1. Each day: reasonable pace (not over-packed); respect meal_timing.\n"
        "  2. Group stops that are geographically close within the same day.\n"
        "  3. Insert a short travel/buffer segment between distant locations.\n"
        "  4. Assign one hotel for each night (accommodation_hotel_id).\n"
        "  5. Stay within the food + activities budgets.\n"
        "  6. Include must_include items, exclude must_exclude items.\n"
        "  7. Reference item ids in `item_ref` whenever you use one of the options.\n"
        "  8. Do not invent places that are not in the provided options.\n\n"
        f"DATA:\n{json.dumps(payload, indent=2)}"
    )


_PLANNER_SYSTEM = (
    "You are the Itinerary Planning Engine. Produce a day-by-day plan "
    "that is realistic, geographically sensible, and within budget."
)


async def _run_llm_planner(
    prompt: str,
    extra_messages: list[dict] | None = None,
) -> dict[str, Any]:
    """Call the planner tool. `extra_messages` is appended before the final
    user prompt — used by the repair pass to show a prior draft + conflicts."""
    messages: list[dict] = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": prompt},
    ]
    if extra_messages:
        messages.extend(extra_messages)

    logger.info("tool_call → return_itinerary (mode=%s)",
                "repair" if extra_messages else "initial")
    resp = await _get_client().chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        tools=[_ITINERARY_TOOL],
        tool_choice={"type": "function", "function": {"name": "return_itinerary"}},
        messages=messages,
    )
    tc = resp.choices[0].message.tool_calls[0]
    args = json.loads(tc.function.arguments)
    logger.info(
        "tool_result ← return_itinerary title=%r days=%d segments=%d",
        args.get("title"),
        len(args.get("days") or []),
        sum(len(d.get("segments") or []) for d in (args.get("days") or [])),
    )
    return args


# --------------------------------------------------------------------------- #
# Step 4 — Conflict Resolution (deterministic post-checks)
# --------------------------------------------------------------------------- #


def resolve_conflicts(
    days: list[DayPlan],
    budget: BudgetBreakdown,
) -> list[str]:
    """Deterministic conflict checks that run after the LLM produces a draft.

    Returns human-readable notes for issues we detected. Rather than mutate
    the schedule silently, we surface conflicts so the user sees them.
    """
    notes: list[str] = []

    # Time-order conflict within a day.
    for day in days:
        segments = sorted(day.segments, key=lambda s: s.start_time)
        for a, b in zip(segments, segments[1:]):
            if a.end_time > b.start_time:
                notes.append(
                    f"Day {day.day_number}: '{a.title}' overlaps '{b.title}' "
                    f"({a.end_time} > {b.start_time})."
                )

    # Budget conflicts.
    food_total = sum(s.cost for d in days for s in d.segments if s.type == "meal")
    activities_total = sum(s.cost for d in days for s in d.segments if s.type == "activity")
    if food_total > budget.food * 1.10:
        notes.append(
            f"Food spend ${food_total:.0f} exceeds allocation ${budget.food:.0f}."
        )
    if activities_total > budget.activities * 1.10:
        notes.append(
            f"Activities spend ${activities_total:.0f} exceeds allocation ${budget.activities:.0f}."
        )
    return notes


# --------------------------------------------------------------------------- #
# Assemble Itinerary from LLM output
# --------------------------------------------------------------------------- #


def _parse_time(s: str) -> Any:
    from datetime import time as _time
    hh, mm = s.split(":")[:2]
    return _time(int(hh), int(mm))


def _hydrate_days(
    raw_days: list[dict[str, Any]],
    hotels_by_id: dict[str, HotelOption],
) -> list[DayPlan]:
    days: list[DayPlan] = []
    for rd in raw_days:
        hotel_id = rd.get("accommodation_hotel_id")
        acc = hotels_by_id.get(hotel_id) if hotel_id else None
        segs = [
            TimeSegment(
                id=str(uuid.uuid4()),
                start_time=_parse_time(s["start_time"]),
                end_time=_parse_time(s["end_time"]),
                type=s["type"],
                title=s["title"],
                description=s.get("description", ""),
                location=s.get("location", ""),
                cost=float(s.get("cost") or 0.0),
                item_ref=s.get("item_ref"),
            )
            for s in rd.get("segments", [])
        ]
        day_total = sum(s.cost for s in segs)
        days.append(
            DayPlan(
                day_number=rd["day_number"],
                date=date.fromisoformat(rd["date"]),
                day_name=rd["day_name"],
                location=rd["location"],
                accommodation=acc,
                segments=segs,
                total_cost=day_total + (acc.price_per_night if acc else 0.0),
                notes=rd.get("notes", []),
            )
        )
    return days


# --------------------------------------------------------------------------- #
# Generate-with-repair (shared by initial planning and revision)
# --------------------------------------------------------------------------- #


async def _generate_with_repair(
    prompt: str,
    hotels_by_id: dict[str, HotelOption],
    budget: BudgetBreakdown,
) -> tuple[dict[str, Any], list[DayPlan], list[str]]:
    """Run the planner LLM, and if the draft has conflicts, run a single
    repair pass. Returns (raw_result, hydrated_days, remaining_conflicts).

    One repair pass only — bounds latency and cost while catching the common
    cases (overlapping segments, minor budget overruns) that pure prompting
    tends to produce.
    """
    raw = await _run_llm_planner(prompt)
    days = _hydrate_days(raw.get("days", []), hotels_by_id)
    conflicts = resolve_conflicts(days, budget)

    if not conflicts:
        logger.info("no conflicts — skipping repair pass")
        return raw, days, []

    logger.info("repair pass triggered — %d conflict(s): %s",
                len(conflicts), conflicts)

    repair_msg = (
        "Your previous draft had these issues — return a corrected itinerary "
        "using the same tool. Keep as much of the plan as possible; only "
        "change what's needed to resolve the issues.\n\n"
        f"DRAFT:\n{json.dumps(raw, indent=2)}\n\n"
        f"ISSUES:\n- " + "\n- ".join(conflicts)
    )
    raw2 = await _run_llm_planner(
        prompt,
        extra_messages=[
            {"role": "assistant", "content": json.dumps(raw)},
            {"role": "user", "content": repair_msg},
        ],
    )
    days2 = _hydrate_days(raw2.get("days", []), hotels_by_id)
    remaining = resolve_conflicts(days2, budget)
    logger.info("repair pass done — %d conflict(s) remain", len(remaining))
    return raw2, days2, remaining


def _compute_total_cost(days: list[DayPlan], budget: BudgetBreakdown) -> float:
    return (
        budget.transport
        + sum((d.accommodation.price_per_night if d.accommodation else 0.0) for d in days)
        + sum(s.cost for d in days for s in d.segments)
    )


# --------------------------------------------------------------------------- #
# Entry point — runs the whole planning engine
# --------------------------------------------------------------------------- #


async def run_planning(state: PlanningState) -> Itinerary:
    """Execute Steps 1–5 and return the assembled Itinerary."""
    trip = state.trip_request
    if trip is None:
        raise ValueError("PlanningState.trip_request is required")
    logger.info(
        "run_planning: %s → %s | %s–%s | %d travelers | $%.0f %s",
        trip.origin, trip.destination, trip.start_date, trip.end_date,
        trip.travelers, trip.total_budget, trip.currency,
    )

    # Step 1 — Load
    ok, errors = load_agent_data(state)
    if not ok:
        raise RuntimeError("Missing agent outputs: " + "; ".join(errors))

    # Pick the cheapest route as the default (Phase-1 heuristic).
    chosen_route = min(state.route_options, key=lambda r: r.total_cost)

    style = (
        state.user_profile.preferences.travel_style
        if state.user_profile else "balanced"
    )

    # Step 2 — Budget
    budget = allocate_budget(trip, style, chosen_route)

    # Step 3+4+5 — Schedule, generate, and repair any conflicts.
    prompt = _build_planner_prompt(
        trip=trip,
        user=state.user_profile,
        route=chosen_route,
        hotels=state.hotel_options,
        restaurants=state.restaurant_options,
        events=state.event_options,
        budget=budget,
    )
    hotels_by_id = {h.hotel_id: h for h in state.hotel_options}
    raw, days, remaining_conflicts = await _generate_with_repair(prompt, hotels_by_id, budget)

    return Itinerary(
        trip_id=str(uuid.uuid4()),
        user_id=(state.user_profile.user_id if state.user_profile else "anonymous"),
        title=raw.get("title") or f"{trip.num_days}-day trip to {trip.destination}",
        days=days,
        total_cost=_compute_total_cost(days, budget),
        budget_breakdown=budget,
        notes=(raw.get("notes") or []) + remaining_conflicts,
        created_at=datetime.utcnow(),
        version=1,
    )


# --------------------------------------------------------------------------- #
# Revision loop (§4 Step 6)
# --------------------------------------------------------------------------- #


def _serialise_itinerary_for_llm(it: Itinerary) -> dict[str, Any]:
    """Compact JSON view of an existing itinerary — day/segment shape the
    planner tool schema already understands, so the LLM can edit in place."""
    return {
        "title": it.title,
        "version": it.version,
        "total_cost": it.total_cost,
        "notes": it.notes,
        "days": [
            {
                "day_number": d.day_number,
                "date": d.date.isoformat(),
                "day_name": d.day_name,
                "location": d.location,
                "accommodation_hotel_id": (d.accommodation.hotel_id if d.accommodation else None),
                "segments": [
                    {
                        "start_time": s.start_time.strftime("%H:%M"),
                        "end_time": s.end_time.strftime("%H:%M"),
                        "type": s.type,
                        "title": s.title,
                        "description": s.description,
                        "location": s.location,
                        "cost": s.cost,
                        "item_ref": s.item_ref,
                    }
                    for s in d.segments
                ],
                "notes": d.notes,
            }
            for d in it.days
        ],
    }


def _build_revision_prompt(
    trip: TripRequest,
    user: UserProfile | None,
    route: RouteOption,
    hotels: list[HotelOption],
    restaurants: list[RestaurantOption],
    events: list[EventOption],
    budget: BudgetBreakdown,
    current: Itinerary,
    feedback: str,
) -> str:
    base = _build_planner_prompt(trip, user, route, hotels, restaurants, events, budget)
    return (
        "REVISE the existing itinerary below according to the user's feedback.\n"
        "Keep everything the user did NOT complain about — only change what's\n"
        "needed to satisfy the feedback. All rules from the planner still apply.\n\n"
        f"USER FEEDBACK:\n{feedback}\n\n"
        f"CURRENT ITINERARY:\n{json.dumps(_serialise_itinerary_for_llm(current), indent=2)}\n\n"
        f"{base}"
    )


def _diff_summary(before: Itinerary, after: Itinerary) -> str:
    """Human-readable diff of segment counts, cost, and hotel swaps."""
    lines: list[str] = []
    if abs(after.total_cost - before.total_cost) > 1.0:
        delta = after.total_cost - before.total_cost
        sign = "+" if delta >= 0 else "-"
        lines.append(f"Total cost {sign}${abs(delta):.0f} (now ${after.total_cost:.0f}).")
    for b, a in zip(before.days, after.days):
        b_hotel = b.accommodation.name if b.accommodation else None
        a_hotel = a.accommodation.name if a.accommodation else None
        if b_hotel != a_hotel:
            lines.append(f"Day {a.day_number}: hotel {b_hotel!r} → {a_hotel!r}.")
        if len(b.segments) != len(a.segments):
            lines.append(
                f"Day {a.day_number}: {len(b.segments)} → {len(a.segments)} segments."
            )
    if len(after.days) != len(before.days):
        lines.append(f"Trip length: {len(before.days)} → {len(after.days)} days.")
    return " ".join(lines) if lines else "Minor adjustments applied."


async def revise_itinerary(state: PlanningState) -> Itinerary:
    """Re-plan the current itinerary using state.revision_feedback."""
    trip = state.trip_request
    current = state.itinerary
    feedback = state.revision_feedback
    if trip is None or current is None or not feedback:
        raise ValueError(
            "revise_itinerary requires trip_request, itinerary, and revision_feedback"
        )
    logger.info("revise_itinerary: v%d → v%d | feedback=%r",
                current.version, current.version + 1, feedback)

    ok, errors = load_agent_data(state)
    if not ok:
        raise RuntimeError("Missing agent outputs: " + "; ".join(errors))

    chosen_route = min(state.route_options, key=lambda r: r.total_cost)
    style = (
        state.user_profile.preferences.travel_style
        if state.user_profile else "balanced"
    )
    budget = allocate_budget(trip, style, chosen_route)

    prompt = _build_revision_prompt(
        trip=trip,
        user=state.user_profile,
        route=chosen_route,
        hotels=state.hotel_options,
        restaurants=state.restaurant_options,
        events=state.event_options,
        budget=budget,
        current=current,
        feedback=feedback,
    )
    hotels_by_id = {h.hotel_id: h for h in state.hotel_options}
    raw, days, remaining_conflicts = await _generate_with_repair(prompt, hotels_by_id, budget)

    return Itinerary(
        trip_id=current.trip_id,
        user_id=current.user_id,
        title=raw.get("title") or current.title,
        days=days,
        total_cost=_compute_total_cost(days, budget),
        budget_breakdown=budget,
        notes=(raw.get("notes") or []) + remaining_conflicts,
        created_at=datetime.utcnow(),
        version=current.version + 1,
    )
