"""Travel Orchestrator — the entry-point that runs the dual-path flow
described in §1 / §4.5 of itenary_agent.md.

Router decides DIRECT vs FULL:
  - DIRECT: check slot gate → parallel dispatch subset of agents → merged options
  - FULL:   intake → parallel dispatch of ALL agents → Itinerary Agent
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from agent_models import (
    IntentClassification,
    PlanRequest,
    PlanResponse,
    PlanningState,
    TripRequest,
    UserProfile,
)
from intake_router import build_slot_question, classify, slot_gate
from itinerary_agent import run_planning
from sub_agents import dispatch_agents


# --------------------------------------------------------------------------- #
# Slot hydration — turn extracted slots + explicit trip_request into a TripRequest
# --------------------------------------------------------------------------- #


def _hydrate_trip_request(
    intent: IntentClassification,
    explicit: TripRequest | None,
) -> TripRequest | None:
    """Prefer explicitly-provided TripRequest; otherwise build from slots."""
    if explicit is not None:
        return explicit

    slots = intent.extracted_slots
    destination = slots.get("destination")
    if not destination:
        return None

    origin = slots.get("origin", "Unknown")
    start = _parse_date(slots.get("start_date") or slots.get("dates"))
    end = _parse_date(slots.get("end_date"))
    if start is None:
        # sensible default: near-future 3-day trip
        start = date.today() + timedelta(days=30)
    if end is None:
        end = start + timedelta(days=2)

    travelers = int(slots.get("travelers") or 1)
    budget = float(slots.get("budget") or 0.0)
    currency = slots.get("currency", "USD")

    return TripRequest(
        origin=origin,
        destination=destination,
        start_date=start,
        end_date=end,
        travelers=travelers,
        total_budget=budget,
        currency=currency,
    )


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Direct-path helpers
# --------------------------------------------------------------------------- #


def _flatten_direct_results(agent_outputs: dict[str, list[Any]]) -> list[dict]:
    """Merge the direct flow's per-agent outputs into a single ranked list."""
    merged: list[dict] = []
    for agent_name, options in agent_outputs.items():
        for opt in options[:5]:  # top-5 per agent keeps the reply compact
            merged.append({"agent": agent_name, **opt.model_dump(mode="json")})
    return merged


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


async def plan(req: PlanRequest) -> PlanResponse:
    state = PlanningState(user_profile=req.user_profile, trip_request=req.trip_request)

    # Step: Intake Router (§1.5)
    intent = await classify(req.message, req.history, req.user_profile)
    state.intent = intent
    state.phase = "routing"

    # ------------------------------------------------------------------ #
    # DIRECT flow (§4.5)
    # ------------------------------------------------------------------ #
    if intent.route == "direct":
        state.phase = "direct_answer"

        # Step 2: slot gate — if anything's missing, ask ONE targeted question.
        trip = _hydrate_trip_request(intent, req.trip_request)
        if trip is None or intent.missing_required_slots:
            missing = intent.missing_required_slots or slot_gate(
                intent.target_agents, intent.extracted_slots
            )
            question = build_slot_question(missing)
            return PlanResponse(
                route="direct",
                intent=intent,
                followup_question=question,
                message=question,
            )

        # Step 3: parallel dispatch (subset only)
        prefs = req.user_profile.preferences if req.user_profile else None
        agent_outputs = await dispatch_agents(intent.target_agents, trip, prefs)

        # Step 4: merge into a targeted response (NOT an Itinerary)
        direct_result = _flatten_direct_results(agent_outputs)
        state.direct_result = direct_result

        # Step 5: offer upgrade (non-escalating)
        offer = (
            "Here are the top matches. Want me to plan the full trip around these?"
            if direct_result
            else "I couldn't find matches — try broadening the search?"
        )
        return PlanResponse(
            route="direct",
            intent=intent,
            direct_result=direct_result,
            message=offer,
        )

    # ------------------------------------------------------------------ #
    # FULL flow — §2 intake → dispatch all 4 → Itinerary Agent (§4)
    # ------------------------------------------------------------------ #
    trip = _hydrate_trip_request(intent, req.trip_request)
    if trip is None:
        # Router said FULL but we don't have enough to build a TripRequest.
        # Ask for the single most important missing slot.
        missing = slot_gate(["route", "hotel"], intent.extracted_slots)
        return PlanResponse(
            route="full",
            intent=intent,
            followup_question=build_slot_question(missing),
            message=build_slot_question(missing),
        )

    state.trip_request = trip
    state.phase = "agent_dispatch"

    prefs = req.user_profile.preferences if req.user_profile else None
    agent_outputs = await dispatch_agents(
        ["route", "hotel", "restaurant", "event"], trip, prefs
    )
    state.route_options       = agent_outputs.get("route", [])
    state.hotel_options       = agent_outputs.get("hotel", [])
    state.restaurant_options  = agent_outputs.get("restaurant", [])
    state.event_options       = agent_outputs.get("event", [])
    state.agent_outputs_received = {k: bool(v) for k, v in agent_outputs.items()}

    state.phase = "planning"
    itinerary = await run_planning(state)
    state.itinerary = itinerary
    state.phase = "review"

    return PlanResponse(
        route="full",
        intent=intent,
        itinerary=itinerary,
        message=f"Here's your {itinerary.days[0].date}–{itinerary.days[-1].date} itinerary for {trip.destination}.",
    )
