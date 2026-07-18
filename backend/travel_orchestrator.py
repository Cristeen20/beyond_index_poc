"""Travel Orchestrator — the entry-point that runs the dual-path flow
described in §1 / §4.5 of itenary_agent.md.

Router decides DIRECT vs FULL:
  - DIRECT: check slot gate → parallel dispatch subset of agents → merged options
  - FULL:   intake → parallel dispatch of ALL agents → Itinerary Agent
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import openai

logger = logging.getLogger("travel_orchestrator")


_CONVERSATIONAL_SYSTEM = (
    "You are a friendly, knowledgeable travel assistant. The user's message has "
    "been classified as conversational — answer from your own knowledge, no "
    "tools or live data. Keep replies natural, concise, and practical. For "
    "questions about specific hotels/restaurants/flights/attractions, remind "
    "the user you can search for those if they'd like."
)


_llm_client: openai.AsyncOpenAI | None = None


def _get_llm() -> openai.AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = openai.AsyncOpenAI()
    return _llm_client


async def _answer_conversational(
    message: str,
    history: list[dict] | None,
) -> str:
    messages: list[dict] = [{"role": "system", "content": _CONVERSATIONAL_SYSTEM}]
    for m in (history or []):
        role = m.get("role") if isinstance(m, dict) else m.role
        content = m.get("content") if isinstance(m, dict) else m.content
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    resp = await _get_llm().chat.completions.create(
        model="gpt-4o",
        max_tokens=800,
        messages=messages,
    )
    return resp.choices[0].message.content or ""

from agent_models import (
    IntentClassification,
    PlanRequest,
    PlanResponse,
    PlanningState,
    ReviseRequest,
    ReviseResponse,
    TripRequest,
    UserProfile,
)
from intake_router import build_slot_question, classify, slot_gate
from itinerary_agent import revise_itinerary, run_planning
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
    logger.info("plan: incoming message=%r", req.message)
    state = PlanningState(user_profile=req.user_profile, trip_request=req.trip_request)

    # Step: Intake Router (§1.5)
    intent = await classify(req.message, req.history, req.user_profile)
    state.intent = intent
    state.phase = "routing"
    logger.info("plan: routed → %s | agents=%s | confidence=%.2f",
                intent.route, intent.target_agents, intent.confidence)

    # ------------------------------------------------------------------ #
    # CONVERSATIONAL flow — no APIs, just an LLM reply.
    # ------------------------------------------------------------------ #
    if intent.route == "conversational":
        logger.info("plan: ENTERING CONVERSATIONAL branch (no agents dispatched)")
        state.phase = "direct_answer"
        text = await _answer_conversational(req.message, req.history)
        logger.info("plan: CONVERSATIONAL complete → %d chars", len(text))
        return PlanResponse(
            route="conversational",
            intent=intent,
            message=text,
        )

    # ------------------------------------------------------------------ #
    # DIRECT flow (§4.5)
    # ------------------------------------------------------------------ #
    if intent.route == "direct":
        logger.info(
            "plan: ENTERING DIRECT branch (agents=%s, extracted_slots=%s)",
            intent.target_agents, intent.extracted_slots,
        )
        state.phase = "direct_answer"

        # Step 2: slot gate — if anything's missing, ask ONE targeted question.
        trip = _hydrate_trip_request(intent, req.trip_request)
        if trip is None or intent.missing_required_slots:
            missing = intent.missing_required_slots or slot_gate(
                intent.target_agents, intent.extracted_slots
            )
            question = build_slot_question(missing)
            logger.info("direct: slot gate missing=%s → question=%r", missing, question)
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
        logger.info(
            "plan: DIRECT complete → %d merged result(s), no itinerary composed",
            len(direct_result),
        )

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
    logger.info(
        "plan: ENTERING FULL branch (all 4 agents + itinerary composition)"
    )
    trip = _hydrate_trip_request(intent, req.trip_request)
    if trip is None:
        # Router said FULL but we don't have enough to build a TripRequest.
        # Ask for the single most important missing slot.
        missing = slot_gate(["route", "hotel"], intent.extracted_slots)
        question = build_slot_question(missing)
        logger.info("plan: FULL slot gate missing=%s → question=%r", missing, question)
        return PlanResponse(
            route="full",
            intent=intent,
            followup_question=question,
            message=question,
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
    logger.info(
        "plan: FULL complete → itinerary v%d, %d days, $%.0f total",
        itinerary.version, len(itinerary.days), itinerary.total_cost,
    )

    return PlanResponse(
        route="full",
        intent=intent,
        itinerary=itinerary,
        message=f"Here's your {itinerary.days[0].date}–{itinerary.days[-1].date} itinerary for {trip.destination}.",
    )


# --------------------------------------------------------------------------- #
# Revision entry point (§4 Step 6)
# --------------------------------------------------------------------------- #


async def revise(req: ReviseRequest) -> ReviseResponse:
    """Re-plan the given itinerary using the user's feedback.

    Reconstructs a PlanningState from the request. If the client cached the
    agent options and sent them back, we reuse them; otherwise we refetch.
    """
    state = PlanningState(
        user_profile=req.user_profile,
        trip_request=req.trip_request,
        itinerary=req.itinerary,
        revision_feedback=req.feedback,
        route_options=req.route_options,
        hotel_options=req.hotel_options,
        restaurant_options=req.restaurant_options,
        event_options=req.event_options,
        phase="revision",
    )

    needs_refetch = not (
        state.route_options
        and state.hotel_options
        and state.event_options
    )
    if needs_refetch:
        prefs = req.user_profile.preferences if req.user_profile else None
        outputs = await dispatch_agents(
            ["route", "hotel", "restaurant", "event"], req.trip_request, prefs
        )
        state.route_options       = outputs.get("route", [])
        state.hotel_options       = outputs.get("hotel", [])
        state.restaurant_options  = outputs.get("restaurant", [])
        state.event_options       = outputs.get("event", [])

        # A revision that swaps to a hotel the client already had cached would
        # otherwise lose that reference; merge the incoming hotels back in so
        # the LLM sees both the freshly-fetched and previously-chosen options.
        seen = {h.hotel_id for h in state.hotel_options}
        state.hotel_options.extend(
            h for h in req.hotel_options if h.hotel_id not in seen
        )

    # Import here to avoid the top-level cycle noise; also keeps the diff
    # helper co-located with the itinerary agent that owns it.
    from itinerary_agent import _diff_summary

    new_itinerary = await revise_itinerary(state)
    summary = _diff_summary(req.itinerary, new_itinerary)

    return ReviseResponse(
        itinerary=new_itinerary,
        changes_summary=summary,
        conflicts_remaining=[
            n for n in new_itinerary.notes if "overlaps" in n or "exceeds" in n
        ],
        message=summary,
    )
