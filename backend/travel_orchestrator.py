"""Travel Orchestrator — thin adapter over the LangGraph flows.

The orchestration described in itenary_agent.md §1 / §4.5 is now expressed
as a compiled `StateGraph` in `backend/graph/`. This module's only jobs are:

  1. Build the initial `PlanningState` from the incoming request.
  2. Invoke the appropriate compiled graph (`_TRAVEL_GRAPH` for /plan,
     `_REVISE_GRAPH` for /revise).
  3. Project the final `PlanningState` back into the API response shapes
     (`PlanResponse` / `ReviseResponse`) the frontend expects.

Every step that used to live here — classify → route → dispatch → planning
— is now a graph node. There are no plain-Python "agent" functions in this
file anymore.
"""

from __future__ import annotations

import logging

from agent_models import (
    IntentClassification,
    PlanningState,
    PlanRequest,
    PlanResponse,
    ReviseRequest,
    ReviseResponse,
)
from graph import build_revise_graph, build_travel_graph

logger = logging.getLogger("travel_orchestrator")


# Compile the graphs once at import time — cheap, deterministic, avoids
# rebuilding the topology on every request.
_TRAVEL_GRAPH = build_travel_graph()
_REVISE_GRAPH = build_revise_graph()


# --------------------------------------------------------------------------- #
# /plan entry point
# --------------------------------------------------------------------------- #


async def plan(req: PlanRequest) -> PlanResponse:
    logger.info("plan: incoming message=%r", req.message)
    initial = PlanningState(
        user_profile=req.user_profile,
        trip_request=req.trip_request,
        incoming_message=req.message,
        history=list(req.history or []),
    )
    final_dict = await _TRAVEL_GRAPH.ainvoke(initial)
    final = PlanningState.model_validate(final_dict)
    return _state_to_plan_response(final)


def _state_to_plan_response(state: PlanningState) -> PlanResponse:
    intent = state.intent or IntentClassification(route="conversational", confidence=0.0)
    route = intent.route

    if route == "conversational":
        return PlanResponse(route=route, intent=intent, message=state.response_message)

    if state.followup_question:
        return PlanResponse(
            route=route,
            intent=intent,
            followup_question=state.followup_question,
            message=state.followup_question,
        )

    if route == "direct":
        return PlanResponse(
            route=route,
            intent=intent,
            direct_result=state.direct_result,
            message=state.response_message,
        )

    # FULL — itinerary may still be missing if the load step failed.
    return PlanResponse(
        route=route,
        intent=intent,
        itinerary=state.itinerary,
        message=state.response_message
            or "Sorry, I couldn't put a plan together.",
    )


# --------------------------------------------------------------------------- #
# /revise entry point
# --------------------------------------------------------------------------- #


async def revise(req: ReviseRequest) -> ReviseResponse:
    logger.info(
        "revise: itinerary v%d feedback=%r",
        req.itinerary.version, req.feedback,
    )
    initial = PlanningState(
        user_profile=req.user_profile,
        trip_request=req.trip_request,
        itinerary=req.itinerary,
        revision_feedback=req.feedback,
        route_options=list(req.route_options),
        hotel_options=list(req.hotel_options),
        restaurant_options=list(req.restaurant_options),
        event_options=list(req.event_options),
        phase="revision",
    )
    final_dict = await _REVISE_GRAPH.ainvoke(initial)
    final = PlanningState.model_validate(final_dict)

    conflicts_remaining = [
        n for n in (final.itinerary.notes if final.itinerary else [])
        if "overlaps" in n or "exceeds" in n
    ]
    return ReviseResponse(
        itinerary=final.itinerary,
        changes_summary=final.changes_summary,
        conflicts_remaining=conflicts_remaining,
        message=final.response_message or final.changes_summary,
    )
