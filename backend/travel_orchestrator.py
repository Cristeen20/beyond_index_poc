"""Travel Orchestrator — thin adapter over the LangGraph flows.

The orchestration described in itenary_agent.md §1 / §4.5 is now expressed
as a compiled `StateGraph` in `backend/graph/`. This module's only jobs are:

  1. Build the initial `PlanningState` from the incoming request.
  2. Invoke the compiled graph.
     - /plan uses `_TRAVEL_GRAPH` (checkpointed by session_id): first call
       for a session runs from START; subsequent calls resume the
       `wait_for_next_message` interrupt with the new message.
     - /revise is a thin wrapper that invokes the revise subgraph
       directly for callers that already have an explicit revision request.
  3. Project the final `PlanningState` back into the API response shapes.
"""

from __future__ import annotations

import logging

from langgraph.types import Command

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
# rebuilding the topology on every request. `_TRAVEL_GRAPH` is checkpointed
# so a session (thread_id) persists across turns.
_TRAVEL_GRAPH = build_travel_graph()
_REVISE_GRAPH = build_revise_graph()


# --------------------------------------------------------------------------- #
# /plan entry point
# --------------------------------------------------------------------------- #


async def plan(req: PlanRequest) -> PlanResponse:
    logger.info(
        "plan: session=%s message=%r", req.session_id, req.message,
    )
    config = {"configurable": {"thread_id": req.session_id}}

    # Is this a fresh session or a resume from wait_for_next_message?
    snapshot = _TRAVEL_GRAPH.get_state(config)
    is_resume = bool(snapshot and snapshot.next)

    if is_resume:
        logger.info("plan: resuming session=%s (paused at %s)",
                    req.session_id, snapshot.next)
        final_dict = await _TRAVEL_GRAPH.ainvoke(
            Command(resume=req.message), config=config
        )
    else:
        initial = PlanningState(
            user_profile=req.user_profile,
            trip_request=req.trip_request,
            incoming_message=req.message,
            history=list(req.history or []),
        )
        final_dict = await _TRAVEL_GRAPH.ainvoke(initial, config=config)

    final = PlanningState.model_validate(final_dict)
    return _state_to_plan_response(final, req.session_id)


def _state_to_plan_response(state: PlanningState, session_id: str) -> PlanResponse:
    intent = state.intent or IntentClassification(route="conversational", confidence=0.0)
    route = intent.route

    if route == "conversational":
        return PlanResponse(
            route=route, intent=intent,
            message=state.response_message, session_id=session_id,
        )

    if state.followup_question:
        return PlanResponse(
            route=route,
            intent=intent,
            followup_question=state.followup_question,
            message=state.followup_question,
            session_id=session_id,
        )

    if route == "direct":
        return PlanResponse(
            route=route,
            intent=intent,
            direct_result=state.direct_result,
            message=state.response_message,
            session_id=session_id,
        )

    # FULL or REVISE — itinerary may still be missing if the load step failed.
    return PlanResponse(
        route=route,
        intent=intent,
        itinerary=state.itinerary,
        message=state.response_message
            or "Sorry, I couldn't put a plan together.",
        session_id=session_id,
    )


# --------------------------------------------------------------------------- #
# /revise entry point — kept as a thin wrapper for callers with an explicit
# ReviseRequest (frontend still POSTs to /revise for the approve/revise UI).
# Not session-checkpointed: the request payload carries everything the
# subgraph needs.
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
