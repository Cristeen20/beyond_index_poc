"""Travel Orchestrator — thin adapter over the LangGraph flow.

The orchestration described in itenary_agent.md §1 / §4.5 is now expressed
as a compiled `StateGraph` in `backend/graph/`. This module's only jobs are:

  1. Build the initial `PlanningState` from the incoming request.
  2. Invoke the compiled `_TRAVEL_GRAPH` (checkpointed by session_id):
     first call for a session runs from START; subsequent calls resume the
     `wait_for_next_message` interrupt with the new message. Revisions are
     handled as an internal branch of this graph.
  3. Project the final `PlanningState` back into the API response shape.
"""

from __future__ import annotations

import logging

from langgraph.types import Command

from agent_models import (
    IntentClassification,
    PlanningState,
    PlanRequest,
    PlanResponse,
)
from graph import build_travel_graph

logger = logging.getLogger("travel_orchestrator")


# Compiled once per process — cheap, deterministic, avoids rebuilding the
# topology on every request. Checkpointed so a session (thread_id) persists
# across turns.
#
# Built on first use rather than at import: the Postgres checkpointer must
# be constructed inside a running event loop (see graph/session.py). main.py's
# lifespan calls init_checkpointer() first, so by the time a request lands
# the checkpointer is already built and this only compiles the topology.
_TRAVEL_GRAPH = None


def _get_graph():
    global _TRAVEL_GRAPH
    if _TRAVEL_GRAPH is None:
        _TRAVEL_GRAPH = build_travel_graph()
    return _TRAVEL_GRAPH


# --------------------------------------------------------------------------- #
# /plan entry point
# --------------------------------------------------------------------------- #


async def plan(req: PlanRequest) -> PlanResponse:
    logger.info(
        "plan: session=%s message=%r", req.session_id, req.message,
    )
    config = {"configurable": {"thread_id": req.session_id}}

    graph = _get_graph()

    # Is this a fresh session or a resume from wait_for_next_message?
    # aget_state, not get_state: AsyncPostgresSaver implements only the
    # async checkpointer methods and raises on the sync ones.
    snapshot = await graph.aget_state(config)
    is_resume = bool(snapshot and snapshot.next)

    if is_resume:
        logger.info("plan: resuming session=%s (paused at %s)",
                    req.session_id, snapshot.next)
        # If the frontend sent an option_action (button/select on an
        # OptionsCard), forward it in the resume payload alongside the
        # message. wait_for_next_message unpacks the dict and stashes
        # option_action on state so the pre-planning parsers see it.
        resume_payload: dict | str
        if req.option_action is not None:
            resume_payload = {
                "message": req.message,
                "option_action": req.option_action.model_dump(mode="json"),
            }
        else:
            resume_payload = req.message
        final_dict = await graph.ainvoke(
            Command(resume=resume_payload), config=config
        )
    else:
        initial = PlanningState(
            user_profile=req.user_profile,
            trip_request=req.trip_request,
            incoming_message=req.message,
            history=list(req.history or []),
        )
        final_dict = await graph.ainvoke(initial, config=config)

    final = PlanningState.model_validate(final_dict)
    return _state_to_plan_response(final, req.session_id)


def _state_to_plan_response(state: PlanningState, session_id: str) -> PlanResponse:
    intent = state.intent or IntentClassification(route="conversational", confidence=0.0)
    route = intent.route

    # Pre-planning payload takes precedence — it's the current UI surface
    # regardless of route. The response_message is the payload's title /
    # explanatory blurb (or an LLM answer to a question about shown items).
    if state.options_payload is not None:
        return PlanResponse(
            route=route,
            intent=intent,
            options_payload=state.options_payload,
            itinerary=state.itinerary,
            message=state.response_message,
            session_id=session_id,
        )

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


