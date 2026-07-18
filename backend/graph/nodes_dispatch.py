"""Dispatch nodes — one per sub-agent, plus the direct-path merger.

The four dispatch nodes fire in parallel from the travel graph. Each is
self-gating: if its agent isn't in `intent.target_agents`, it returns an
empty state update. That way the graph topology stays the same on both
DIRECT (subset) and FULL (all four) paths.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_models import PlanningState
from sub_agents import (
    run_event_agent,
    run_hotel_agent,
    run_restaurant_agent,
    run_route_agent,
)

logger = logging.getLogger("graph.dispatch")


def _prefs(state: PlanningState):
    return state.user_profile.preferences if state.user_profile else None


def _selected(state: PlanningState, agent: str) -> bool:
    return bool(state.intent) and agent in state.intent.target_agents


# --------------------------------------------------------------------------- #
# Per-agent dispatch nodes
# --------------------------------------------------------------------------- #


async def dispatch_hotel(state: PlanningState) -> dict:
    if not _selected(state, "hotel") or state.trip_request is None:
        return {}
    options = await run_hotel_agent(state.trip_request, _prefs(state))
    logger.info("dispatch_hotel → %d option(s)", len(options))
    return {"hotel_options": options}


async def dispatch_restaurant(state: PlanningState) -> dict:
    if not _selected(state, "restaurant") or state.trip_request is None:
        return {}
    options = await run_restaurant_agent(state.trip_request, _prefs(state))
    logger.info("dispatch_restaurant → %d option(s)", len(options))
    return {"restaurant_options": options}


async def dispatch_route(state: PlanningState) -> dict:
    if not _selected(state, "route") or state.trip_request is None:
        return {}
    options = await run_route_agent(state.trip_request, _prefs(state))
    logger.info("dispatch_route → %d option(s)", len(options))
    return {"route_options": options}


async def dispatch_event(state: PlanningState) -> dict:
    if not _selected(state, "event") or state.trip_request is None:
        return {}
    options = await run_event_agent(state.trip_request, _prefs(state))
    logger.info("dispatch_event → %d option(s)", len(options))
    return {"event_options": options}


# --------------------------------------------------------------------------- #
# Post-dispatch synchronisation + direct-path merge
# --------------------------------------------------------------------------- #


def post_dispatch(state: PlanningState) -> dict:
    """Join point for the four parallel dispatch nodes. Records which agents
    returned data so downstream conditional edges can inspect it."""
    received = {
        "route": bool(state.route_options),
        "hotel": bool(state.hotel_options),
        "restaurant": bool(state.restaurant_options),
        "event": bool(state.event_options),
    }
    logger.info("post_dispatch received=%s", received)
    return {"agent_outputs_received": received, "phase": "agent_dispatch"}


def _flatten_options(state: PlanningState) -> list[dict[str, Any]]:
    """Merge the direct flow's per-agent outputs into a single ranked list."""
    merged: list[dict[str, Any]] = []
    for agent_name, options in (
        ("hotel", state.hotel_options),
        ("restaurant", state.restaurant_options),
        ("route", state.route_options),
        ("event", state.event_options),
    ):
        for opt in options[:5]:  # top-5 per agent keeps the reply compact
            merged.append({"agent": agent_name, **opt.model_dump(mode="json")})
    return merged


def merge_direct(state: PlanningState) -> dict:
    """Assemble the DIRECT-path response (§4.5 Step 4). Not an Itinerary —
    a lightweight list of raw options plus a non-escalating upgrade offer."""
    merged = _flatten_options(state)
    offer = (
        "Here are the top matches. Want me to plan the full trip around these?"
        if merged
        else "I couldn't find matches — try broadening the search?"
    )
    logger.info("merge_direct → %d merged result(s)", len(merged))
    return {
        "direct_result": merged,
        "response_message": offer,
        "phase": "direct_answer",
    }
