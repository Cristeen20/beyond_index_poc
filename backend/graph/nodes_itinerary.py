"""Itinerary sub-graph nodes (§4 of itenary_agent.md).

Each step of the design's Load → Budget → Schedule → Conflict → Generate
pipeline is a node here. The pure helpers (`allocate_budget`,
`resolve_conflicts`, `_hydrate_days`, prompt builders) live in
`itinerary_agent.py` and are imported — they are not agents in the LLM
sense, just serializers and rule checks.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from agent_models import Itinerary, PlanningState
from itinerary_agent import (
    _build_planner_prompt,
    _build_revision_prompt,
    _compute_total_cost,
    _diff_summary,
    _hydrate_days,
    _run_llm_planner,
    allocate_budget,
    load_agent_data,
    resolve_conflicts,
)
from sub_agents import dispatch_agents

logger = logging.getLogger("graph.itinerary")


MAX_REPAIR_ATTEMPTS = 1


# --------------------------------------------------------------------------- #
# Step 1 — Load
# --------------------------------------------------------------------------- #


def load_agent_data_node(state: PlanningState) -> dict:
    ok, errors = load_agent_data(state)
    if not ok:
        logger.warning("load_agent_data → errors=%s", errors)
        return {"error_notes": errors}
    logger.info("load_agent_data → ok")
    return {}


# --------------------------------------------------------------------------- #
# Step 2 — Budget
# --------------------------------------------------------------------------- #


def allocate_budget_node(state: PlanningState) -> dict:
    trip = state.trip_request
    if trip is None:
        return {}
    chosen = min(state.route_options, key=lambda r: r.total_cost)
    style = (
        state.user_profile.preferences.travel_style
        if state.user_profile else "balanced"
    )
    budget = allocate_budget(trip, style, chosen)
    logger.info(
        "allocate_budget → route=%s $%.0f | budget total=$%.0f accom=$%.0f",
        chosen.mode, chosen.total_cost, budget.total, budget.accommodation,
    )
    return {"chosen_route": chosen, "budget": budget}


# --------------------------------------------------------------------------- #
# Step 3+5 — LLM planner (initial pass)
# --------------------------------------------------------------------------- #


async def run_llm_planner_node(state: PlanningState) -> dict:
    prompt = _build_planner_prompt(
        trip=state.trip_request,
        user=state.user_profile,
        route=state.chosen_route,
        hotels=state.hotel_options,
        restaurants=state.restaurant_options,
        events=state.event_options,
        budget=state.budget,
    )
    raw = await _run_llm_planner(prompt)
    return {"draft_itinerary": raw, "phase": "planning"}


async def run_llm_planner_revision_node(state: PlanningState) -> dict:
    """LLM pass for the revision graph — uses _build_revision_prompt so the
    model sees the prior itinerary + user feedback and edits in place."""
    prompt = _build_revision_prompt(
        trip=state.trip_request,
        user=state.user_profile,
        route=state.chosen_route,
        hotels=state.hotel_options,
        restaurants=state.restaurant_options,
        events=state.event_options,
        budget=state.budget,
        current=state.itinerary,
        feedback=state.revision_feedback,
    )
    raw = await _run_llm_planner(prompt)
    return {"draft_itinerary": raw, "phase": "revision"}


# --------------------------------------------------------------------------- #
# Step 4 — Conflict detection + bounded repair
# --------------------------------------------------------------------------- #


def _hotels_by_id(state: PlanningState):
    return {h.hotel_id: h for h in state.hotel_options}


def check_conflicts_node(state: PlanningState) -> dict:
    days = _hydrate_days(state.draft_itinerary.get("days", []), _hotels_by_id(state))
    notes = resolve_conflicts(days, state.budget)
    logger.info(
        "check_conflicts → %d conflict(s) attempt=%d",
        len(notes), state.repair_attempts,
    )
    return {"conflict_notes": notes}


async def repair_planner_node(state: PlanningState) -> dict:
    """Single-pass repair — matches the previous _generate_with_repair loop."""
    prompt = _build_planner_prompt(
        trip=state.trip_request,
        user=state.user_profile,
        route=state.chosen_route,
        hotels=state.hotel_options,
        restaurants=state.restaurant_options,
        events=state.event_options,
        budget=state.budget,
    )
    repair_msg = (
        "Your previous draft had these issues — return a corrected itinerary "
        "using the same tool. Keep as much of the plan as possible; only "
        "change what's needed to resolve the issues.\n\n"
        f"DRAFT:\n{json.dumps(state.draft_itinerary, indent=2)}\n\n"
        f"ISSUES:\n- " + "\n- ".join(state.conflict_notes)
    )
    raw = await _run_llm_planner(
        prompt,
        extra_messages=[
            {"role": "assistant", "content": json.dumps(state.draft_itinerary)},
            {"role": "user", "content": repair_msg},
        ],
    )
    return {
        "draft_itinerary": raw,
        "repair_attempts": state.repair_attempts + 1,
    }


# --------------------------------------------------------------------------- #
# Step 5 — Assemble the final Itinerary
# --------------------------------------------------------------------------- #


def assemble_itinerary_node(state: PlanningState) -> dict:
    if state.error_notes:
        # Load step failed — surface the error without an Itinerary.
        msg = "Missing agent outputs: " + "; ".join(state.error_notes)
        logger.warning("assemble_itinerary → aborting: %s", msg)
        return {"response_message": msg, "phase": "review"}

    raw = state.draft_itinerary or {}
    days = _hydrate_days(raw.get("days", []), _hotels_by_id(state))
    trip = state.trip_request
    itinerary = Itinerary(
        trip_id=str(uuid.uuid4()),
        user_id=(state.user_profile.user_id if state.user_profile else "anonymous"),
        title=raw.get("title") or f"{trip.num_days}-day trip to {trip.destination}",
        days=days,
        total_cost=_compute_total_cost(days, state.budget),
        budget_breakdown=state.budget,
        notes=(raw.get("notes") or []) + state.conflict_notes,
        created_at=datetime.utcnow(),
        version=1,
    )
    summary = (
        f"Here's your {itinerary.days[0].date}–{itinerary.days[-1].date} "
        f"itinerary for {trip.destination}."
    )
    logger.info(
        "assemble_itinerary → v%d days=%d total=$%.0f",
        itinerary.version, len(itinerary.days), itinerary.total_cost,
    )
    return {
        "itinerary": itinerary,
        "response_message": summary,
        "phase": "review",
    }


def assemble_revision_node(state: PlanningState) -> dict:
    raw = state.draft_itinerary or {}
    days = _hydrate_days(raw.get("days", []), _hotels_by_id(state))
    current = state.itinerary
    new_itin = Itinerary(
        trip_id=current.trip_id,
        user_id=current.user_id,
        title=raw.get("title") or current.title,
        days=days,
        total_cost=_compute_total_cost(days, state.budget),
        budget_breakdown=state.budget,
        notes=(raw.get("notes") or []) + state.conflict_notes,
        created_at=datetime.utcnow(),
        version=current.version + 1,
    )
    summary = _diff_summary(current, new_itin)
    logger.info(
        "assemble_revision → v%d→v%d | %s",
        current.version, new_itin.version, summary,
    )
    return {
        "itinerary": new_itin,
        "changes_summary": summary,
        "response_message": summary,
        "phase": "review",
    }


# --------------------------------------------------------------------------- #
# Revision-graph prelude — refetch agents when the client didn't cache them
# --------------------------------------------------------------------------- #


async def ensure_agent_data_node(state: PlanningState) -> dict:
    """The revise endpoint accepts optional cached option lists. If any of the
    critical ones are missing we refetch, then merge the client-provided
    hotel_options back in (mirrors the previous revise() behaviour)."""
    needs_refetch = not (
        state.route_options and state.hotel_options and state.event_options
    )
    if not needs_refetch:
        return {}

    prefs = state.user_profile.preferences if state.user_profile else None
    outputs = await dispatch_agents(
        ["route", "hotel", "restaurant", "event"],
        state.trip_request,
        prefs,
    )

    # Merge client-cached hotels back in so a revision that references a
    # previously-shown hotel still resolves.
    fresh_hotels = outputs.get("hotel", [])
    seen = {h.hotel_id for h in fresh_hotels}
    for h in state.hotel_options:
        if h.hotel_id not in seen:
            fresh_hotels.append(h)

    logger.info(
        "ensure_agent_data → refetched (route=%d hotel=%d rest=%d event=%d)",
        len(outputs.get("route", [])), len(fresh_hotels),
        len(outputs.get("restaurant", [])), len(outputs.get("event", [])),
    )

    # We want to REPLACE the lists here, not append via reducer. LangGraph's
    # `operator.add` reducer would concatenate — so return the fresh lists
    # already merged with the cached ones. Because the incoming state.*_options
    # values are also in the reduction, this would double-count. Instead, we
    # emit the DIFF: only what's not already there.
    def diff(new: list, existing: list, key: str) -> list:
        seen_ids = {getattr(x, key) for x in existing}
        return [x for x in new if getattr(x, key) not in seen_ids]

    return {
        "route_options": diff(outputs.get("route", []), state.route_options, "route_id"),
        "hotel_options": diff(fresh_hotels, state.hotel_options, "hotel_id"),
        "restaurant_options": diff(outputs.get("restaurant", []), state.restaurant_options, "restaurant_id"),
        "event_options": diff(outputs.get("event", []), state.event_options, "event_id"),
    }


# --------------------------------------------------------------------------- #
# Conditional-edge helpers
# --------------------------------------------------------------------------- #


def load_data_router(state: PlanningState) -> str:
    """After load: continue to budget/planner if ok, else jump to assemble."""
    return "assemble" if state.error_notes else "continue"


def conflict_router(state: PlanningState) -> str:
    """After check_conflicts: repair if we still have issues and haven't burned
    our attempt yet; otherwise assemble."""
    if state.conflict_notes and state.repair_attempts < MAX_REPAIR_ATTEMPTS:
        return "repair"
    return "assemble"
