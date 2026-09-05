"""Pre-planning nodes (features/pre_planning.md).

Sits between `check_slot_gate` and the existing `itinerary_planning` subgraph
on the FULL route. Walks the user through:

  confirm_basics → elicit_scope → propose_places → propose_stays_for_selection
  → itinerary_planning

Each "propose_*" or "elicit_*" node writes a structured `options_payload`
to state and edges into `wait_for_next_message`. The next turn arrives with
either an `option_action` (structured, from a button click) or free text —
the paired `parse_*` node consumes it, updates state, and routes to the
next stage.

Only the FULL route uses this; DIRECT / CONVERSATIONAL / REVISE are
untouched.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import asyncio

from langfuse.openai import openai

from agent_models import PlanningState
from sub_agents import (
    run_event_agent,
    run_hotel_agent,
    run_restaurant_agent,
    run_route_agent,
)

logger = logging.getLogger("graph.preplanning")


_llm_client: openai.AsyncOpenAI | None = None


def _get_llm() -> openai.AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = openai.AsyncOpenAI()
    return _llm_client


def _prefs(state: PlanningState):
    return state.user_profile.preferences if state.user_profile else None


# --------------------------------------------------------------------------- #
# Payload helpers
# --------------------------------------------------------------------------- #


def _confirm_basics_payload(trip, show_travelers: bool = False) -> dict:
    # Origin is optional (FULL_OPTIONAL_SLOTS) — a local day out has no
    # journey, and _hydrate_trip_request leaves the placeholder "Unknown"
    # behind. Say "Trip to X" rather than showing the placeholder.
    lead = (
        f"From {trip.origin} to {trip.destination}"
        if trip.has_origin
        else f"Trip to {trip.destination}"
    )

    parts = [lead]
    # Dates only appear when the user actually provided them; otherwise we
    # defer to the ask_num_days step.
    if trip.start_date and trip.end_date:
        parts.append(
            f"{trip.num_days} day(s) ({trip.start_date} → {trip.end_date})"
        )
    # Same for travelers: only echo the count when we're going to skip the
    # ask (i.e. the user gave us a number). Otherwise showing "1
    # traveler(s)" looks like a decision we've made for them.
    if show_travelers:
        parts.append(f"{trip.travelers} traveler(s)")
    description = ", ".join(parts) + "."

    if trip.must_include:
        description += f" Including {', '.join(trip.must_include)}."
    return {
        "kind": "confirm_basics",
        "title": "Ready to plan?",
        "description": description,
        "items": [],
        "actions": [
            {"id": "confirm", "label": "Yes, let's plan"},
        ],
        "select": "none",
        "page": 0,
        "has_more": False,
    }


def _num_days_payload(current: int) -> dict:
    return {
        "kind": "num_days",
        "title": "How many days is this trip?",
        "description": (
            "Pick a number of days, or type a custom number."
        ),
        "items": [],
        "actions": [
            {"id": "1", "label": "1 day"},
            {"id": "2", "label": "2 days"},
            {"id": "3", "label": "3 days"},
            {"id": "4", "label": "4 days"},
            {"id": "5", "label": "5 days"},
            {"id": "7", "label": "1 week"},
        ],
        "select": "none",
        "page": 0,
        "has_more": False,
        "meta": {"current": current},
    }


def _num_travelers_payload(current: int) -> dict:
    return {
        "kind": "num_travelers",
        "title": "How many travelers?",
        "description": (
            "Pick the number of travelers, or type a custom count."
        ),
        "items": [],
        "actions": [
            {"id": "1", "label": "Solo"},
            {"id": "2", "label": "2 people"},
            {"id": "3", "label": "3 people"},
            {"id": "4", "label": "4 people"},
            {"id": "5", "label": "5 people"},
            {"id": "6", "label": "6+ people"},
        ],
        "select": "none",
        "page": 0,
        "has_more": False,
        "meta": {"current": current},
    }


def _scope_payload() -> dict:
    return {
        "kind": "scope",
        "title": "How would you like to plan this?",
        "description": (
            "Pick a scope. You can always ask for more later."
        ),
        "items": [],
        "actions": [
            {"id": "places_only",  "label": "Just show me places"},
            {"id": "day_by_day",   "label": "Full day-by-day itinerary"},
        ],
        "select": "none",
        "page": 0,
        "has_more": False,
    }


def _items_payload(
    kind: str,
    title: str,
    items: list[dict],
    page: int,
    has_more: bool,
    select: str = "multi",
    hint: str | None = None,
) -> dict:
    actions = []
    if has_more:
        actions.append({"id": "more", "label": "More options"})
    return {
        "kind": kind,
        "title": title,
        "description": hint or "",
        "items": items,
        "actions": actions,
        "select": select,
        "page": page,
        "has_more": has_more,
    }


# --------------------------------------------------------------------------- #
# Ranker prompts
# --------------------------------------------------------------------------- #


_PLACES_RANKER_SYSTEM = (
    "You are ranking candidate places for a traveler. For each input place, "
    "assign a rank (1 = highest priority) and a ONE-sentence rationale that "
    "explains WHY it's worth including — what kind of experience it offers "
    "and when in the trip it would fit best (early / mid / late). "
    "Prefer variety across types (landmark, museum, nature, food-cultural) "
    "and lower walking-tolerance-friendly picks first when possible. "
    "Return JSON: {\"ranked\": [{\"id\": str, \"rank\": int, \"rationale\": str}, ...]}. "
    "Include EVERY input id exactly once — no additions, no omissions."
)


_STAYS_RANKER_SYSTEM = (
    "You are ranking candidate hotels for a traveler who has already picked "
    "the places they want to visit. For each hotel, write a ONE-sentence "
    "rationale that emphasises TRAVEL-TIME impact — roughly how close it is "
    "to the selected places and how much daily commuting it saves versus a "
    "generic central hotel. Use approximate minutes if you can ("
    "'~10 min to X, ~15 min to Y'). Return JSON: "
    "{\"ranked\": [{\"id\": str, \"rank\": int, \"rationale\": str}, ...]}. "
    "Include EVERY input hotel id exactly once."
)


_FREETEXT_PARSER_SYSTEM = (
    "You translate the user's free-text reply into a structured action for "
    "a UI card that showed them options. The card kind, currently shown "
    "items, and available actions are provided. Decide ONE action:\n"
    "- 'select' + ids[] when the user picks specific items (by name, number, "
    "  or index like 'the first two')\n"
    "- 'more' when they want additional options\n"
    "- 'question' + text when they ask about the shown items\n"
    "- 'confirm' when they say yes / correct / go ahead (for confirm/scope/"
    "  day_by_day kinds)\n"
    "- 'correct' + text when they want to change the confirmed basics\n"
    "For scope cards return 'confirm' with ids=[scope_id_they_picked]. "
    "Return JSON: {\"action\": str, \"ids\": [str], \"text\": str}."
)


async def _rank_items_llm(
    system_prompt: str, context: str, items_json: list[dict]
) -> dict[str, tuple[int, str]]:
    """Ask the ranker to score items. Returns {id: (rank, rationale)}."""
    if not items_json:
        return {}
    try:
        resp = await _get_llm().chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=800,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context + "\n\nItems:\n" + json.dumps(items_json)},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except (json.JSONDecodeError, openai.OpenAIError) as exc:
        logger.warning("ranker failed: %s — falling back to input order", exc)
        return {it["id"]: (i + 1, "") for i, it in enumerate(items_json)}
    ranked: dict[str, tuple[int, str]] = {}
    for r in data.get("ranked", []):
        rid = r.get("id")
        if not isinstance(rid, str):
            continue
        rank = int(r.get("rank") or (len(ranked) + 1))
        rationale = str(r.get("rationale") or "").strip()
        ranked[rid] = (rank, rationale)
    # Fill in anything the ranker forgot so the UI always renders.
    for i, it in enumerate(items_json):
        ranked.setdefault(it["id"], (len(ranked) + i + 1, ""))
    return ranked


# --------------------------------------------------------------------------- #
# Stage 1 — confirm basics
# --------------------------------------------------------------------------- #


def confirm_basics_node(state: PlanningState) -> dict:
    """Echo the hydrated trip back with Yes / Change buttons.

    Skipped once `basics_confirmed=True` — the edge fn short-circuits so we
    don't ask twice within the same trip.
    """
    trip = state.trip_request
    payload = _confirm_basics_payload(trip, show_travelers=state.travelers_confirmed)
    logger.info("confirm_basics_node → asking to confirm %s→%s (%d days)",
                trip.origin, trip.destination, trip.num_days)
    return {
        "options_payload": payload,
        "pending_stage": "confirm_basics",
        "phase": "pre_planning",
        "response_message": payload["description"],
    }


async def parse_confirm_node(state: PlanningState) -> dict:
    """Parse the reply to the confirm_basics card.

    The card only offers a single "Yes, let's plan" button — parameters
    aren't editable at this stage. So any non-question reply is treated
    as confirmation; free-text questions keep the card up and get an
    LLM-answered response.
    """
    action = await _resolve_action(state)
    kind = action.get("action")

    if kind == "question":
        answer = await _answer_over_card(
            question=action.get("text") or state.incoming_message,
            payload=state.options_payload or {},
        )
        return {"response_message": answer, "option_action": None}

    logger.info("parse_confirm_node → basics confirmed")
    return {
        "basics_confirmed": True,
        "options_payload": None,
        "pending_stage": None,
        "option_action": None,
    }


# --------------------------------------------------------------------------- #
# Stage 1b — ask number of days
# --------------------------------------------------------------------------- #


def ask_num_days_node(state: PlanningState) -> dict:
    """Ask the user to confirm the trip length before pre-planning advances.

    Fires on every FULL trip (per design decision 1a). Even when the
    classifier extracted num_days or a date range, we still surface the
    ask so the user gets one canonical place to change the length. The
    default `num_days` from state is passed through in payload meta so
    the frontend can highlight it.
    """
    trip = state.trip_request
    current = max(getattr(trip, "num_days", 1) or 1, 1) if trip else 1
    payload = _num_days_payload(current)
    logger.info("ask_num_days_node → asking (current default=%d)", current)
    return {
        "options_payload": payload,
        "pending_stage": "num_days",
        "phase": "pre_planning",
        "response_message": payload["title"],
    }


def _extract_num_days(action: dict) -> int | None:
    """Pull a positive integer out of the action.

    Order: `ids[0]` (button click carries the numeric id), then free
    text. Returns None if nothing sensible parsed.
    """
    ids = action.get("ids") or []
    if ids:
        try:
            n = int(str(ids[0]).strip())
            if n >= 1:
                return n
        except (TypeError, ValueError):
            pass

    text = (action.get("text") or "").strip().lower()
    if not text:
        return None
    # Grab the first integer in the reply (handles "3", "3 days", "make it 5").
    import re
    m = re.search(r"\d+", text)
    if m:
        try:
            n = int(m.group(0))
            if n >= 1:
                return n
        except ValueError:
            pass
    words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "week": 7, "weekend": 2,
    }
    for w, n in words.items():
        if w in text:
            return n
    return None


async def parse_num_days_node(state: PlanningState) -> dict:
    """Parse the reply to the ask_num_days card.

    Accepts numeric button ids ("3") or free text ("3 days", "a week").
    Free-text questions about the trip keep the card up and get answered.
    Updates `trip_request.num_days` in-place; also derives an `end_date`
    when the trip already has a `start_date` so downstream stays/planner
    stay coherent.
    """
    action = await _resolve_action(state)
    kind = action.get("action")

    if kind == "question":
        answer = await _answer_over_card(
            question=action.get("text") or state.incoming_message,
            payload=state.options_payload or {},
        )
        return {"response_message": answer, "option_action": None}

    n = _extract_num_days(action)
    if n is None:
        # Unrecognised reply — keep the card up with a hint.
        logger.info("parse_num_days_node → could not parse %r", action)
        return {
            "response_message": (
                "I couldn't read a number of days from that. "
                "Tap one of the options or type a number like '3'."
            ),
            "option_action": None,
        }

    trip = state.trip_request
    if trip is None:
        logger.warning("parse_num_days_node → no trip_request; skipping update")
        return {
            "options_payload": None,
            "pending_stage": None,
            "option_action": None,
        }

    from datetime import timedelta
    updated = trip.model_copy(update={"num_days": n})
    if updated.start_date:
        updated = updated.model_copy(
            update={"end_date": updated.start_date + timedelta(days=n - 1)}
        )
    logger.info("parse_num_days_node → num_days=%d", n)
    return {
        "trip_request": updated,
        "num_days_confirmed": True,
        "options_payload": None,
        "pending_stage": None,
        "option_action": None,
    }


# --------------------------------------------------------------------------- #
# Stage 2b — ask number of travelers
# --------------------------------------------------------------------------- #


def ask_num_travelers_node(state: PlanningState) -> dict:
    """Ask the user to confirm the traveler count before scope selection.

    Sticky like ask_num_days: fires once per session, then travelers_confirmed
    keeps it out of the way. The classifier's default (1) is passed through
    in `meta.current` so the frontend can highlight it.
    """
    trip = state.trip_request
    current = max(getattr(trip, "travelers", 1) or 1, 1) if trip else 1
    payload = _num_travelers_payload(current)
    logger.info("ask_num_travelers_node → asking (current default=%d)", current)
    return {
        "options_payload": payload,
        "pending_stage": "num_travelers",
        "phase": "pre_planning",
        "response_message": payload["title"],
    }


async def parse_num_travelers_node(state: PlanningState) -> dict:
    """Parse the reply to the ask_num_travelers card.

    Reuses _extract_num_days — the extraction logic (button id first,
    then a positive int scraped from free text) is the same shape.
    """
    action = await _resolve_action(state)
    kind = action.get("action")

    if kind == "question":
        answer = await _answer_over_card(
            question=action.get("text") or state.incoming_message,
            payload=state.options_payload or {},
        )
        return {"response_message": answer, "option_action": None}

    n = _extract_num_days(action)
    if n is None:
        logger.info("parse_num_travelers_node → could not parse %r", action)
        return {
            "response_message": (
                "I couldn't read a number of travelers from that. "
                "Tap one of the options or type a number like '2'."
            ),
            "option_action": None,
        }

    trip = state.trip_request
    if trip is None:
        logger.warning("parse_num_travelers_node → no trip_request; skipping update")
        return {
            "options_payload": None,
            "pending_stage": None,
            "option_action": None,
        }

    updated = trip.model_copy(update={"travelers": n})
    logger.info("parse_num_travelers_node → travelers=%d", n)
    return {
        "trip_request": updated,
        "travelers_confirmed": True,
        "options_payload": None,
        "pending_stage": None,
        "option_action": None,
    }


# --------------------------------------------------------------------------- #
# Stage 2 — elicit scope
# --------------------------------------------------------------------------- #


def elicit_scope_node(state: PlanningState) -> dict:
    payload = _scope_payload()
    logger.info("elicit_scope_node → asking for scope")
    return {
        "options_payload": payload,
        "pending_stage": "scope",
        "phase": "pre_planning",
        "response_message": payload["title"],
    }


async def parse_scope_node(state: PlanningState) -> dict:
    """Parse the reply to the scope card.

    Buttons carry the scope id in `ids[0]` (places_only / day_by_day).
    Free-text asking about the scopes is answered without finalising —
    the card stays up. Only a clear scope pick advances.
    """
    action = await _resolve_action(state)
    kind = action.get("action")

    if kind == "question":
        answer = await _answer_over_card(
            question=action.get("text") or state.incoming_message,
            payload=state.options_payload or {},
        )
        return {"response_message": answer, "option_action": None}

    # Structured pick → ids[0] is authoritative. Free text falls back to
    # keyword sniff so typing "day by day" still works.
    scope_id = ""
    ids = action.get("ids") or []
    if ids:
        scope_id = ids[0].lower().strip()
    elif kind == "confirm":
        scope_id = (action.get("text") or "").lower().strip()

    if scope_id in ("places_only", "day_by_day"):
        scope = scope_id
    elif "day" in scope_id or "itinerary" in scope_id or "full" in scope_id:
        scope = "day_by_day"
    elif "place" in scope_id or "just" in scope_id or "only" in scope_id:
        scope = "places_only"
    else:
        # Unrecognised — keep the card up and ask the LLM to help.
        logger.info("parse_scope_node → couldn't parse %r, keeping card up", scope_id)
        answer = await _answer_over_card(
            question=action.get("text") or state.incoming_message,
            payload=state.options_payload or {},
        )
        return {"response_message": answer, "option_action": None}

    logger.info("parse_scope_node → scope=%s (raw=%r)", scope, scope_id)
    return {
        "planning_scope": scope,
        "options_payload": None,
        "pending_stage": None,
        "option_action": None,
        "options_page": 0,
    }


# --------------------------------------------------------------------------- #
# Stage 3 — propose places (with pagination)
# --------------------------------------------------------------------------- #


def _num_places_per_page(trip) -> int:
    # Roughly 3 per day, floored at 3 and capped at 15.
    return max(3, min(3 * max(trip.num_days, 1), 15))


def _matches_must_include(name: str, must_include: list[str]) -> bool:
    """Loose name match — Google rarely echoes the user's wording exactly
    ("moonlight beach" → "Moonlight Beach Park"), so match either direction."""
    n = (name or "").strip().lower()
    if not n:
        return False
    return any(
        n in m.strip().lower() or m.strip().lower() in n
        for m in must_include if m.strip()
    )


async def propose_places_node(state: PlanningState) -> dict:
    trip = state.trip_request
    page = state.options_page or 0
    per_page = _num_places_per_page(trip)

    # Page 0 fetches fresh; subsequent pages slice from the cached list so
    # "more" is deterministic and cheap.
    if page == 0 or not state.event_options:
        candidates = await run_event_agent(trip, _prefs(state), limit=48)
    else:
        candidates = state.event_options

    # Hide places the user already named in their query — they're
    # auto-included in parse_places_reply. Full candidate list stays in
    # state.event_options so downstream (fill_missing_agents, itinerary)
    # can still resolve them by id.
    must_include = trip.must_include or []
    visible = [c for c in candidates
               if not _matches_must_include(c.name, must_include)]

    start = page * per_page
    slice_ = visible[start:start + per_page]
    if not slice_:
        logger.info("propose_places_node → page %d empty, offering current picks", page)
        payload = _items_payload(
            kind="places",
            title=f"Places to visit in {trip.destination}",
            items=[],
            page=page,
            has_more=False,
            hint="That's all I could find. Pick from earlier options or ask something else.",
        )
        return {
            "options_payload": payload,
            "pending_stage": "places",
            "phase": "pre_planning",
        }

    items_json = [
        {"id": e.event_id, "name": e.name, "type": e.type,
         "cost": e.cost, "address": e.address}
        for e in slice_
    ]
    context = (
        f"Trip: {trip.num_days}-day visit to {trip.destination}, "
        f"{trip.travelers} traveler(s), style="
        f"{_prefs(state).travel_style if _prefs(state) else 'balanced'}."
    )
    ranks = await _rank_items_llm(_PLACES_RANKER_SYSTEM, context, items_json)

    items = []
    for e in slice_:
        rank, rationale = ranks.get(e.event_id, (999, ""))
        items.append({
            "id": e.event_id,
            "name": e.name,
            "rank": rank,
            "rationale": rationale,
            "meta": {
                "type": e.type,
                "address": e.address,
                "cost": e.cost,
                "lat": e.latitude,
                "lng": e.longitude,
            },
        })
    items.sort(key=lambda it: it["rank"])

    has_more = (start + per_page) < len(visible)
    payload = _items_payload(
        kind="places",
        title=f"Places to visit in {trip.destination}"
              + (f" (page {page + 1})" if page > 0 else ""),
        items=items,
        page=page,
        has_more=has_more,
        select="multi",
        hint="Pick a few to include. You can ask for more or ask a question about any of them.",
    )
    updates: dict = {
        "options_payload": payload,
        "pending_stage": "places",
        "phase": "pre_planning",
        "response_message": payload["title"],
    }
    if page == 0:
        updates["event_options"] = candidates
        updates["selected_place_ids"] = []
    return updates


async def parse_places_reply_node(state: PlanningState) -> dict:
    """Handle a reply to the places card.

    Actions:
      - select  → record ids, clear payload/stage, advance to stays/end
      - more    → clear payload, bump page, re-enter propose_places
      - question → answer over cached items, keep payload up
    """
    action = await _resolve_action(state)
    kind = action["action"]

    if kind == "more":
        logger.info("parse_places_reply_node → more (page %d → %d)",
                    state.options_page, state.options_page + 1)
        return {
            "options_page": (state.options_page or 0) + 1,
            "options_payload": None,   # signal to edge: re-propose
            "option_action": None,
        }

    if kind == "question":
        answer = await _answer_over_items(
            question=action.get("text") or state.incoming_message,
            items=state.options_payload.get("items", []) if state.options_payload else [],
            title=(state.options_payload or {}).get("title", "your options"),
        )
        logger.info("parse_places_reply_node → question answered (%d chars)", len(answer))
        return {
            "response_message": answer,
            "option_action": None,
            # options_payload + pending_stage unchanged — user keeps picking.
        }

    ids = action.get("ids") or []
    # Places named in the user's query were hidden from the picker;
    # fold them back into the selection so they land in the plan.
    must_include = (state.trip_request.must_include or []) if state.trip_request else []
    auto_ids = [
        e.event_id for e in state.event_options
        if _matches_must_include(e.name, must_include)
    ]
    merged_ids = list(dict.fromkeys(ids + auto_ids))
    logger.info(
        "parse_places_reply_node → selected %d place(s) (+ %d auto-included)",
        len(ids), len(auto_ids),
    )
    updates: dict = {
        "selected_place_ids": merged_ids,
        "options_payload": None,
        "pending_stage": None,
        "option_action": None,
        "options_page": 0,  # reset for the next stage's paging
    }

    # For places_only this IS the last node of the turn — route_after_parse_places
    # goes straight to wait_for_next_message, so nothing downstream writes a
    # response. wait_for_next_message blanks response_message on every resume,
    # so without this the turn ends empty and the API projection falls back to
    # "Sorry, I couldn't put a plan together." Other scopes overwrite this with
    # their own card title a node later.
    if state.planning_scope == "places_only":
        picked = [
            e.name for e in state.event_options
            if e.event_id in set(merged_ids)
        ]
        updates["response_message"] = (
            f"Saved your {len(picked)} pick(s): {', '.join(picked)}. "
            "Ask me anything about them, or say 'plan the days' for a "
            "full itinerary."
            if picked else
            "I didn't catch which places you wanted — tell me the names "
            "and I'll save them."
        )

    return updates


# --------------------------------------------------------------------------- #
# Stage 4 — propose stays for selection
# --------------------------------------------------------------------------- #


def _centroid(coords: list[tuple[float, float]]) -> tuple[float, float] | None:
    if not coords:
        return None
    lat = sum(c[0] for c in coords) / len(coords)
    lng = sum(c[1] for c in coords) / len(coords)
    return (lat, lng)


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


async def propose_stays_for_selection_node(state: PlanningState) -> dict:
    trip = state.trip_request
    page = state.options_page or 0
    per_page = 5

    if page == 0 or not state.hotel_options:
        candidates = await run_hotel_agent(trip, _prefs(state), limit=20)
    else:
        candidates = state.hotel_options

    selected_places = [e for e in state.event_options
                       if e.event_id in set(state.selected_place_ids)]
    centroid = _centroid([(e.latitude, e.longitude) for e in selected_places])
    if centroid:
        candidates = sorted(
            candidates,
            key=lambda h: _haversine_km(centroid, (h.latitude, h.longitude)),
        )

    start = page * per_page
    slice_ = candidates[start:start + per_page]
    if not slice_:
        payload = _items_payload(
            kind="stays",
            title="Stays",
            items=[],
            page=page,
            has_more=False,
            hint="No more stays to show. Pick one from earlier or continue.",
        )
        return {
            "options_payload": payload,
            "pending_stage": "stays",
            "phase": "pre_planning",
        }

    items_json = [
        {"id": h.hotel_id, "name": h.name, "address": h.address,
         "price_per_night": h.price_per_night, "star_rating": h.star_rating,
         "distance_km_to_selected": round(
             min(
                 (_haversine_km((h.latitude, h.longitude), (e.latitude, e.longitude))
                  for e in selected_places),
                 default=0.0,
             ),
             2,
         ),
         "selected_places": [
             {"name": e.name,
              "km": round(_haversine_km(
                  (h.latitude, h.longitude), (e.latitude, e.longitude)), 2)}
             for e in selected_places[:5]
         ]}
        for h in slice_
    ]
    context = (
        f"Selected places for this trip: "
        f"{', '.join(e.name for e in selected_places) or '(none — pick any)'}. "
        f"Trip: {trip.num_days} day(s) in {trip.destination}."
    )
    ranks = await _rank_items_llm(_STAYS_RANKER_SYSTEM, context, items_json)

    items = []
    for h in slice_:
        rank, rationale = ranks.get(h.hotel_id, (999, ""))
        items.append({
            "id": h.hotel_id,
            "name": h.name,
            "rank": rank,
            "rationale": rationale,
            "meta": {
                "address": h.address,
                "price_per_night": h.price_per_night,
                "star_rating": h.star_rating,
                "lat": h.latitude,
                "lng": h.longitude,
            },
        })
    items.sort(key=lambda it: it["rank"])

    # Trips longer than 2 days may split nights across multiple hotels;
    # shorter trips still lock to a single stay.
    allow_multi = (trip.num_days or 1) > 2
    has_more = (start + per_page) < len(candidates)
    payload = _items_payload(
        kind="stays",
        title="Stays near your picks"
              + (f" (page {page + 1})" if page > 0 else ""),
        items=items,
        page=page,
        has_more=has_more,
        select="multi" if allow_multi else "single",
        hint=(
            "Pick one or more stays, ask a question, or see more."
            if allow_multi
            else "Pick a stay, ask a question, or see more."
        ),
    )
    updates: dict = {
        "options_payload": payload,
        "pending_stage": "stays",
        "phase": "pre_planning",
        "response_message": payload["title"],
    }
    if page == 0:
        updates["hotel_options"] = candidates
        updates["selected_stay_ids"] = []
    return updates


async def parse_stays_reply_node(state: PlanningState) -> dict:
    action = await _resolve_action(state)
    kind = action["action"]

    if kind == "more":
        return {
            "options_page": (state.options_page or 0) + 1,
            "options_payload": None,   # signal to edge: re-propose
            "option_action": None,
        }

    if kind == "question":
        answer = await _answer_over_items(
            question=action.get("text") or state.incoming_message,
            items=state.options_payload.get("items", []) if state.options_payload else [],
            title=(state.options_payload or {}).get("title", "these stays"),
        )
        return {
            "response_message": answer,
            "option_action": None,
        }

    ids = action.get("ids") or []
    logger.info("parse_stays_reply_node → selected %d stay(s)", len(ids))
    return {
        "selected_stay_ids": ids,
        "options_payload": None,
        "pending_stage": None,
        "option_action": None,
        "options_page": 0,
    }


# --------------------------------------------------------------------------- #
# Bridge — fill missing agents before entering itinerary_planning
# --------------------------------------------------------------------------- #


async def fill_missing_agents_node(state: PlanningState) -> dict:
    """Prep state for itinerary_planning:
      1. Fetch route + restaurant options (pre-planning didn't need them).
      2. Filter event_options and hotel_options down to the user's picks
         so the LLM planner builds days AROUND the selections instead of
         substituting from the full ranked list.
    """
    trip = state.trip_request
    prefs = _prefs(state)

    tasks: dict[str, Any] = {}
    if not state.route_options:
        tasks["route"] = run_route_agent(trip, prefs)
    if not state.restaurant_options:
        tasks["restaurant"] = run_restaurant_agent(trip, prefs)

    updates: dict = {}
    if tasks:
        logger.info("fill_missing_agents_node → fetching %s", list(tasks))
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for key, res in zip(tasks.keys(), results):
            if isinstance(res, Exception):
                logger.warning("fill_missing_agents_node → %s failed: %s", key, res)
                continue
            if key == "route":
                updates["route_options"] = res
            elif key == "restaurant":
                updates["restaurant_options"] = res
    else:
        logger.info("fill_missing_agents_node → nothing to fetch")

    # Pin the planner to the user's picks. Full lists stay on the
    # checkpointer for pagination if they navigate back; only the version
    # itinerary_planning sees is scoped down.
    if state.selected_place_ids:
        picked_events = [
            e for e in state.event_options
            if e.event_id in set(state.selected_place_ids)
        ]
        if picked_events:
            logger.info("fill_missing_agents_node → scoping events to %d pick(s)",
                        len(picked_events))
            updates["event_options"] = picked_events

    if state.selected_stay_ids:
        picked_stays = [
            h for h in state.hotel_options
            if h.hotel_id in set(state.selected_stay_ids)
        ]
        if picked_stays:
            logger.info("fill_missing_agents_node → scoping stays to %d pick(s)",
                        len(picked_stays))
            updates["hotel_options"] = picked_stays

    return updates


# --------------------------------------------------------------------------- #
# Helpers — action resolution + free-text answering
# --------------------------------------------------------------------------- #


async def _resolve_action(state: PlanningState) -> dict[str, Any]:
    """Return a normalised action dict {action, ids, text}.

    Prefers the structured `option_action` when the frontend sent one
    (button/select). Falls back to a small LLM parser over the pending
    payload for free-text replies.
    """
    if state.option_action:
        act = state.option_action
        return {
            "action": act.get("action") or "select",
            "ids": act.get("ids") or [],
            "text": act.get("text"),
        }

    payload = state.options_payload or {}
    text = (state.incoming_message or "").strip()
    if not text:
        return {"action": "confirm", "ids": [], "text": None}

    context = (
        f"Card kind: {payload.get('kind')}\n"
        f"Title: {payload.get('title')}\n"
        f"Currently shown items (id → name):\n"
        + "\n".join(f"- {it.get('id')} → {it.get('name')}"
                    for it in payload.get("items", []))
        + "\nAvailable actions:\n"
        + "\n".join(f"- {a.get('id')} ({a.get('label')})"
                    for a in payload.get("actions", []))
        + f"\n\nUser reply: {text!r}"
    )
    try:
        resp = await _get_llm().chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _FREETEXT_PARSER_SYSTEM},
                {"role": "user", "content": context},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except (json.JSONDecodeError, openai.OpenAIError) as exc:
        logger.warning("free-text parser failed: %s — defaulting to question", exc)
        data = {"action": "question", "ids": [], "text": text}

    return {
        "action": data.get("action") or "question",
        "ids": [str(i) for i in (data.get("ids") or []) if i],
        "text": data.get("text") or text,
    }


_ANSWER_OVER_ITEMS_SYSTEM = (
    "You are answering a user's question about the options currently on "
    "their screen. Use ONLY the JSON items provided. Answer in one or two "
    "short sentences. Cite items by name. If the info they want isn't in "
    "the items, say so and offer what IS available."
)


_ANSWER_OVER_CARD_SYSTEM = (
    "You are helping a user who is on a decision card in a trip-planning "
    "flow. The card kind, title, and available button choices are given. "
    "The user asked a question — answer it in ONE short paragraph and end "
    "with a nudge to pick one of the button options. Don't invent live "
    "data (prices, availability) — answer from general travel knowledge."
)


async def _answer_over_card(question: str, payload: dict) -> str:
    """Answer a free-text question asked while a button-only card is up.

    Keeps the card visible; the user still has to click a button to
    advance. Used by parse_confirm / parse_scope.
    """
    if not question:
        return "Pick one of the options above to continue."
    ctx = (
        f"Card kind: {payload.get('kind')}\n"
        f"Card title: {payload.get('title')}\n"
        f"Card description: {payload.get('description')}\n"
        f"Button choices: "
        + ", ".join(
            f"{a.get('id')} ({a.get('label')})"
            for a in payload.get("actions", [])
        )
        + f"\n\nUser's question: {question!r}"
    )
    try:
        resp = await _get_llm().chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[
                {"role": "system", "content": _ANSWER_OVER_CARD_SYSTEM},
                {"role": "user", "content": ctx},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except openai.OpenAIError as exc:
        logger.warning("answer_over_card failed: %s", exc)
        return "Pick one of the buttons above to continue."


async def _answer_over_items(question: str, items: list[dict], title: str) -> str:
    if not question:
        return "What would you like to know about these?"
    ctx = (
        f"Card title: {title}\n"
        f"Items JSON:\n{json.dumps(items, indent=2, default=str)}\n\n"
        f"User's question: {question!r}"
    )
    try:
        resp = await _get_llm().chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[
                {"role": "system", "content": _ANSWER_OVER_ITEMS_SYSTEM},
                {"role": "user", "content": ctx},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except openai.OpenAIError as exc:
        logger.warning("answer_over_items failed: %s", exc)
        return "I couldn't check that just now — try picking from the list and we'll keep going."


# --------------------------------------------------------------------------- #
# Edge functions
# --------------------------------------------------------------------------- #


def route_after_wait(state: PlanningState) -> str:
    """After wait_for_next_message resumes.

    - If a pre-planning payload was pending, route to the matching parser.
    - Otherwise, run the normal intent_decision flow.
    """
    stage = state.pending_stage
    if stage == "confirm_basics":
        return "parse_confirm"
    if stage == "num_days":
        return "parse_num_days"
    if stage == "num_travelers":
        return "parse_num_travelers"
    if stage == "scope":
        return "parse_scope"
    if stage == "places":
        return "parse_places_reply"
    if stage == "stays":
        return "parse_stays_reply"
    return "intent_decision"


def route_after_parse_confirm(state: PlanningState) -> str:
    # parse_confirm always sets basics_confirmed=True (there's no
    # "Change" button anymore). Sticky flags let us skip each ask once
    # the user has answered it in this session.
    if not state.num_days_confirmed:
        return "ask_num_days"
    if not state.travelers_confirmed:
        return "ask_num_travelers"
    return "elicit_scope"


def route_after_parse_num_days(state: PlanningState) -> str:
    """After the num_days card:
    - options_payload still set → we answered a question; keep the card up
    - otherwise → advance to ask_num_travelers or elicit_scope (sticky).
    """
    if state.options_payload is not None:
        return "wait_for_next_message"
    if not state.travelers_confirmed:
        return "ask_num_travelers"
    return "elicit_scope"


def route_after_parse_num_travelers(state: PlanningState) -> str:
    """After the num_travelers card:
    - options_payload still set → question was answered, keep card up
    - otherwise → advance to elicit_scope
    """
    if state.options_payload is not None:
        return "wait_for_next_message"
    return "elicit_scope"


def route_after_parse_scope(state: PlanningState) -> str:
    return "propose_places"


def route_after_parse_places(state: PlanningState) -> str:
    """After a places reply:
      - payload still set  → question was answered; wait for next reply
      - payload=None + pending_stage=places → user hit 'more' → re-propose
      - payload=None + pending_stage=None   → selection recorded → advance
    """
    if state.options_payload is not None:
        return "wait_for_next_message"
    if state.pending_stage == "places":
        return "propose_places"
    if state.planning_scope == "places_only":
        return "wait_for_next_message"
    return "propose_stays_for_selection"


def route_after_parse_stays(state: PlanningState) -> str:
    if state.options_payload is not None:
        return "wait_for_next_message"
    if state.pending_stage == "stays":
        return "propose_stays_for_selection"
    return "fill_missing_agents"
