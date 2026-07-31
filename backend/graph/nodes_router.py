"""Router nodes for the top-level travel graph.

Implements the Intake Router section of the diagram in itenary_agent.md §1:
`intent_decision` (LLM classifier), the conversational-answer branch, and
the trip-hydration / slot-gate sequence that precedes agent dispatch.

The confidence gate lives in `route_after_intent` (an edge function, not a
node body) so that routing logic stays on the graph edges where LangGraph
can inspect it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, timedelta

import openai

from agent_models import IntentClassification, PlanningState, TripRequest
from intake_router import CONFIDENCE_TAU, classify, slot_gate
from places import resolve_venue

logger = logging.getLogger("graph.router")


_CONVERSATIONAL_SYSTEM = (
    "You are a travel agent. Your job is to help users decide where to visit, "
    "answer questions about destinations and venues, and help them build "
    "itineraries. Stay in that role — if a message is off-topic, redirect "
    "politely back to travel planning.\n\n"
    "REPLY LENGTH: at most 2 sentences. Be direct and information-dense. No "
    "throat-clearing, no filler, no sign-offs like 'hope this helps'. If the "
    "user asks a question, answer it — don't restate it.\n\n"
    "SPECIFIC NAMED VENUES: if the user names a specific real-world venue "
    "(hotel, restaurant, cafe, shop, attraction, etc.), you will usually be "
    "given a GROUND TRUTH block below listing what Google Places says about "
    "each. Use ONLY that ground truth for facts about the venue (its category, "
    "address, rating, hours). NEVER infer the category from words in the name — "
    "tokens like 'Inn', 'Cafe', 'Palace', 'Lodge', 'House', 'Club', 'Garden', "
    "'Villa', 'Resort' inside a name mean NOTHING about what the venue actually "
    "is. If GROUND TRUTH says a venue was not found, say so in one sentence and "
    "ask for a clarifying detail (city, neighbourhood, alternate spelling). "
    "If no GROUND TRUTH block is present, no specific venue was detected — "
    "answer generically."
)


_VENUE_EXTRACT_SYSTEM = (
    "You extract Google Places search queries from a user's chat message.\n\n"
    "If the message names one or more specific real-world venues (a proper-noun "
    "hotel, restaurant, cafe, shop, attraction, etc.), return a search query "
    "for each — combining the venue name with any location context in the "
    "message (city, neighbourhood, region). If no specific venue is named "
    "(e.g. only generic requests like 'best restaurants in Paris'), return an "
    "empty list.\n\n"
    "Return JSON of the form {\"queries\": [\"Venue Name, City\", ...]}.\n\n"
    "Examples:\n"
    "- 'best time to visit Omars Inn Kannur' → {\"queries\": [\"Omars Inn Kannur\"]}\n"
    "- 'compare Taj Mahal Palace Mumbai with ITC Grand Chola Chennai' → "
    "{\"queries\": [\"Taj Mahal Palace Mumbai\", \"ITC Grand Chola Chennai\"]}\n"
    "- 'best cafes in Paris' → {\"queries\": []}\n"
    "- 'hi how are you' → {\"queries\": []}"
)


_llm_client: openai.AsyncOpenAI | None = None


def _get_llm() -> openai.AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = openai.AsyncOpenAI()
    return _llm_client


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


async def intent_decision(state: PlanningState) -> dict:
    """LLM classifier — decides conversational / direct / full / revise.

    Runs at the top of every turn (fresh call or checkpointer resume).
    Session-persisted `trip_request` + `itinerary` are fed to the
    classifier so it can (a) distinguish follow-ups from topic resets
    and (b) know when REVISE is a valid choice for this turn.

    If we paused on a slot ask last turn, `state.intent` and
    `state.followup_question` still hold the pending intent + the exact
    question asked. Those are passed as PENDING TURN CONTEXT so the
    classifier can MERGE the user's answer into the pending intent
    instead of reclassifying a fragmentary follow-up ("in Kyoto") from
    scratch.
    """
    intent: IntentClassification = await classify(
        state.incoming_message,
        state.history,
        state.user_profile,
        trip_request=state.trip_request,
        itinerary=state.itinerary,
        prior_intent=state.intent,
        pending_question=state.followup_question,
    )
    logger.info(
        "intent_decision → route=%s agents=%s confidence=%.2f slots=%s",
        intent.route, intent.target_agents, intent.confidence,
        intent.extracted_slots,
    )
    return {"intent": intent, "phase": "routing"}


def route_after_intent(state: PlanningState) -> str:
    """Edge fn — applies the confidence gate + revise gating.

    Direct and full both go to the single `planning` lane
    (hydrate_trip → check_slot_gate → dispatch); the fanout shape is
    decided later by `_slot_gate_route` based on `intent.route`.
    """
    intent = state.intent
    if intent is None:
        return "conversational"
    if intent.confidence < CONFIDENCE_TAU:
        logger.info(
            "route_after_intent → confidence %.2f < %.2f, collapsing to conversational",
            intent.confidence, CONFIDENCE_TAU,
        )
        return "conversational"
    if intent.route == "conversational":
        return "conversational"
    if intent.route == "revise" and state.itinerary is not None:
        return "revise"
    return "planning"















async def answer_conversational(state: PlanningState) -> dict:
    """LLM reply grounded by Google Places for any named venues in the message."""
    ground_truth = await _lookup_named_venues(state.incoming_message)
    system_prompt = _CONVERSATIONAL_SYSTEM
    if ground_truth:
        system_prompt += "\n\nGROUND TRUTH — venues resolved via Google Places:\n" + ground_truth

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for m in state.history or []:
        role = m.get("role") if isinstance(m, dict) else m.role
        content = m.get("content") if isinstance(m, dict) else m.content
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": state.incoming_message})

    resp = await _get_llm().chat.completions.create(
        model="gpt-4o",
        max_tokens=200,
        messages=messages,
    )
    text = resp.choices[0].message.content or ""
    logger.info("answer_conversational → %d chars (venues=%d)",
                len(text), ground_truth.count("\n") + (1 if ground_truth else 0))
    return {"response_message": text, "phase": "direct_answer"}


async def _extract_venue_queries(message: str) -> list[str]:
    """Ask a mini LLM to pull out proper-noun venues (with location context)."""
    try:
        resp = await _get_llm().chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _VENUE_EXTRACT_SYSTEM},
                {"role": "user", "content": message},
            ],
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except (json.JSONDecodeError, openai.OpenAIError) as exc:
        logger.warning("venue extractor failed: %s", exc)
        return []
    raw = data.get("queries", [])
    return [q.strip() for q in raw if isinstance(q, str) and q.strip()][:3]


async def _lookup_named_venues(message: str) -> str:
    """Extract venues from the message and resolve each via Google Places.

    Returns a rendered ground-truth block ready to append to the system
    prompt, or an empty string if nothing was detected / resolved.
    """
    queries = await _extract_venue_queries(message)
    if not queries:
        return ""
    results = await asyncio.gather(
        *(resolve_venue(q) for q in queries), return_exceptions=True
    )
    lines: list[str] = []
    for query, r in zip(queries, results):
        if isinstance(r, Exception):
            logger.warning("resolve_venue(%r) errored: %s", query, r)
            lines.append(f"- '{query}': lookup failed.")
            continue
        if r is None:
            lines.append(f"- '{query}': not found on Google Places.")
            continue
        types = ", ".join(r.get("types", [])) or "unknown category"
        parts = [f"'{r['name']}' — types: [{types}] — {r['address']}"]
        if r.get("rating") is not None:
            parts.append(f"rating {r['rating']}")
        if r.get("price_level"):
            parts.append(f"price {r['price_level']}")
        if r.get("open_now") is not None:
            parts.append("open now" if r["open_now"] else "closed now")
        if r.get("weekday_hours"):
            parts.append("hours: " + " | ".join(r["weekday_hours"]))
        if r.get("phone"):
            parts.append(f"phone {r['phone']}")
        if r.get("website"):
            parts.append(f"website {r['website']}")
        if r.get("summary"):
            parts.append(r["summary"])
        lines.append("- " + " — ".join(parts))
    return "\n".join(lines)


def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _hydrate_trip_request(
    intent: IntentClassification,
    explicit: TripRequest | None,
) -> TripRequest | None:
    """Prefer explicitly-provided TripRequest; otherwise build from slots.

    Exception: if the classifier extracted a destination that differs from
    the persisted trip's, treat this as a new trip and rebuild from slots
    so a stale trip_request can't shadow the user's new intent.
    """
    slots = intent.extracted_slots
    slot_dest = (slots.get("destination") or "").strip().lower()
    if explicit is not None:
        explicit_dest = (explicit.destination or "").strip().lower()
        if slot_dest and slot_dest != explicit_dest:
            explicit = None  # fall through to build fresh from slots
        else:
            return explicit

    destination = slots.get("destination")
    if not destination:
        return None

    origin = slots.get("origin", "Unknown")
    start = _parse_date(slots.get("start_date") or slots.get("dates"))
    end = _parse_date(slots.get("end_date"))
    if start is None:
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


def hydrate_trip(state: PlanningState) -> dict:
    """Build a TripRequest from the classifier's extracted slots (or pass an
    explicit one straight through).

    When we detect a new-trip switch (destination changed vs. the persisted
    trip), wipe stale per-agent options, itinerary, and scratchpad so the
    downstream sub-graphs rebuild from a clean slate instead of reusing the
    prior turn's data.
    """
    trip = _hydrate_trip_request(state.intent, state.trip_request)
    if trip is None:
        return {}
    updates: dict = {"trip_request": trip}
    prior = state.trip_request
    if prior is not None and (
        (trip.destination or "").strip().lower()
        != (prior.destination or "").strip().lower()
    ):
        logger.info(
            "hydrate_trip: destination changed %r → %r, clearing stale state",
            prior.destination, trip.destination,
        )
        updates.update({
            "itinerary": None,
            "route_options": [],
            "hotel_options": [],
            "restaurant_options": [],
            "event_options": [],
            "direct_result": None,
            "chosen_route": None,
            "budget": None,
            "draft_itinerary": None,
            "conflict_notes": [],
            "changes_summary": "",
            "agent_outputs_received": {},
            "error_notes": [],
            "repair_attempts": 0,
            "missing_slots": [],
            "followup_question": None,
            "revision_feedback": None,
        })
    return updates


def check_slot_gate(state: PlanningState) -> dict:
    """Compute missing required slots for the target agents.

    Deterministic only — writes `missing_slots`. The follow-up question
    itself is written by `ask_missing_slots` (LLM node) so we don't ship
    hard-coded strings.

    Also clears `followup_question` on the dispatch path (missing == []).
    Pending state was preserved through `wait_for_next_message` so
    `intent_decision` could see it; once the merged intent no longer has
    missing slots we're about to dispatch, so we wipe the stale question
    before the response projection reads state. (Option B — wipe on
    dispatch — from itinerary_langgraph_flow.md.)
    """
    intent = state.intent
    if intent is None:
        return {"missing_slots": [], "followup_question": None}

    if state.trip_request is None:
        agents_for_gate = intent.target_agents or ["route", "hotel"]
        missing = slot_gate(agents_for_gate, intent.extracted_slots)
    elif intent.route == "direct":
        missing = intent.missing_required_slots or slot_gate(
            intent.target_agents, intent.extracted_slots
        )
    else:
        # FULL with a fully-hydrated trip — nothing to ask.
        missing = []

    logger.info("check_slot_gate → missing=%s", missing)
    if missing:
        # Preserve prior followup_question during the pending pause; the
        # forthcoming ask_missing_slots node will overwrite it fresh.
        return {"missing_slots": missing}
    # Dispatch path — clear the pending question so it doesn't leak
    # into the API response projection.
    return {"missing_slots": [], "followup_question": None}


# --------------------------------------------------------------------------- #
# ask_missing_slots — LLM node that phrases the follow-up question
# --------------------------------------------------------------------------- #


_ASK_SYSTEM = (
    "You are a travel assistant collecting the minimum information needed to "
    "run a search. The user has been talking to you already; you know some "
    "details and are missing a few. Ask ONE short, natural question that "
    "collects the missing pieces. Follow these rules:\n\n"
    "- Batch related slots into one sentence when natural (dates = start + "
    "  end date; ask together, don't split).\n"
    "- Reference details you already know so the ask feels contextual — e.g. "
    "  'for your Kannur trip, what dates?' beats 'What dates are you looking "
    "  at?'.\n"
    "- Match the user's tone: informal if they typed casually, otherwise "
    "  neutral. Never over-apologize, never repeat the question stem.\n"
    "- Do NOT ask about slots that are already present in the extracted "
    "  slots list.\n"
    "- Keep it to one sentence, no preamble, no closing pleasantries.\n"
    "- Do not offer choices, do not enumerate — just ask."
)


async def ask_missing_slots(state: PlanningState) -> dict:
    """Generate a context-aware follow-up question for the missing slots.

    Runs only when `check_slot_gate` wrote a non-empty `missing_slots`.
    Uses gpt-4o-mini — a single short sentence per turn doesn't need the
    full model.
    """
    intent = state.intent
    known = intent.extracted_slots if intent else {}
    target_agents = intent.target_agents if intent else []
    route = intent.route if intent else "unknown"

    context_lines = [
        f"Router decision: route={route}, target_agents={target_agents}",
        f"Slots already known: {known or '(none)'}",
        f"Slots still missing: {state.missing_slots}",
    ]

    messages: list[dict] = [
        {"role": "system", "content": _ASK_SYSTEM},
        {"role": "system", "content": "\n".join(context_lines)},
    ]
    # Keep recent history so the ask sounds continuous, not out of the blue.
    for m in (state.history or [])[-4:]:
        role = m.get("role") if isinstance(m, dict) else m.role
        content = m.get("content") if isinstance(m, dict) else m.content
        messages.append({"role": role, "content": content})
    if state.incoming_message:
        messages.append({"role": "user", "content": state.incoming_message})

    resp = await _get_llm().chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=80,
        messages=messages,
    )
    question = (resp.choices[0].message.content or "").strip()
    logger.info("ask_missing_slots → %r", question)
    return {"followup_question": question, "response_message": question}
