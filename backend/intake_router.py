"""Intake Router (§1.5).

An LLM classifier that decides whether an incoming turn takes the DIRECT
(agent-skipping) or FULL (multi-step planning) path, and — on the direct
path — which subset of agents to dispatch.

Also exports a `slot_gate()` helper that the direct flow uses to compute
missing required slots per §3.4.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable

import openai

from agent_models import (
    REQUIRED_SLOTS,
    IntentClassification,
    Itinerary,
    TripRequest,
    UserProfile,
)

logger = logging.getLogger("intake_router")


# Confidence threshold below which we default to FULL (§1.5 routing table).
CONFIDENCE_TAU = 0.55


_ROUTER_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_intent",
        "description": (
            "Classify the user's request as CONVERSATIONAL (LLM answers from "
            "knowledge, no APIs), DIRECT (targeted Google-backed lookup), "
            "FULL (day-by-day itinerary), or REVISE (edit the existing "
            "itinerary already on state). Extract any slots you can."
        ),
        "parameters": {
            "type": "object",
            "required": ["route", "target_agents", "extracted_slots", "confidence"],
            "properties": {
                "route": {
                    "type": "string",
                    "enum": ["conversational", "direct", "full", "revise"],
                    "description": (
                        "'conversational' for general questions the LLM can answer "
                        "from its own knowledge — travel tips, weather, packing, visa, "
                        "cultural/etiquette info, currency, greetings, follow-ups that "
                        "don't need live data. NO agents dispatched.\n"
                        "'direct' for a narrow query targeting specific agent capabilities "
                        "(hotels, restaurants, routes, events) with enough entities to act.\n"
                        "'full' for a trip/plan request, vague query, or day-by-day schedule.\n"
                        "'revise' ONLY when the SESSION CONTEXT shows an existing itinerary "
                        "AND the user is asking to change/edit/replace parts of it "
                        "(e.g. 'swap the hotel', 'make day 2 cheaper', 'add a museum')."
                    ),
                },
                "target_agents": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["route", "hotel", "restaurant", "event"]},
                    "description": "Agents needed to answer. For 'full' route, use all four.",
                },
                "extracted_slots": {
                    "type": "object",
                    "description": (
                        "Slot values you were able to extract. Common keys: "
                        "origin, destination, dates, start_date, end_date, travelers, "
                        "budget, currency, hotel_rating, cuisine."
                    ),
                    "additionalProperties": {"type": "string"},
                },
                "confidence": {
                    "type": "number",
                    "description": "0.0–1.0 confidence in this classification.",
                },
                "rationale": {
                    "type": "string",
                    "description": "One-sentence justification for the routing decision.",
                },
                "answer_mode": {
                    "type": "string",
                    "enum": ["list", "answer"],
                    "description": (
                        "ONLY meaningful when route='direct'. "
                        "'list' when the user wants recommendations / a list of "
                        "options (e.g. 'best biryani in Kannur', 'hotels in Kyoto', "
                        "'top attractions in Osaka'). "
                        "'answer' when the user is asking a specific FACT about a "
                        "named place — hours, phone, address, 'is it open now', "
                        "'how far is X from Y'. The response will be a single natural "
                        "sentence synthesized from the top place-search hit, not a list. "
                        "Default 'list'. Ignored for conversational / full / revise routes."
                    ),
                },
            },
        },
    },
}


_ROUTER_SYSTEM = (
    "You are the Intake Router for a travel planning system. On every turn you "
    "decide which of FOUR paths handles the request:\n\n"
    "1) CONVERSATIONAL — the LLM answers from its own knowledge; no Google APIs "
    "   or agents are called. Use this for:\n"
    "   - general travel tips, best time to visit, cultural etiquette, safety\n"
    "   - visa / entry requirements, currency, tipping norms\n"
    "   - weather patterns, packing advice, language basics\n"
    "   - greetings, chit-chat, meta-questions ('what can you do?')\n"
    "   - follow-up clarifications the LLM can answer without fresh data\n"
    "   Examples: 'best time to visit Japan', 'do I need a visa for Kenya', "
    "   'what's the tipping etiquette in Paris', 'hi'.\n\n"
    "2) DIRECT — a narrow query that needs live Google-backed data for a subset "
    "   of these capabilities:\n"
    "   - hotel: find/recommend hotels\n"
    "   - restaurant: find/recommend restaurants or cafes\n"
    "   - route: find flights/trains/buses between cities\n"
    "   - event: find things to do, attractions, museums, festivals\n"
    "   Examples: 'find 4-star hotels in Kyoto next weekend', 'best ramen in "
    "   Tokyo', 'trains from Rome to Florence on Friday'.\n\n"
    "3) FULL — a full day-by-day itinerary planned by the Itinerary Agent. "
    "   Use when the request spans the whole trip, is vague ('plan a trip'), "
    "   or explicitly asks for a day-by-day schedule.\n\n"
    "4) REVISE — edit an existing itinerary. ONLY valid when the SESSION "
    "   CONTEXT below shows an itinerary is already on state AND the user is "
    "   asking to change, swap, add, remove, or adjust part of it.\n"
    "   Examples (with an itinerary already on state): 'make day 2 cheaper', "
    "   'swap the hotel for something closer to the station', 'add a museum "
    "   on day 3', 'shorten the trip by a day'.\n\n"
    "Rules of thumb:\n"
    "- If the question is answerable from general knowledge and does NOT "
    "  require finding specific places/times/prices, choose CONVERSATIONAL.\n"
    "- If the user names or implies a NEW place they want us to find, choose "
    "  DIRECT or FULL — even if a prior itinerary exists, a genuinely new "
    "  topic (different destination, greeting, unrelated question) is NOT a "
    "  revise.\n"
    "- REVISE requires BOTH an existing itinerary AND edit intent. When in "
    "  doubt between REVISE and DIRECT/FULL, prefer the fresh path.\n"
    "- SESSION CONTEXT is provided as a system message; if it says "
    "  'no itinerary yet', you MUST NOT return revise.\n\n"
    "Extract every slot you can from the message + history. If a prior "
    "trip_request is in the SESSION CONTEXT, prefer its destination/dates "
    "unless the user is explicitly asking about a new place. If unsure, "
    "lower your confidence; anything below ~0.55 for DIRECT is safely "
    "escalated to FULL. CONVERSATIONAL is never escalated.\n\n"
    "For DIRECT queries you must ALSO set `answer_mode`:\n"
    "  - 'list'   — user wants recommendations / options (default).\n"
    "               'best biryani in Kannur', 'hotels in Kyoto',\n"
    "               'restaurants near me', 'top museums in Rome'.\n"
    "  - 'answer' — user is asking a specific FACT about a named place.\n"
    "               'what time does Omar's Inn open', 'phone number for\n"
    "               the Ritz-Carlton Kyoto', 'is X open now', 'address\n"
    "               of Blue Bottle Coffee'. These get a one-sentence\n"
    "               natural answer synthesized from the top hit, not a\n"
    "               list. If the user names a specific place AND asks a\n"
    "               fact-shaped question, prefer 'answer'.\n\n"
    "When answer_mode='answer' YOU MUST extract the specific place name "
    "into `extracted_slots.place_name` (e.g. \"Omar's Inn\", "
    "\"Ritz-Carlton Kyoto\"). The subgraph uses `place_name` as the "
    "actual Google search query, so it must be the exact name the user "
    "typed. Do NOT emit `place_name` for list-mode queries."
)


_client: openai.AsyncOpenAI | None = None


def _get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI()
    return _client


def _memory_slots(user: UserProfile | None) -> dict[str, str]:
    """Hydrate slots already known from User Memory."""
    if user is None:
        return {}
    slots: dict[str, str] = {}
    prefs = user.preferences
    if prefs.preferred_hotel_rating is not None:
        slots["hotel_rating"] = str(prefs.preferred_hotel_rating)
    if prefs.preferred_foods:
        slots["cuisine"] = ", ".join(prefs.preferred_foods)
    return slots


def _session_context_message(
    trip: TripRequest | None,
    itinerary: Itinerary | None,
) -> str:
    """Compact per-turn context the classifier uses to disambiguate follow-ups
    from topic resets and to know when REVISE is even a valid choice."""
    lines: list[str] = ["SESSION CONTEXT (persisted across turns):"]
    if trip is not None:
        lines.append(
            f"- current trip_request: destination={trip.destination!r} "
            f"origin={trip.origin!r} dates={trip.start_date}→{trip.end_date} "
            f"travelers={trip.travelers}"
        )
    else:
        lines.append("- current trip_request: none")
    if itinerary is not None:
        lines.append(
            f"- current itinerary: v{itinerary.version} '{itinerary.title}' "
            f"({len(itinerary.days)} days, ~${itinerary.total_cost:.0f}) — "
            f"REVISE is a valid route this turn."
        )
    else:
        lines.append("- current itinerary: none — REVISE is NOT valid this turn.")
    return "\n".join(lines)


def _pending_turn_context_message(
    prior_intent: IntentClassification | None,
    pending_question: str | None,
) -> str | None:
    """Emit a PENDING TURN CONTEXT block iff last turn asked for missing
    slots. The classifier uses it to decide whether the new message is a
    slot answer (→ MERGE) or a topic reset (→ RECLASSIFY FRESH)."""
    if not prior_intent or not pending_question:
        return None
    if not prior_intent.missing_required_slots:
        return None

    known = prior_intent.extracted_slots or {}
    lines = [
        "PENDING TURN CONTEXT (last turn paused waiting on a slot answer):",
        f"- Question we just asked: {pending_question!r}",
        f"- We were collecting for: route={prior_intent.route}, "
        f"target_agents={prior_intent.target_agents}",
        f"- Slots already known: {known or '(none)'}",
        f"- Slots still missing: {prior_intent.missing_required_slots}",
        "",
        "Decide:",
        "(a) MERGE — if the user's message answers the pending question "
        "(fully or partially): KEEP the same `route` and `target_agents`; "
        "set `extracted_slots` to the UNION of the known slots and any new "
        "values from this turn's message; set confidence ≥ 0.85.",
        "(b) RECLASSIFY FRESH — if the message is a clear topic reset "
        "(greeting like 'hi', unrelated question, a NEW destination the "
        "pending intent wasn't about, or a request that needs a different "
        "route). Classify normally; ignore the pending state.",
        "Prefer (a) when the message is short and looks like an answer.",
    ]
    return "\n".join(lines)


async def classify(
    message: str,
    history: Iterable[dict] | None = None,
    user_profile: UserProfile | None = None,
    trip_request: TripRequest | None = None,
    itinerary: Itinerary | None = None,
    prior_intent: IntentClassification | None = None,
    pending_question: str | None = None,
) -> IntentClassification:
    """Run the LLM classifier and return an IntentClassification.

    Applies the confidence gate: if confidence < τ or no target agents were
    named while route='direct', we defensively downgrade to FULL. If the
    classifier returns 'revise' but no itinerary is on state, we downgrade
    to 'full' (revise is a lie without an itinerary to revise).

    When `prior_intent` has still-missing slots and `pending_question` is
    set, the classifier gets a PENDING TURN CONTEXT block and decides
    merge-vs-reclassify (option B in itinerary_langgraph_flow.md).
    """
    hydrated = _memory_slots(user_profile)

    messages: list[dict] = [{"role": "system", "content": _ROUTER_SYSTEM}]
    messages.append({
        "role": "system",
        "content": _session_context_message(trip_request, itinerary),
    })
    pending = _pending_turn_context_message(prior_intent, pending_question)
    if pending:
        messages.append({"role": "system", "content": pending})
    if hydrated:
        messages.append({
            "role": "system",
            "content": f"Slots hydrated from User Memory: {json.dumps(hydrated)}",
        })
    for m in (history or []):
        # accept both dict-form and pydantic-form history items
        role = m.get("role") if isinstance(m, dict) else m.role
        content = m.get("content") if isinstance(m, dict) else m.content
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": message})

    resp = await _get_client().chat.completions.create(
        model="gpt-4o",
        max_tokens=512,
        tools=[_ROUTER_TOOL],
        tool_choice={"type": "function", "function": {"name": "classify_intent"}},
        messages=messages,
    )

    tc = resp.choices[0].message.tool_calls[0]
    args = json.loads(tc.function.arguments)

    # Coerce all slot values to str — the LLM sometimes returns numeric
    # values (e.g. hotel_rating=5) despite the schema declaring `string`.
    raw_extracted = args.get("extracted_slots") or {}
    extracted = {**hydrated, **{k: str(v) for k, v in raw_extracted.items()}}
    raw_route = args["route"]
    raw_agents = args.get("target_agents") or []
    confidence = float(args.get("confidence") or 0.0)

    logger.info(
        "classifier raw: route=%s agents=%s confidence=%.2f slots=%s rationale=%s",
        raw_route, raw_agents, confidence, extracted, args.get("rationale"),
    )

    # Confidence gate — over-serve rather than mis-route.
    #  - conversational: never downgraded; if wrong, worst case is a chatty reply
    #  - direct: needs target_agents + confidence ≥ τ, else escalate to FULL
    #  - full: default fan-out to all four agents
    #  - revise: only valid when an itinerary is on state, else downgrade to FULL
    route = raw_route
    target_agents = raw_agents
    downgraded = False
    if route == "revise" and itinerary is None:
        route = "full"
        target_agents = ["route", "hotel", "restaurant", "event"]
        downgraded = True

    if route == "direct" and (confidence < CONFIDENCE_TAU or not target_agents):
        route = "full"
        target_agents = ["route", "hotel", "restaurant", "event"]
        downgraded = True

    if route == "full" and not target_agents:
        target_agents = ["route", "hotel", "restaurant", "event"]

    if route in ("conversational", "revise"):
        target_agents = []  # neither path dispatches sub-agents at the top level

    missing = slot_gate(target_agents, extracted) if route == "direct" else []

    if downgraded:
        logger.info(
            "confidence gate downgraded direct→full (tau=%.2f, confidence=%.2f)",
            CONFIDENCE_TAU, confidence,
        )
    raw_answer_mode = args.get("answer_mode") or "list"
    answer_mode = raw_answer_mode if raw_answer_mode in ("list", "answer") else "list"
    # answer_mode is meaningless outside DIRECT — clamp so downstream can't
    # accidentally branch on it for other routes.
    if route != "direct":
        answer_mode = "list"

    logger.info(
        "classified: route=%s agents=%s missing_slots=%s answer_mode=%s",
        route, target_agents, missing, answer_mode,
    )

    return IntentClassification(
        route=route,
        target_agents=target_agents,
        extracted_slots=extracted,
        missing_required_slots=missing,
        confidence=confidence,
        rationale=args.get("rationale"),
        answer_mode=answer_mode,
    )


def slot_gate(target_agents: list[str], extracted_slots: dict[str, str]) -> list[str]:
    """Return the required slots that are missing for the union of target agents.

    A slot is considered satisfied if it (or an equivalent) is present in
    extracted_slots. `dates` is satisfied if any of {dates, start_date, end_date}
    is present.
    """
    required: set[str] = set()
    for agent in target_agents:
        required |= REQUIRED_SLOTS.get(agent, set())

    def has(slot: str) -> bool:
        if slot == "dates":
            return any(k in extracted_slots for k in ("dates", "start_date", "end_date"))
        return slot in extracted_slots

    return [s for s in sorted(required) if not has(s)]


