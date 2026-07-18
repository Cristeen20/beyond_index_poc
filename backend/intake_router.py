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

from agent_models import REQUIRED_SLOTS, IntentClassification, UserProfile

logger = logging.getLogger("intake_router")


# Confidence threshold below which we default to FULL (§1.5 routing table).
CONFIDENCE_TAU = 0.55


_ROUTER_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_intent",
        "description": (
            "Classify the user's request as CONVERSATIONAL (LLM answers from "
            "knowledge, no APIs), DIRECT (targeted Google-backed lookup), or "
            "FULL (day-by-day itinerary). Extract any slots you can."
        ),
        "parameters": {
            "type": "object",
            "required": ["route", "target_agents", "extracted_slots", "confidence"],
            "properties": {
                "route": {
                    "type": "string",
                    "enum": ["conversational", "direct", "full"],
                    "description": (
                        "'conversational' for general questions the LLM can answer "
                        "from its own knowledge — travel tips, weather, packing, visa, "
                        "cultural/etiquette info, currency, greetings, follow-ups that "
                        "don't need live data. NO agents dispatched.\n"
                        "'direct' for a narrow query targeting specific agent capabilities "
                        "(hotels, restaurants, routes, events) with enough entities to act.\n"
                        "'full' for a trip/plan request, vague query, or day-by-day schedule."
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
            },
        },
    },
}


_ROUTER_SYSTEM = (
    "You are the Intake Router for a travel planning system. On every turn you "
    "decide which of THREE paths handles the request:\n\n"
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
    "Rule of thumb: if the question is answerable from general knowledge and "
    "does NOT require finding specific places/times/prices, choose "
    "CONVERSATIONAL. If the user names or implies a place they want us to "
    "find, choose DIRECT or FULL.\n\n"
    "Extract every slot you can from the message + history. If unsure, lower "
    "your confidence; anything below ~0.55 for a DIRECT route will be safely "
    "escalated to FULL. CONVERSATIONAL is never escalated."
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


async def classify(
    message: str,
    history: Iterable[dict] | None = None,
    user_profile: UserProfile | None = None,
) -> IntentClassification:
    """Run the LLM classifier and return an IntentClassification.

    Applies the confidence gate: if confidence < τ or no target agents were
    named while route='direct', we defensively downgrade to FULL.
    """
    hydrated = _memory_slots(user_profile)

    messages: list[dict] = [{"role": "system", "content": _ROUTER_SYSTEM}]
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

    extracted = {**hydrated, **(args.get("extracted_slots") or {})}
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
    route = raw_route
    target_agents = raw_agents
    downgraded = False
    if route == "direct" and (confidence < CONFIDENCE_TAU or not target_agents):
        route = "full"
        target_agents = ["route", "hotel", "restaurant", "event"]
        downgraded = True

    if route == "full" and not target_agents:
        target_agents = ["route", "hotel", "restaurant", "event"]

    if route == "conversational":
        target_agents = []  # never dispatch on the conversational path

    missing = slot_gate(target_agents, extracted) if route == "direct" else []

    if downgraded:
        logger.info(
            "confidence gate downgraded direct→full (tau=%.2f, confidence=%.2f)",
            CONFIDENCE_TAU, confidence,
        )
    logger.info(
        "classified: route=%s agents=%s missing_slots=%s",
        route, target_agents, missing,
    )

    return IntentClassification(
        route=route,
        target_agents=target_agents,
        extracted_slots=extracted,
        missing_required_slots=missing,
        confidence=confidence,
        rationale=args.get("rationale"),
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


def build_slot_question(missing: list[str]) -> str:
    """One targeted question for the first missing slot (§4.5 step 2)."""
    if not missing:
        return ""
    slot = missing[0]
    prompts = {
        "origin": "Where are you departing from?",
        "destination": "Which destination did you have in mind?",
        "dates": "What dates are you looking at?",
    }
    return prompts.get(slot, f"Could you tell me the {slot.replace('_', ' ')}?")
