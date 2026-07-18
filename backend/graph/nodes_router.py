"""Router nodes for the top-level travel graph.

Implements the Intake Router section of the diagram in itenary_agent.md §1:
`classify_intent` (LLM classifier), the conversational-answer branch, and
the trip-hydration / slot-gate sequence that precedes agent dispatch.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import openai

from agent_models import IntentClassification, PlanningState, TripRequest
from intake_router import build_slot_question, classify, slot_gate

logger = logging.getLogger("graph.router")


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


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


async def classify_intent(state: PlanningState) -> dict:
    """LLM classifier — decides conversational / direct / full."""
    intent: IntentClassification = await classify(
        state.incoming_message, state.history, state.user_profile
    )
    logger.info(
        "classify_intent → route=%s agents=%s confidence=%.2f",
        intent.route, intent.target_agents, intent.confidence,
    )
    return {"intent": intent, "phase": "routing"}


async def answer_conversational(state: PlanningState) -> dict:
    """LLM-only reply — no agents dispatched."""
    messages: list[dict] = [{"role": "system", "content": _CONVERSATIONAL_SYSTEM}]
    for m in state.history or []:
        role = m.get("role") if isinstance(m, dict) else m.role
        content = m.get("content") if isinstance(m, dict) else m.content
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": state.incoming_message})

    resp = await _get_llm().chat.completions.create(
        model="gpt-4o",
        max_tokens=800,
        messages=messages,
    )
    text = resp.choices[0].message.content or ""
    logger.info("answer_conversational → %d chars", len(text))
    return {"response_message": text, "phase": "direct_answer"}


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
    explicit one straight through)."""
    trip = _hydrate_trip_request(state.intent, state.trip_request)
    return {"trip_request": trip} if trip else {}


def check_slot_gate(state: PlanningState) -> dict:
    """Compute missing required slots for the target agents. On FULL when the
    trip can't be hydrated at all, fall back to the (route, hotel) slot set
    to ask for the single most important missing entity."""
    intent = state.intent
    if intent is None:
        return {}

    if state.trip_request is None:
        # Router said FULL/DIRECT but we couldn't build a TripRequest.
        agents_for_gate = intent.target_agents or ["route", "hotel"]
        missing = slot_gate(agents_for_gate, intent.extracted_slots)
    elif intent.route == "direct":
        missing = intent.missing_required_slots or slot_gate(
            intent.target_agents, intent.extracted_slots
        )
    else:
        # FULL with a fully-hydrated trip — nothing to ask.
        missing = []

    if not missing:
        return {}

    question = build_slot_question(missing)
    logger.info("slot_gate missing=%s → question=%r", missing, question)
    return {"followup_question": question, "response_message": question}
