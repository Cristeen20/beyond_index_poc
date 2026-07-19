"""Router nodes for the top-level travel graph.

Implements the Intake Router section of the diagram in itenary_agent.md §1:
`intent_decision` (LLM classifier), the conversational-answer branch, and
the trip-hydration / slot-gate sequence that precedes agent dispatch.

The confidence gate lives in `route_after_intent` (an edge function, not a
node body) so that routing logic stays on the graph edges where LangGraph
can inspect it.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import openai

from agent_models import IntentClassification, PlanningState, TripRequest
from intake_router import CONFIDENCE_TAU, classify, slot_gate

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

    # Answer mode (place-lookup) has different requirements from booking
    # searches: we just need a place_name and a location for the search
    # context — dates/origin are irrelevant to "what time does X open?".
    if intent.route == "direct" and intent.answer_mode == "answer":
        missing: list[str] = []
        if not intent.extracted_slots.get("place_name"):
            missing.append("place_name")
        if state.trip_request is None and not intent.extracted_slots.get("destination"):
            missing.append("destination")
    elif state.trip_request is None:
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
