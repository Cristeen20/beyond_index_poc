"""Dispatch fan-in + direct-path renderers.

The per-agent fetch logic lives in `graph/subgraphs/`. The travel graph
routes to one subgraph (DIRECT list/answer) or all four (FULL). This
module holds the nodes that sit downstream of the subgraphs:

- `post_dispatch`       — fan-in join; records which agents produced output.
- `merge_direct`        — DIRECT list mode: flatten top-5 into
  `direct_result`, plus a non-escalating upsell offer.
- `answer_from_places`  — DIRECT answer mode: LLM (gpt-4o-mini) synthesizes
  a one-sentence factual answer from the top hit (hours, address, etc.).
"""

from __future__ import annotations

import json
import logging
from typing import Any

import openai

from agent_models import PlanningState

logger = logging.getLogger("graph.dispatch")


_llm_client: openai.AsyncOpenAI | None = None


def _get_llm() -> openai.AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        _llm_client = openai.AsyncOpenAI()
    return _llm_client


def post_dispatch(state: PlanningState) -> dict:
    """Fan-in for the parallel subgraphs. Records which agents returned
    data so downstream conditional edges can inspect it."""
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
    """Assemble the DIRECT list-mode response. Not an Itinerary — a
    lightweight list of raw options plus a non-escalating upgrade offer."""
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


# --------------------------------------------------------------------------- #
# answer_from_places — DIRECT answer-mode renderer
# --------------------------------------------------------------------------- #


_ANSWER_SYSTEM = (
    "You are a travel assistant answering a specific factual question about "
    "a named place. You are given the top hit from a live Google Places "
    "search — use ONLY that data plus the user's message to answer.\n\n"
    "Rules:\n"
    "- Answer in ONE short natural sentence, no preamble, no bullets.\n"
    "- Cite the place by name in the answer.\n"
    "- If the specific fact the user asked about is NOT in the data (e.g. "
    "  they asked for a phone number but no phone was returned), say so "
    "  briefly and offer what IS available (hours, address, rating).\n"
    "- If multiple hits are provided and the top one clearly isn't the "
    "  place the user meant, say the match looks off and ask which one.\n"
    "- Do not invent data. Do not summarise data the user didn't ask about."
)


_TARGET_TO_OPTIONS = {
    "hotel": "hotel_options",
    "restaurant": "restaurant_options",
    "event": "event_options",
    "route": "route_options",
}


def _top_hits(state: PlanningState) -> list[dict[str, Any]]:
    """Return the top few results from the option list matching THIS turn's
    target agent. Using intent.target_agents[0] (not iterating all lists)
    matters because the checkpointer persists prior turns' options — an
    older restaurant search would otherwise leak into a fresh hotel
    answer-mode query."""
    if not state.intent or not state.intent.target_agents:
        return []
    target = state.intent.target_agents[0]
    field = _TARGET_TO_OPTIONS.get(target)
    if field is None:
        return []
    options = getattr(state, field, None) or []
    return [opt.model_dump(mode="json") for opt in options[:3]]


async def answer_from_places(state: PlanningState) -> dict:
    """LLM node — synthesize a one-sentence factual answer from the top
    place-search hit. Runs when the classifier set answer_mode='answer'."""
    hits = _top_hits(state)
    if not hits:
        msg = "I couldn't find that place — could you double-check the name?"
        logger.info("answer_from_places → no hits, returning fallback")
        return {
            "response_message": msg,
            "direct_result": [],
            "phase": "direct_answer",
        }

    context = (
        f"User's question: {state.incoming_message!r}\n\n"
        f"Top place-search hit(s):\n{json.dumps(hits, indent=2, default=str)}"
    )
    resp = await _get_llm().chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=160,
        messages=[
            {"role": "system", "content": _ANSWER_SYSTEM},
            {"role": "user", "content": context},
        ],
    )
    answer = (resp.choices[0].message.content or "").strip()
    logger.info("answer_from_places → %r", answer[:120])
    return {
        "response_message": answer,
        # Keep the raw hit(s) on direct_result so the frontend can render
        # a card alongside the answer if it wants to; the formatter picks
        # `message` first when it's non-empty.
        "direct_result": hits,
        "phase": "direct_answer",
    }
