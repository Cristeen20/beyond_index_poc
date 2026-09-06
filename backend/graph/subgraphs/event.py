"""Event / activity sub-agent as a compiled subgraph + outer-graph wrapper.

See hotel.py for why the wrapper exists (parallel subgraph fanout would
otherwise cause concurrent writes on shared PlanningState channels).
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agent_models import PlanningState
from sub_agents import run_event_agent

logger = logging.getLogger("graph.subgraphs.event")


def _prefs(state: PlanningState):
    return state.user_profile.preferences if state.user_profile else None


def _place_name(state: PlanningState) -> str | None:
    if state.intent and state.intent.answer_mode == "answer":
        return state.intent.extracted_slots.get("place_name")
    return None


async def fetch_event_node(state: PlanningState) -> dict:
    options = await run_event_agent(
        state.trip_request, _prefs(state), place_name=_place_name(state)
    )
    logger.info("event_sub.fetch → %d option(s)", len(options))
    return {"event_options": options}


def build_event_subgraph():
    g = StateGraph(PlanningState)
    g.add_node("fetch_event", fetch_event_node)
    g.add_edge(START, "fetch_event")
    g.add_edge("fetch_event", END)
    return g.compile()


_COMPILED = build_event_subgraph()


async def event_sub_node(state: PlanningState) -> dict:
    result = await _COMPILED.ainvoke(state)
    return {"event_options": result.get("event_options", [])}
