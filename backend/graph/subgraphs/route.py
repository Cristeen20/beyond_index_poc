"""Route sub-agent as a compiled subgraph + outer-graph wrapper.

See hotel.py for why the wrapper exists (parallel subgraph fanout would
otherwise cause concurrent writes on shared PlanningState channels).
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agent_models import PlanningState
from sub_agents import run_route_agent

logger = logging.getLogger("graph.subgraphs.route")


def _prefs(state: PlanningState):
    return state.user_profile.preferences if state.user_profile else None


async def fetch_route_node(state: PlanningState) -> dict:
    options = await run_route_agent(state.trip_request, _prefs(state))
    logger.info("route_sub.fetch → %d option(s)", len(options))
    return {"route_options": options}


def build_route_subgraph():
    g = StateGraph(PlanningState)
    g.add_node("fetch_route", fetch_route_node)
    g.add_edge(START, "fetch_route")
    g.add_edge("fetch_route", END)
    return g.compile()


_COMPILED = build_route_subgraph()


async def route_sub_node(state: PlanningState) -> dict:
    result = await _COMPILED.ainvoke(state)
    return {"route_options": result.get("route_options", [])}
