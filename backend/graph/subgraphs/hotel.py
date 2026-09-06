"""Hotel sub-agent as a compiled subgraph + a thin outer-graph wrapper.

The wrapper exists because embedding a compiled subgraph directly as an
outer-graph node causes LangGraph to merge the subgraph's ENTIRE final
state back into the parent state. With four subgraphs fanning out in
parallel, every non-Annotated field (`user_profile`, `trip_request`,
etc.) gets written from four nodes in the same super-step → concurrent
write error. `hotel_sub_node` invokes the compiled subgraph and returns
only the delta this subgraph is responsible for.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agent_models import PlanningState
from sub_agents import run_hotel_agent

logger = logging.getLogger("graph.subgraphs.hotel")


def _prefs(state: PlanningState):
    return state.user_profile.preferences if state.user_profile else None


def _place_name(state: PlanningState) -> str | None:
    """Named-place lookup (answer mode) — subgraph uses this as the
    Google search query instead of the generic 'hotels' category."""
    if state.intent and state.intent.answer_mode == "answer":
        return state.intent.extracted_slots.get("place_name")
    return None


async def fetch_hotel_node(state: PlanningState) -> dict:
    options = await run_hotel_agent(
        state.trip_request, _prefs(state), place_name=_place_name(state)
    )
    logger.info("hotel_sub.fetch → %d option(s)", len(options))
    return {"hotel_options": options}


def build_hotel_subgraph():
    g = StateGraph(PlanningState)
    g.add_node("fetch_hotel", fetch_hotel_node)
    g.add_edge(START, "fetch_hotel")
    g.add_edge("fetch_hotel", END)
    return g.compile()


_COMPILED = build_hotel_subgraph()


async def hotel_sub_node(state: PlanningState) -> dict:
    """Outer-graph wrapper — invokes the compiled subgraph but returns
    only `hotel_options`, so parallel siblings don't collide on shared
    channels."""
    result = await _COMPILED.ainvoke(state)
    return {"hotel_options": result.get("hotel_options", [])}
