"""Revision StateGraph — v2 topology (see itinerary_langgraph_flow.md §3).

  START ──► _fanout_refetch(state)
              │  (for each *_options list that arrived empty on the
              │   ReviseRequest, add the matching subgraph to the fanout;
              │   if all four are cached, jump straight to allocate_budget)
              │
      ┌───────┴───────┬────────────┬─────────────┐
      ▼               ▼            ▼             ▼
  hotel_sub  restaurant_sub    route_sub     event_sub
      │               │            │             │
      └───────────────┴──────┬─────┴─────────────┘
                             ▼
                       allocate_budget
                             │
                             ▼
                 run_llm_planner_revision
                             │
                             ▼
                       check_conflicts ◄──┐
                        clean │ conflicts │
                              ▼           │
                      assemble_revision   │
                              │           │
                              ▼           │
                            (END)         │
                                          │
                       repair_planner ────┘  (bounded, MAX_REPAIR_ATTEMPTS=1)
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from agent_models import PlanningState
from graph.nodes_itinerary import (
    allocate_budget_node,
    assemble_revision_node,
    check_conflicts_node,
    conflict_router,
    repair_planner_node,
    run_llm_planner_revision_node,
)
from graph.subgraphs import (
    event_sub_node,
    hotel_sub_node,
    restaurant_sub_node,
    route_sub_node,
)

logger = logging.getLogger("graph.revise")


def _fanout_refetch(state: PlanningState):
    """Edge fn — for each *_options list that arrived empty, fan out to
    the matching subgraph. If all four are already cached, skip refetch
    entirely and jump straight to allocate_budget."""
    to_run: list[str] = []
    if not state.hotel_options:
        to_run.append("hotel_sub")
    if not state.restaurant_options:
        to_run.append("restaurant_sub")
    if not state.route_options:
        to_run.append("route_sub")
    if not state.event_options:
        to_run.append("event_sub")

    if not to_run:
        logger.info("_fanout_refetch → all cached, skipping to allocate_budget")
        return "allocate_budget"

    logger.info("_fanout_refetch → refetching %s", to_run)
    return to_run


def build_revise_graph():
    """Compile and return the revision StateGraph."""
    g = StateGraph(PlanningState)

    # Per-agent subgraph wrappers (see graph/subgraphs/__init__.py for
    # why we use the wrappers here rather than the raw compiled subgraphs).
    g.add_node("hotel_sub", hotel_sub_node)
    g.add_node("restaurant_sub", restaurant_sub_node)
    g.add_node("route_sub", route_sub_node)
    g.add_node("event_sub", event_sub_node)

    g.add_node("allocate_budget", allocate_budget_node)
    g.add_node("run_llm_planner_revision", run_llm_planner_revision_node)
    g.add_node("check_conflicts", check_conflicts_node)
    g.add_node("repair_planner", repair_planner_node)
    g.add_node("assemble_revision", assemble_revision_node)

    # START fans out to any subgraph whose list is empty, or skips ahead
    g.add_conditional_edges(
        START,
        _fanout_refetch,
        {
            "hotel_sub": "hotel_sub",
            "restaurant_sub": "restaurant_sub",
            "route_sub": "route_sub",
            "event_sub": "event_sub",
            "allocate_budget": "allocate_budget",
        },
    )

    # Fan-in: every refetch subgraph edges into allocate_budget
    for name in ("hotel_sub", "restaurant_sub", "route_sub", "event_sub"):
        g.add_edge(name, "allocate_budget")

    g.add_edge("allocate_budget", "run_llm_planner_revision")
    g.add_edge("run_llm_planner_revision", "check_conflicts")
    g.add_conditional_edges(
        "check_conflicts",
        conflict_router,
        {"repair": "repair_planner", "assemble": "assemble_revision"},
    )
    g.add_edge("repair_planner", "check_conflicts")
    g.add_edge("assemble_revision", END)

    return g.compile()
