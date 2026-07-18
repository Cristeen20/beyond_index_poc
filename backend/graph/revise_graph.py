"""Revision StateGraph — implements §4 Step 6 (Revision Loop).

  ensure_agent_data   (refetch missing option lists; merge cached hotels)
        │
        ▼
  allocate_budget
        │
        ▼
  run_llm_planner_revision   (LLM edits the prior itinerary in place)
        │
        ▼
  check_conflicts
        ├── clean ──► assemble_revision ──► END
        └── conflicts + repair_attempts==0
              └► repair_planner ──► check_conflicts   (bounded to one pass)
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent_models import PlanningState
from graph.nodes_itinerary import (
    allocate_budget_node,
    assemble_revision_node,
    check_conflicts_node,
    conflict_router,
    ensure_agent_data_node,
    repair_planner_node,
    run_llm_planner_revision_node,
)


def build_revise_graph():
    """Compile and return the revision StateGraph."""
    g = StateGraph(PlanningState)

    g.add_node("ensure_agent_data", ensure_agent_data_node)
    g.add_node("allocate_budget", allocate_budget_node)
    g.add_node("run_llm_planner_revision", run_llm_planner_revision_node)
    g.add_node("check_conflicts", check_conflicts_node)
    g.add_node("repair_planner", repair_planner_node)
    g.add_node("assemble_revision", assemble_revision_node)

    g.add_edge(START, "ensure_agent_data")
    g.add_edge("ensure_agent_data", "allocate_budget")
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
