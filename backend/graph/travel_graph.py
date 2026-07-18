"""Top-level travel StateGraph — implements the dual-path flow diagrammed
in itenary_agent.md §1.

  classify_intent
    ├── conversational ──► answer_conversational ──► END
    └── planning (direct | full)
          └► hydrate_trip ──► check_slot_gate
                ├── ask ────────────────────────────► END
                └── go  ──► [hotel ∥ restaurant ∥ route ∥ event]
                              (self-gated by intent.target_agents)
                              └──► post_dispatch
                                     ├── direct ──► merge_direct ──► END
                                     └── full   ──► itinerary_planning
                                                        (compiled subgraph)
                                                                    └──► END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent_models import PlanningState
from graph.nodes_dispatch import (
    dispatch_event,
    dispatch_hotel,
    dispatch_restaurant,
    dispatch_route,
    merge_direct,
    post_dispatch,
)
from graph.nodes_itinerary import (
    allocate_budget_node,
    assemble_itinerary_node,
    check_conflicts_node,
    conflict_router,
    load_agent_data_node,
    load_data_router,
    repair_planner_node,
    run_llm_planner_node,
)
from graph.nodes_router import (
    answer_conversational,
    check_slot_gate,
    classify_intent,
    hydrate_trip,
)


# --------------------------------------------------------------------------- #
# Itinerary sub-graph — the §4 Planning Engine (Load → Budget → Schedule →
# Conflict → Repair? → Assemble). Compiled and embedded as a single node in
# the top-level graph.
# --------------------------------------------------------------------------- #


def _build_itinerary_subgraph():
    g = StateGraph(PlanningState)

    g.add_node("load_agent_data", load_agent_data_node)
    g.add_node("allocate_budget", allocate_budget_node)
    g.add_node("run_llm_planner", run_llm_planner_node)
    g.add_node("check_conflicts", check_conflicts_node)
    g.add_node("repair_planner", repair_planner_node)
    g.add_node("assemble_itinerary", assemble_itinerary_node)

    g.add_edge(START, "load_agent_data")
    g.add_conditional_edges(
        "load_agent_data",
        load_data_router,
        {"continue": "allocate_budget", "assemble": "assemble_itinerary"},
    )
    g.add_edge("allocate_budget", "run_llm_planner")
    g.add_edge("run_llm_planner", "check_conflicts")
    g.add_conditional_edges(
        "check_conflicts",
        conflict_router,
        {"repair": "repair_planner", "assemble": "assemble_itinerary"},
    )
    g.add_edge("repair_planner", "check_conflicts")
    g.add_edge("assemble_itinerary", END)

    return g.compile()


# --------------------------------------------------------------------------- #
# Top-level travel graph
# --------------------------------------------------------------------------- #


def _route_after_classify(state: PlanningState) -> str:
    intent = state.intent
    if intent is None or intent.route == "conversational":
        return "conversational"
    return "planning"


def _route_after_slot_gate(state: PlanningState):
    if state.followup_question:
        return END
    return [
        "dispatch_hotel",
        "dispatch_restaurant",
        "dispatch_route",
        "dispatch_event",
    ]


def _route_after_dispatch(state: PlanningState) -> str:
    intent = state.intent
    return "direct" if (intent and intent.route == "direct") else "full"


def build_travel_graph():
    """Compile and return the top-level travel StateGraph."""
    subgraph = _build_itinerary_subgraph()

    g = StateGraph(PlanningState)

    g.add_node("classify_intent", classify_intent)
    g.add_node("answer_conversational", answer_conversational)
    g.add_node("hydrate_trip", hydrate_trip)
    g.add_node("check_slot_gate", check_slot_gate)
    g.add_node("dispatch_hotel", dispatch_hotel)
    g.add_node("dispatch_restaurant", dispatch_restaurant)
    g.add_node("dispatch_route", dispatch_route)
    g.add_node("dispatch_event", dispatch_event)
    g.add_node("post_dispatch", post_dispatch)
    g.add_node("merge_direct", merge_direct)
    g.add_node("itinerary_planning", subgraph)

    g.add_edge(START, "classify_intent")
    g.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "conversational": "answer_conversational",
            "planning": "hydrate_trip",
        },
    )
    g.add_edge("answer_conversational", END)

    g.add_edge("hydrate_trip", "check_slot_gate")
    g.add_conditional_edges("check_slot_gate", _route_after_slot_gate)

    # Fan-in: all four dispatch nodes converge on post_dispatch.
    for name in (
        "dispatch_hotel",
        "dispatch_restaurant",
        "dispatch_route",
        "dispatch_event",
    ):
        g.add_edge(name, "post_dispatch")

    g.add_conditional_edges(
        "post_dispatch",
        _route_after_dispatch,
        {"direct": "merge_direct", "full": "itinerary_planning"},
    )
    g.add_edge("merge_direct", END)
    g.add_edge("itinerary_planning", END)

    return g.compile()
