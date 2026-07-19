"""Top-level travel StateGraph — v4 topology with session loop-back
+ shared planning lane + LLM slot-question node
(see itinerary_langgraph_flow.md).

  START ──► intent_decision  ◄──────────────────────────────┐
                │                                            │
      _route_after_intent  (τ + revise guard)                │
   ┌────────────┬────────────┬────────────┐                  │
   ▼            ▼            ▼            ▼                  │
conv         revise      planning                            │
   │            │            │                               │
   ▼            ▼            ▼                               │
answer_    revise       hydrate_trip                         │
conv       (subgraph)       │                                │
   │            │            ▼                               │
   │            │      check_slot_gate                       │
   │            │            │                               │
   │            │   _slot_gate_route                         │
   │            │    ┌────┬──┴────────┬────────────┐         │
   │            │    ▼    ▼           ▼            ▼         │
   │            │  ask_  hotel_sub /   [4 subgraphs]         │
   │            │  missing_ …          in parallel           │
   │            │  slots (DIRECT one)  (FULL)                │
   │            │    │            │       │                  │
   │            │    │            └───┬───┘                  │
   │            │    │                ▼                      │
   │            │    │          post_dispatch                │
   │            │    │       direct│full                     │
   │            │    │           │   │                       │
   │            │    │           ▼   ▼                       │
   │            │    │  merge_direct itinerary_planning      │
   │            │    │           │   │                       │
   └────────────┴────┴───────────┴───┴─────┐                 │
                                           ▼                 │
                                 wait_for_next_message ──────┘
                                     (interrupt)
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agent_models import PlanningState
from graph.nodes_dispatch import answer_from_places, merge_direct, post_dispatch
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
    ask_missing_slots,
    check_slot_gate,
    hydrate_trip,
    intent_decision,
    route_after_intent,
)
from graph.revise_graph import build_revise_graph
from graph.session import CHECKPOINTER, wait_for_next_message
from graph.subgraphs import (
    event_sub_node,
    hotel_sub_node,
    restaurant_sub_node,
    route_sub_node,
)


# --------------------------------------------------------------------------- #
# Itinerary sub-graph — Load → Budget → Schedule → Conflict → Repair? → Assemble.
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
# Edge functions
# --------------------------------------------------------------------------- #


_AGENT_TO_SUBGRAPH = {
    "hotel": "hotel_sub",
    "restaurant": "restaurant_sub",
    "route": "route_sub",
    "event": "event_sub",
}

_ALL_SUBGRAPHS = ["hotel_sub", "restaurant_sub", "route_sub", "event_sub"]


def _slot_gate_route(state: PlanningState):
    """Decide what happens after check_slot_gate wrote `missing_slots`.

    - Missing anything → ask_missing_slots (LLM phrases the question).
    - DIRECT + complete → single subgraph named by intent.target_agents[0].
    - FULL + complete → fan out to all four subgraphs.
    """
    if state.missing_slots:
        return "ask_missing_slots"

    intent = state.intent
    if intent is None or not intent.target_agents:
        # Defensive: nothing to dispatch, drop into ask so the user can help.
        return "ask_missing_slots"

    if intent.route == "direct":
        return _AGENT_TO_SUBGRAPH[intent.target_agents[0]]
    return _ALL_SUBGRAPHS


def _route_after_dispatch(state: PlanningState) -> str:
    """After post_dispatch fan-in:

    - DIRECT + answer_mode='answer' → answer_from_places (LLM synthesises
      a one-sentence factual answer about the top hit).
    - DIRECT + answer_mode='list'   → merge_direct (bullet list).
    - FULL                           → itinerary_planning.
    """
    intent = state.intent
    if intent and intent.route == "direct":
        return "answer" if intent.answer_mode == "answer" else "list"
    return "full"


# --------------------------------------------------------------------------- #
# Top-level travel graph
# --------------------------------------------------------------------------- #


def build_travel_graph():
    """Compile and return the top-level travel StateGraph.

    Compiled with the process-wide MemorySaver checkpointer so state
    persists across turns per thread_id (session). Terminal branches
    edge into `wait_for_next_message` (which calls interrupt()) and the
    resume edge goes back to `intent_decision`.
    """
    itinerary_subgraph = _build_itinerary_subgraph()
    revise_subgraph = build_revise_graph()

    g = StateGraph(PlanningState)

    # Router + intake
    g.add_node("intent_decision", intent_decision)
    g.add_node("answer_conversational", answer_conversational)
    g.add_node("hydrate_trip", hydrate_trip)
    g.add_node("check_slot_gate", check_slot_gate)
    g.add_node("ask_missing_slots", ask_missing_slots)

    # Per-agent subgraphs — shared by DIRECT (one) and FULL (all four).
    # Wrapped so each returns only its own *_options field; otherwise
    # parallel subgraph invocation would merge full state back and cause
    # concurrent writes on shared PlanningState channels.
    g.add_node("hotel_sub", hotel_sub_node)
    g.add_node("restaurant_sub", restaurant_sub_node)
    g.add_node("route_sub", route_sub_node)
    g.add_node("event_sub", event_sub_node)

    # Fan-in + downstream
    g.add_node("post_dispatch", post_dispatch)
    g.add_node("merge_direct", merge_direct)
    g.add_node("answer_from_places", answer_from_places)
    g.add_node("itinerary_planning", itinerary_subgraph)
    g.add_node("revise", revise_subgraph)

    # Loop-back node — interrupts and awaits the next user message
    g.add_node("wait_for_next_message", wait_for_next_message)

    # START → intent_decision → { conversational | planning | revise }
    g.add_edge(START, "intent_decision")
    g.add_conditional_edges(
        "intent_decision",
        route_after_intent,
        {
            "conversational": "answer_conversational",
            "planning": "hydrate_trip",
            "revise": "revise",
        },
    )

    # Planning lane (shared by DIRECT and FULL)
    g.add_edge("hydrate_trip", "check_slot_gate")
    g.add_conditional_edges("check_slot_gate", _slot_gate_route)

    # Fan-in: every subgraph edges into post_dispatch
    for name in _ALL_SUBGRAPHS:
        g.add_edge(name, "post_dispatch")

    g.add_conditional_edges(
        "post_dispatch",
        _route_after_dispatch,
        {
            "list": "merge_direct",
            "answer": "answer_from_places",
            "full": "itinerary_planning",
        },
    )

    # All terminal branches → wait_for_next_message → back to intent_decision
    for terminal in (
        "answer_conversational",
        "ask_missing_slots",
        "merge_direct",
        "answer_from_places",
        "itinerary_planning",
        "revise",
    ):
        g.add_edge(terminal, "wait_for_next_message")
    g.add_edge("wait_for_next_message", "intent_decision")

    return g.compile(checkpointer=CHECKPOINTER)
