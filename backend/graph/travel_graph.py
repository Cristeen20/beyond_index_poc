"""Top-level travel StateGraph — v5 topology.

Adds the pre-planning chain (features/pre_planning.md) for the FULL
route, replacing the old four-subgraph fan-out. DIRECT still fans out
to a single subgraph → post_dispatch → merge_direct / answer_from_places.

  START ──► intent_decision  ◄────────────────────────────────────┐
                │                                                  │
      _route_after_intent  (τ + revise guard)                      │
   ┌────────────┬────────────┬────────────┐                        │
   ▼            ▼            ▼                                     │
 conv        revise      planning                                  │
   │            │            │                                     │
   ▼            ▼            ▼                                     │
 answer_    revise       hydrate_trip                              │
 conv       (sub)         │                                        │
   │            │         ▼                                        │
   │            │   check_slot_gate                                │
   │            │         │                                        │
   │            │   _slot_gate_route                               │
   │            │   ┌──────┼───────┬──────────────────┐            │
   │            │   ▼      ▼       ▼                  ▼            │
   │            │  ask_   hotel_sub / …         confirm_basics     │
   │            │  missing_ (DIRECT one)          (FULL entry)     │
   │            │  slots      │                       │            │
   │            │             ▼                       │            │
   │            │        post_dispatch                │            │
   │            │        ┌───┴───┐                    │            │
   │            │        ▼       ▼                    │            │
   │            │   merge_    answer_                 ▼            │
   │            │   direct    from_places   [pre-planning chain]   │
   │            │        │       │                    │            │
   │            │        │       │      elicit_scope→propose_places│
   │            │        │       │      →propose_stays→            │
   │            │        │       │      fill_missing_agents→       │
   │            │        │       │      itinerary_planning         │
   │            │        │       │                    │            │
   └────────────┴────────┴───────┴────────────────────┴────────────┤
                                                                   ▼
                                                       wait_for_next_message
                                                          (interrupt)
                                                                   │
                                                     route_after_wait
                                                     ├─ pending_stage → parse_*
                                                     └─ else          → intent_decision
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
from graph.nodes_preplanning import (
    ask_num_days_node,
    confirm_basics_node,
    elicit_scope_node,
    fill_missing_agents_node,
    parse_confirm_node,
    parse_num_days_node,
    parse_places_reply_node,
    parse_scope_node,
    parse_stays_reply_node,
    propose_places_node,
    propose_stays_for_selection_node,
    route_after_parse_confirm,
    route_after_parse_num_days,
    route_after_parse_places,
    route_after_parse_scope,
    route_after_parse_stays,
    route_after_wait,
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
from graph.session import get_checkpointer, wait_for_next_message
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
    - FULL + complete → enter the pre-planning chain (confirm_basics →
      elicit_scope → propose_places → …). Skip confirm_basics if the user
      already confirmed for this trip in an earlier turn.
    """
    if state.missing_slots:
        return "ask_missing_slots"

    intent = state.intent
    if intent is None or not intent.target_agents:
        # Defensive: nothing to dispatch, drop into ask so the user can help.
        return "ask_missing_slots"

    if intent.route == "direct":
        return _AGENT_TO_SUBGRAPH[intent.target_agents[0]]
    # FULL → pre-planning. Order: confirm_basics → ask_num_days → elicit_scope.
    # Each step's sticky flag skips it once completed for this session.
    if not state.basics_confirmed:
        return "confirm_basics"
    if not state.num_days_confirmed:
        return "ask_num_days"
    return "elicit_scope"


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

    # Pre-planning nodes (features/pre_planning.md) — the FULL route flows
    # through these instead of the parallel four-subgraph fan-out.
    g.add_node("confirm_basics", confirm_basics_node)
    g.add_node("parse_confirm", parse_confirm_node)
    g.add_node("ask_num_days", ask_num_days_node)
    g.add_node("parse_num_days", parse_num_days_node)
    g.add_node("elicit_scope", elicit_scope_node)
    g.add_node("parse_scope", parse_scope_node)
    g.add_node("propose_places", propose_places_node)
    g.add_node("parse_places_reply", parse_places_reply_node)
    g.add_node("propose_stays_for_selection", propose_stays_for_selection_node)
    g.add_node("parse_stays_reply", parse_stays_reply_node)
    g.add_node("fill_missing_agents", fill_missing_agents_node)

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

    # ── Pre-planning chain (FULL route) ──────────────────────────────────
    # confirm_basics → wait → parse_confirm → ask_num_days | elicit_scope | hydrate_trip
    g.add_edge("confirm_basics", "wait_for_next_message")
    g.add_conditional_edges(
        "parse_confirm",
        route_after_parse_confirm,
        {
            "ask_num_days": "ask_num_days",
            "elicit_scope": "elicit_scope",
        },
    )
    # ask_num_days → wait → parse_num_days → elicit_scope (or stay if question)
    g.add_edge("ask_num_days", "wait_for_next_message")
    g.add_conditional_edges(
        "parse_num_days",
        route_after_parse_num_days,
        {
            "wait_for_next_message": "wait_for_next_message",
            "elicit_scope": "elicit_scope",
        },
    )
    # elicit_scope → wait → parse_scope → propose_places
    g.add_edge("elicit_scope", "wait_for_next_message")
    g.add_conditional_edges(
        "parse_scope",
        route_after_parse_scope,
        {"propose_places": "propose_places"},
    )
    # propose_places → wait → parse_places_reply → {wait|propose_places|
    # propose_stays_for_selection}
    g.add_edge("propose_places", "wait_for_next_message")
    g.add_conditional_edges(
        "parse_places_reply",
        route_after_parse_places,
        {
            "wait_for_next_message": "wait_for_next_message",
            "propose_places": "propose_places",
            "propose_stays_for_selection": "propose_stays_for_selection",
        },
    )
    # propose_stays_for_selection → wait → parse_stays_reply → {wait|
    # propose_stays_for_selection|fill_missing_agents}
    g.add_edge("propose_stays_for_selection", "wait_for_next_message")
    g.add_conditional_edges(
        "parse_stays_reply",
        route_after_parse_stays,
        {
            "wait_for_next_message": "wait_for_next_message",
            "propose_stays_for_selection": "propose_stays_for_selection",
            "fill_missing_agents": "fill_missing_agents",
        },
    )
    # fill_missing_agents → itinerary_planning (existing subgraph, unchanged)
    g.add_edge("fill_missing_agents", "itinerary_planning")

    # All terminal branches → wait_for_next_message. On resume, the
    # route_after_wait edge fn checks pending_stage: pre-planning parsers
    # if a card was up, otherwise intent_decision.
    for terminal in (
        "answer_conversational",
        "ask_missing_slots",
        "merge_direct",
        "answer_from_places",
        "itinerary_planning",
        "revise",
    ):
        g.add_edge(terminal, "wait_for_next_message")
    g.add_conditional_edges(
        "wait_for_next_message",
        route_after_wait,
        {
            "intent_decision": "intent_decision",
            "parse_confirm": "parse_confirm",
            "parse_num_days": "parse_num_days",
            "parse_scope": "parse_scope",
            "parse_places_reply": "parse_places_reply",
            "parse_stays_reply": "parse_stays_reply",
        },
    )

    return g.compile(checkpointer=get_checkpointer())
