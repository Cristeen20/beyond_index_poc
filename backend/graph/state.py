"""Graph state — re-exports PlanningState from agent_models.

The graph-only fields (`incoming_message`, `response_message`,
`followup_question`, `repair_attempts`, `chosen_route`, `budget`,
`draft_itinerary`, `conflict_notes`, `error_notes`, `changes_summary`)
live on the shared `PlanningState` so that the FastAPI adapters in
`travel_orchestrator.py` and the graph nodes work off the same schema.
Each `*_options` list is written by exactly one subgraph (hotel_sub →
hotel_options, etc.), so no reducer is needed — plain assignment
overwrites on each turn.
"""

from agent_models import PlanningState

__all__ = ["PlanningState"]
