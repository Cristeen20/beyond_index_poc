"""Graph state — re-exports PlanningState from agent_models.

The graph-only fields (`incoming_message`, `response_message`,
`followup_question`, `repair_attempts`, `chosen_route`, `budget`,
`draft_itinerary`, `conflict_notes`, `error_notes`, `changes_summary`)
live on the shared `PlanningState` so that the FastAPI adapters in
`travel_orchestrator.py` and the graph nodes work off the same schema.
The four `*_options` lists on `PlanningState` are annotated with an
`operator.add` reducer so the four parallel dispatch nodes can each
return a slice of state without racing.
"""

from agent_models import PlanningState

__all__ = ["PlanningState"]
