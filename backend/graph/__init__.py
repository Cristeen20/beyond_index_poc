"""LangGraph implementation of the Travel Orchestrator + Itinerary Agent flow.

Re-exports the two compiled StateGraphs the FastAPI layer needs:

- `build_travel_graph()` — top-level dual-path flow (Intake Router →
  DIRECT dispatch / FULL dispatch → Itinerary sub-graph). See itenary_agent.md
  §1 for the diagram this graph implements.

- `build_revise_graph()` — revision loop (§4 Step 6) as its own StateGraph.

Every routing / dispatch / LLM-invoking step is a graph node. Sub-agent
data fetchers (`sub_agents.run_hotel_agent` etc.) and pure helpers
(`itinerary_agent.allocate_budget`, `resolve_conflicts`) are imported by
the nodes — they are inputs, not agents in the LLM sense.
"""

from graph.travel_graph import build_travel_graph
from graph.revise_graph import build_revise_graph

__all__ = ["build_travel_graph", "build_revise_graph"]
