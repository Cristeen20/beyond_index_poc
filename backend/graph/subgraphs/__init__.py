"""Per-agent compiled subgraphs + outer-graph node wrappers.

Each sub-agent (hotel, restaurant, route, event) is a compiled StateGraph
so the top-level travel/revise graphs can wire them in as single nodes
and each sub-agent can grow internal steps (search → filter → rank …)
later without touching the outer graphs.

Each module exports **both** the builder (`build_*_subgraph()`) and a
thin outer-graph wrapper (`*_sub_node`). The outer graphs use the
wrapper, not the compiled subgraph directly — otherwise the subgraph's
full final state gets merged back into the parent, causing concurrent
writes on shared PlanningState channels when four subgraphs run in
parallel.
"""

from graph.subgraphs.event import build_event_subgraph, event_sub_node
from graph.subgraphs.hotel import build_hotel_subgraph, hotel_sub_node
from graph.subgraphs.restaurant import (
    build_restaurant_subgraph,
    restaurant_sub_node,
)
from graph.subgraphs.route import build_route_subgraph, route_sub_node

__all__ = [
    "build_event_subgraph",
    "build_hotel_subgraph",
    "build_restaurant_subgraph",
    "build_route_subgraph",
    "event_sub_node",
    "hotel_sub_node",
    "restaurant_sub_node",
    "route_sub_node",
]
