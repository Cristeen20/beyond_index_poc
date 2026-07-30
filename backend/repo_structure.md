# Repository Structure

## Directory Tree

```
backend/
├── main.py                          # FastAPI entry point
├── agent_models.py                  # All Pydantic models (shared schema)
├── intake_router.py                 # LLM intent classifier
├── places.py                        # Google Places API wrapper
├── sub_agents.py                    # Sub-agent implementations
├── itinerary_agent.py               # Planning engine
├── travel_orchestrator.py           # FastAPI → Graph adapter
├── graph/
│   ├── __init__.py                  # Re-exports graph builders
│   ├── state.py                     # Re-exports PlanningState
│   ├── session.py                   # Checkpointer + interrupt logic
│   ├── travel_graph.py              # Top-level StateGraph topology
│   ├── revise_graph.py              # Revision StateGraph topology
│   ├── nodes_router.py              # Router graph nodes
│   ├── nodes_dispatch.py            # Dispatch fan-in nodes
│   ├── nodes_itinerary.py           # Itinerary sub-graph nodes
│   └── subgraphs/
│       ├── __init__.py              # Exports subgraph builders + wrappers
│       ├── hotel.py                 # Hotel subgraph
│       ├── restaurant.py            # Restaurant subgraph
│       ├── route.py                 # Route subgraph
│       └── event.py                 # Event subgraph
├── repo_structure.md                # this file
└── requirements.txt                 # Dependencies
```

---

## File-by-File Breakdown

### `main.py` — FastAPI Entry Point

- Loads `.env` via `dotenv`
- Configures logging for all internal modules
- Registers endpoints:
  - `POST /plan` — session-checkpointed LangGraph flow
  - `POST /revise` — standalone revision subgraph
  - `GET /health` — health check
- Delegates to `travel_orchestrator.plan()` / `.revise()`

---

### `agent_models.py` — Shared Schema Layer

Every Pydantic model in the system. Single source of truth.

| Category | Models |
|---|---|
| User context | `UserPreferences`, `UserProfile`, `TripSummary`, `TripRequest` |
| Sub-agent output | `TransportSegment`, `RouteOption`, `HotelOption`, `RestaurantOption`, `EventOption` |
| Itinerary | `TimeSegment`, `DayPlan`, `BudgetBreakdown`, `Itinerary` |
| Router contract | `IntentClassification` (route enum, answer_mode, target_agents, extracted_slots, etc.) |
| Required slots | `REQUIRED_SLOTS` dict per agent |
| Master state | `PlanningState` — flows through every LangGraph node |
| API wrappers | `PlanRequest`/`PlanResponse`, `ReviseRequest`/`ReviseResponse` |

Imported by **every other backend module**.

---

### `intake_router.py` — LLM Intent Classifier

- `classify()` — calls GPT-4o tool-call to classify message into: `conversational`, `direct`, `full`, `revise`
- Extracts slots (destination, dates, etc.) from message + prior context
- Confidence gate (τ = 0.55): low-confidence DIRECT → escalate to FULL
- Merge-vs-reclassify logic for answering follow-up slot questions
- `slot_gate()` — deterministic check for missing required slots

Called by `graph/nodes_router.py`'s `intent_decision` node.

---

### `places.py` — Google Places API Wrapper

- `fetch_places(destination, interests)` — calls `places.googleapis.com/v1/places:searchText`
- Normalizes response into clean dict (name, address, lat/lng, rating, types, hours, etc.)
- Deduplicates by name
- Reads `GOOGLE_MAPS_API_KEY` from env

Called by all four sub-agents in `sub_agents.py`.

---

### `sub_agents.py` — Sub-Agent Implementations

| Function | What it does | Data Source |
|---|---|---|
| `run_hotel_agent()` | Searches hotels, ranks by star+score | Google Places |
| `run_restaurant_agent()` | Searches restaurants, classifies cuisine/meal | Google Places |
| `run_event_agent()` | Searches attractions, classifies event type | Google Places |
| `run_route_agent()` | Creates flight/train options | **Mock** (hardcoded) |

Each accepts `TripRequest`, optional `Preferences`, optional `place_name`. Maps raw data to typed Pydantic option objects. Does post-processing (price estimation, cuisine guessing, etc.).

Called by the four subgraph modules in `graph/subgraphs/`.

---

### `itinerary_agent.py` — Planning Engine

- `load_agent_data()` — validates agent outputs
- `allocate_budget()` — splits budget by travel style ratios (relaxed/balanced/packed)
- `_build_planner_prompt()` — constructs LLM prompt with all options + budget
- `_run_llm_planner()` — calls GPT-4o tool-call for day-by-day schedule
- `resolve_conflicts()` — deterministic time/budget overlap checks
- `run_planning()` — main: Load → Budget → Schedule → Conflict → Repair → Assemble
- `revise_itinerary()` — revision: takes current itinerary + feedback, rebuilds
- `_generate_with_repair()` — LLM planner + one repair pass if needed

Pure helpers called by `graph/nodes_itinerary.py`.

---

### `travel_orchestrator.py` — FastAPI ↔ Graph Adapter

- `plan(req)` — checks session checkpoint. New: starts graph with `PlanningState`. Resume: sends `Command(resume=message)` to paused graph.
- `revise(req)` — builds `PlanningState`, invokes revision subgraph directly.
- Compiles both graphs once at import: `_TRAVEL_GRAPH`, `_REVISE_GRAPH`.

Called by `main.py`. Calls `build_travel_graph()` / `build_revise_graph()` from `graph/__init__.py`.

---

### `graph/__init__.py` — Package Exports

Re-exports: `build_travel_graph()`, `build_revise_graph()`

---

### `graph/state.py` — State Re-export

Re-exports `PlanningState` from `agent_models` under `graph.state` namespace.

---

### `graph/session.py` — Checkpointer & Interrupt

- `CHECKPOINTER`: in-memory `MemorySaver` instance
- `wait_for_next_message()`: calls `interrupt()` to pause. On resume, clears per-turn fields (`response_message`, `direct_result`, etc.) while preserving `intent`, `missing_slots`, `followup_question` for merge logic.

---

### `graph/travel_graph.py` — Top-Level Graph

The full user-message journey:

```
START → intent_decision → (route_after_intent)
         ↓
  ┌──────┼──────┐
conv   revise   planning
  │      │        ↓
  │      │    hydrate_trip → check_slot_gate → _slot_gate_route
  │      │                          ↓
  │      │            ask_missing_slots  OR  subgraphs (1 or 4)
  │      │                                    ↓
  │      │                              post_dispatch
  │      │                           direct│full
  │      │                              ↓    ↓
  │      │                          merge/  itinerary_planning
  │      │                          answer  (sub-graph)
  └──────┴────────────────────────────┬──────┘
                                      ↓
                             wait_for_next_message (interrupt → loop back)
```

**Edge functions**: `_slot_gate_route()`, `_route_after_dispatch()`

**Nested itinerary sub-graph**: Load → Budget → LLM Plan → Conflict Check → Repair? → Assemble

---

### `graph/revise_graph.py` — Revision Graph

```
START → _fanout_refetch (refetch only empty option lists)
         ↓
  ┌─────┼─────┬─────┐
hotel rest route event (if needed)
  └─────┴──┬──┴─────┘
           ↓
     allocate_budget
           ↓
     run_llm_planner_revision
           ↓
     check_conflicts ←──┐
    clean │  conflicts   │
          ↓              │
    assemble_revision    │
          ↓              │
         END             │
                  repair_planner ──┘
```

`_fanout_refetch()` only refetches lists that arrived empty. All cached → skip to `allocate_budget`.

---

### `graph/nodes_router.py` — Router Nodes

| Node | What it does |
|---|---|
| `intent_decision()` | Calls `intake_router.classify()` |
| `route_after_intent()` | Edge fn — confidence gate + revise guard |
| `answer_conversational()` | GPT-4o reply for chit-chat |
| `hydrate_trip()` | Builds `TripRequest` from classifier slots |
| `check_slot_gate()` | Computes missing required slots |
| `ask_missing_slots()` | GPT-4o-mini — natural language slot question |

---

### `graph/nodes_dispatch.py` — Dispatch Nodes

| Node | What it does |
|---|---|
| `post_dispatch()` | Fan-in join — records which agents returned data |
| `merge_direct()` | DIRECT list-mode: top-5 per agent + upsell offer |
| `answer_from_places()` | DIRECT answer-mode: GPT-4o-mini synthesizes one-sentence answer from top hit |

---

### `graph/nodes_itinerary.py` — Itinerary Nodes

| Node | What it does |
|---|---|
| `load_agent_data_node()` | Validates agent outputs |
| `allocate_budget_node()` | Picks cheapest route, allocates by style |
| `run_llm_planner_node()` | LLM call → `draft_itinerary` |
| `run_llm_planner_revision_node()` | Same but revision prompt |
| `check_conflicts_node()` | Hydrates days, runs `resolve_conflicts()` |
| `repair_planner_node()` | Single repair pass |
| `assemble_itinerary_node()` | Builds `Itinerary`, version=1 |
| `assemble_revision_node()` | Builds `Itinerary`, version bump + diff |
| `load_data_router()` | Edge: error → assemble, ok → continue |
| `conflict_router()` | Edge: repair → retry, assemble → done |

---

### `graph/subgraphs/__init__.py` — Subgraph Exports

Each subgraph module exports both:
1. `build_*_subgraph()` — returns compiled `StateGraph`
2. `*_sub_node()` — wrapper that returns **only one key** (e.g. `{"hotel_options": [...]}`) to avoid parallel write conflicts

---

### `graph/subgraphs/hotel.py` — Hotel Subgraph

- `fetch_hotel_node()` → `sub_agents.run_hotel_agent()`
- Returns `{"hotel_options": [...]}`

### `graph/subgraphs/restaurant.py` — Restaurant Subgraph

- `fetch_restaurant_node()` → `sub_agents.run_restaurant_agent()`
- Returns `{"restaurant_options": [...]}`

### `graph/subgraphs/route.py` — Route Subgraph

- `fetch_route_node()` → `sub_agents.run_route_agent()`
- Returns `{"route_options": [...]}`

### `graph/subgraphs/event.py` — Event Subgraph

- `fetch_event_node()` → `sub_agents.run_event_agent()`
- Returns `{"event_options": [...]}`

---

## How They Connect

```
HTTP Request
     │
     ▼
main.py  ──►  travel_orchestrator.py
                   │
                   ├── build_travel_graph()  ──► graph/travel_graph.py
                   │                                │
                   │                           nodes_router.py (classify, route, hydrate, slot gate)
                   │                                │
                   │                           subgraphs/*.py → sub_agents.py → places.py
                   │                                │
                   │                           nodes_dispatch.py (post_dispatch, merge, answer)
                   │                                │
                   │                           nodes_itinerary.py → itinerary_agent.py
                   │                                │
                   │                           session.py (wait_for_next_message / interrupt)
                   │
                   └── build_revise_graph() ──► graph/revise_graph.py
                                                    │
                                               subgraphs/*.py (refetch missing)
                                                    │
                                               nodes_itinerary.py (budget → revise → conflict → assemble)


All nodes share:  agent_models.PlanningState
```

---

## Key Patterns

- **StateGraph** — everything is a state machine; each node returns a delta dict
- **MemorySaver checkpointing** — sessions persist across turns; `Command(resume=)` feeds new input
- **Parallel subgraph fanout** — 4 sub-agents run in parallel; wrappers prevent concurrent write collisions
- **Single repair pass** — max 1 LLM retry for conflicts (bounded cost)
- **Confidence gating** — τ = 0.55; low-confidence DIRECT → escalate to FULL
- **LLM tiers** — GPT-4o for classification + planning; GPT-4o-mini for slot questions + factual answers

---

## External Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `httpx` | Async HTTP client (Google Places) |
| `openai` | GPT-4o / GPT-4o-mini |
| `pydantic` | Data modeling |
| `python-dotenv` | Env vars |
| `langgraph` | StateGraph + checkpointer |
| `twilio` | SMS (not used yet) |
