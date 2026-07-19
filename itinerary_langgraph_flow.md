# Itinerary LangGraph Flow

LangGraph-only implementation of the Travel Orchestrator + Itinerary Agent.
Every routing, dispatch, and LLM step is a graph node or edge function —
no plain-Python orchestration wrappers left. Two compiled `StateGraph`s
cover the whole surface:

1. **Travel graph** (`graph/travel_graph.py`) — session-checkpointed
   dual-path flow with loop-back. Handles all `/plan` traffic.
2. **Revise graph** (`graph/revise_graph.py`) — the revision subgraph.
   Used two ways: embedded as a node inside the travel graph (for
   `route=revise` classifications) and invoked standalone from the
   `/revise` endpoint (for the explicit approve/revise UI).

Both graphs share `PlanningState` (`agent_models.PlanningState`). Each
`*_options` list is written by exactly one subgraph (`hotel_sub` →
`hotel_options`, etc.), so there is no reducer — plain assignment lets a
fresh turn cleanly overwrite the prior turn's list.

Each sub-agent (`hotel`, `restaurant`, `route`, `event`) is a compiled
subgraph under `graph/subgraphs/`. Today each is a single fetch node
wrapping `run_*_agent`; the shape leaves room to grow internal steps
without touching the outer graphs.

---

## 1. Session model

The travel graph is compiled with a process-wide `MemorySaver`
checkpointer (`graph/session.py:CHECKPOINTER`). Each session is a
LangGraph *thread*, keyed by `session_id`, which the **frontend
generates** (UUID) and reuses across turns.

`PlanRequest.session_id` is required. The orchestrator uses
`graph.get_state(config)` to detect whether the thread has state:

- **No state → fresh call.** Invoke the graph with the initial
  `PlanningState`. Graph runs from START → `intent_decision`.
- **State with `next=('wait_for_next_message',)` → resume.** Invoke with
  `Command(resume=req.message)`. Graph unpauses inside
  `wait_for_next_message`, writes the new message onto `incoming_message`,
  edges back to `intent_decision`.

`trip_request` and `itinerary` persist across turns; per-turn scratch
(`response_message`, `followup_question`, `direct_result`, `error_notes`,
`conflict_notes`, `repair_attempts`) is reset by
`wait_for_next_message` on resume.

---

## 2. Travel graph topology

```
START ──► intent_decision  ◄──────────────────────────────────┐
              │                                                │
     _route_after_intent (τ + revise guard)                    │
   ┌──────────┬────────────┬────────────┐                      │
   ▼          ▼            ▼                                    │
conversational planning   revise                                │
   │            │            │                                  │
   ▼            ▼            ▼                                  │
answer_     hydrate_trip  revise (compiled                      │
conversational  │           subgraph)                           │
   │            ▼            │                                  │
   │      check_slot_gate    │                                  │
   │       (deterministic —  │                                  │
   │        writes missing_  │                                  │
   │        slots only)      │                                  │
   │            │            │                                  │
   │    _slot_gate_route     │                                  │
   │     ┌──────┴───────┬────────────┐                          │
   │     ▼              ▼            ▼                          │
   │  ask_missing_  hotel_sub /   [hotel_sub ∥ restaurant_sub   │
   │  slots         restaurant_    ∥ route_sub ∥ event_sub]     │
   │  (LLM node —   sub /          (FULL — parallel fan-out)    │
   │   gpt-4o-      route_sub /                │                │
   │   mini)        event_sub                  │                │
   │     │          (DIRECT — one)             │                │
   │     │              │                      │                │
   │     │              └──────────┬───────────┘                │
   │     │                         ▼                            │
   │     │                   post_dispatch                      │
   │     │                 direct│full                          │
   │     │                     │  │                             │
   │     │                     ▼  ▼                             │
   │     │            merge_direct  itinerary_planning          │
   │     │                     │  │                             │
   └─────┴─────────────────────┴──┴───┐                         │
                                      ▼                         │
                            wait_for_next_message ──────────────┘
                                (interrupt)
```

### Nodes

| Node                    | File                      | Role                                                                                                    |
| ----------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------- |
| `intent_decision`       | `nodes_router.py`         | LLM classifier with session context (trip_request + itinerary) → `state.intent`.                        |
| `answer_conversational` | `nodes_router.py`         | LLM reply for chit-chat / low-confidence cases; no dispatch.                                            |
| `hydrate_trip`          | `nodes_router.py`         | Fill `trip_request` from extracted slots + defaults. Shared by DIRECT and FULL — the fanout shape is decided later. |
| `check_slot_gate`       | `nodes_router.py`         | Deterministic — computes missing required slots for `intent.target_agents` and writes `state.missing_slots`. Does NOT phrase the question. |
| `ask_missing_slots`     | `nodes_router.py`         | LLM node (`gpt-4o-mini`) — given the target agents, known slots, missing slots, and recent history, produces one short natural question. Writes `followup_question` + `response_message`. Batches related slots ("dates" = start+end) into one ask. |
| `hotel_sub`             | `subgraphs/hotel.py`      | Compiled subgraph — calls `run_hotel_agent`, writes `hotel_options`.                                    |
| `restaurant_sub`        | `subgraphs/restaurant.py` | Same, for restaurants.                                                                                  |
| `route_sub`             | `subgraphs/route.py`      | Same, for transport.                                                                                    |
| `event_sub`             | `subgraphs/event.py`      | Same, for activities.                                                                                   |
| `post_dispatch`         | `nodes_dispatch.py`       | Fan-in join for the subgraph(s); records `agent_outputs_received`.                                      |
| `merge_direct`          | `nodes_dispatch.py`       | DIRECT reply: flatten top-5 into `direct_result` + non-escalating upsell ("want the full trip?").       |
| `itinerary_planning`    | `travel_graph.py`         | FULL: embedded compiled sub-graph (Load → Budget → Plan → Conflict → Repair? → Assemble).               |
| `revise`                | `revise_graph.py`         | REVISE: embedded compiled sub-graph (see §4).                                                           |
| `wait_for_next_message` | `session.py`              | Calls `interrupt()` to pause; on resume, writes new message onto `incoming_message` and resets scratch. |

### Edge functions

- **`route_after_intent(state)`** — applies confidence gate + revise
  gating. Order:
  1. `intent is None` or `confidence < τ` → `"conversational"`.
  2. `intent.route == "conversational"` → `"conversational"`.
  3. `intent.route == "revise"` AND `state.itinerary is not None` → `"revise"`.
  4. Otherwise → `"planning"` (shared lane for DIRECT and FULL).

  Direct/full share the entry into `hydrate_trip` → `check_slot_gate`;
  the dispatch shape is decided by `_slot_gate_route`.

- **`_slot_gate_route(state)`** — reads `state.missing_slots` (written by
  `check_slot_gate`) and `intent.route`:
  1. If `missing_slots` non-empty → `"ask_missing_slots"` (LLM phrases
     the question).
  2. Else if `intent.route == "direct"` → single subgraph name matching
     `intent.target_agents[0]`.
  3. Else (FULL) → the list `["hotel_sub", "restaurant_sub", "route_sub",
     "event_sub"]` for parallel dispatch.

- **`_route_after_dispatch(state)`** — after fan-in at `post_dispatch`.
  Returns `"direct"` (→ `merge_direct`) or `"full"` (→ `itinerary_planning`).

### Mid-turn topic reset (bug fix)

The classifier receives a system message with `SESSION CONTEXT`
containing the persisted `trip_request` and `itinerary`. It uses these
to disambiguate follow-ups from resets:

- "what about omars inn" after biryani in Kannur → keeps
  `destination="Kannur"` from prior turn, routes DIRECT restaurant.
- "hey" after any prior context → routes CONVERSATIONAL (greeting is
  never a follow-up).
- "actually plan me a trip to Osaka" after a Kannur direct query →
  routes FULL with a fresh `trip_request`.

### Slot-answer follow-ups (merge vs. reclassify)

When we pause on `ask_missing_slots`, the pending `intent` and
`followup_question` are **preserved** through `wait_for_next_message`
(only `response_message` / `direct_result` / `error_notes` get reset).
On the next turn, `intent_decision` passes those as PENDING TURN
CONTEXT to the classifier along with the exact question we asked and
the still-missing slot list.

The classifier then explicitly picks:

- **(a) MERGE** — the message answers the pending question (fully or
  partially). Keep the prior `route` + `target_agents`; return
  `extracted_slots` = union of known + newly-extracted. Confidence
  ≥ 0.85. Runs when the follow-up is fragmentary ("in kyoto", "next
  weekend", "yes, 5-star").
- **(b) RECLASSIFY FRESH** — the message is a clear topic reset
  (greeting, unrelated question, a new destination the pending intent
  wasn't asking about, or a request that needs a different route).
  Classify normally, ignoring the pending state.

After (a), `ask_missing_slots` re-runs on the still-missing slots and
references what was just answered ("What dates are you thinking for
your trip to Kyoto?" instead of "Which destination did you have in
mind?"). Pending state gets **wiped on the dispatch path** by
`check_slot_gate` returning `followup_question=None` once all required
slots are present (Option B — wipe on dispatch).

---

## 3. Itinerary sub-graph (embedded in `itinerary_planning`)

```
       load_agent_data
              │
     error?  │  ok
    ┌────────┴────────┐
    ▼                 ▼
assemble_itinerary  allocate_budget
    │                 │
    │                 ▼
    │           run_llm_planner
    │                 │
    │                 ▼
    │          check_conflicts ◄──┐
    │           clean │ conflicts │
    │                 ▼           │
    │        assemble_itinerary   │
    │                 │           │
    │                 ▼           │
    └─────────────► (END)         │
                                  │
              repair_planner ─────┘  (bounded, MAX_REPAIR_ATTEMPTS = 1)
```

---

## 4. Revise subgraph

```
                              START
                                │
                                ▼
                      _fanout_refetch(state)
             for each agent whose *_options list is empty,
                     add its subgraph to the fanout;
             if all four are cached, jump straight to
                       "allocate_budget"
                                │
        ┌───────────┬───────────┼───────────┬──────────────┐
        ▼           ▼           ▼           ▼              │
   hotel_sub  restaurant_sub  route_sub  event_sub         │
        │           │           │           │              │
        └───────────┴─────┬─────┴───────────┘              │
                          ▼                                │
                  allocate_budget  ◄──────────────────────┘
                          │
                          ▼
             run_llm_planner_revision
                          │
                          ▼
                  check_conflicts ◄──┐
              clean │ conflicts      │
                    ▼                │
             assemble_revision       │
                    │                │
                    ▼                │
                  (END)              │
                                     │
              repair_planner ────────┘
```

Used two ways:

- **Embedded in the travel graph** as the `revise` node. Triggered when
  `intent.route == "revise"`. Reads `state.itinerary`,
  `state.revision_feedback` (set by classifier or prompt), and the
  cached option lists on state.

- **Standalone from `/revise`.** `travel_orchestrator.revise()` builds
  a `PlanningState` from the `ReviseRequest` payload and calls
  `_REVISE_GRAPH.ainvoke(initial)` directly. Not session-checkpointed —
  the request payload carries everything.

---

## 5. Shared state

`PlanningState` (`agent_models.py`) is the sole state channel. Fields
worth calling out:

- `intent` — populated by `intent_decision`; read by every edge fn.
- `trip_request` / `itinerary` — persist across turns via checkpointer.
- `route_options` / `hotel_options` / `restaurant_options` /
  `event_options` — plain lists (no reducer). Each is written by
  exactly one subgraph, so a fresh turn overwrites cleanly.
- `incoming_message` — reset by `wait_for_next_message` on resume.
- `missing_slots` — written by `check_slot_gate`, consumed by
  `_slot_gate_route` and `ask_missing_slots`. Per-turn scratch.
- `response_message` / `followup_question` / `direct_result` /
  `error_notes` / `conflict_notes` / `repair_attempts` — per-turn
  scratch, reset by `wait_for_next_message`.

---

## 6. Entry points

`travel_orchestrator.py`:

- `plan(req)` — checkpointed. First call for `req.session_id` runs from
  START; subsequent calls resume via `Command(resume=req.message)`.
- `revise(req)` — thin standalone wrapper over the revise subgraph, for
  the explicit `/revise` endpoint. Not session-checkpointed.

`main.py`:

- `POST /plan` — `PlanRequest{message, session_id, ...}` → `PlanResponse`.
- `POST /revise` — unchanged. Wraps the same revise subgraph.
- `POST /chat` — legacy shim; auto-generates `session_id` if absent and
  returns it in `ChatResponse` so the frontend can reuse it.

---

## 7. What was removed

- `sub_agents.dispatch_agents` + `AGENT_RUNNERS` — parallel
  `asyncio.gather` helper. Replaced by LangGraph subgraphs.
- `graph.nodes_dispatch.dispatch_hotel / dispatch_restaurant /
  dispatch_route / dispatch_event` — self-gating no-op nodes. Replaced
  by subgraphs + edge-function routing.
- `graph.nodes_itinerary.ensure_agent_data_node` — single-node
  `asyncio.gather` shortcut.
- `PlanningState.*_options` `operator.add` reducers — unnecessary
  (each field is written by exactly one subgraph) and would cause
  cross-turn accumulation with the checkpointer.
- `intake_router.build_slot_question` — hardcoded slot→string lookup
  table. Replaced by the `ask_missing_slots` LLM node.
- `hydrate_trip_direct` / `hydrate_trip_full` and
  `check_slot_gate_direct` / `check_slot_gate_full` node duplication —
  collapsed into single `hydrate_trip` and `check_slot_gate` nodes,
  with the fanout shape decided later in `_slot_gate_route`. Similarly
  `_pick_direct_agent` and `_fanout_full` are folded into
  `_slot_gate_route`.
