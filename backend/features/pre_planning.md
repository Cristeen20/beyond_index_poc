# Pre-Planning Flow — Feature Spec

## What & Why

Before dropping the user into a full day-by-day itinerary, walk them through
a short structured intake so we plan the trip *they* want, not the trip our
defaults imply. The user picks the scope (places-only, places+stays, or
day-by-day) and hand-picks options at each step. UI buttons keep the loop
deterministic; the LLM stays for ranking and free-form Q&A.

This flow **replaces** the previous FULL route's straight-to-`itinerary_planning`
dispatch. DIRECT, CONVERSATIONAL, and REVISE paths are untouched.

## Scope (this version)

| Capability | In? |
|---|---|
| Confirm origin / destination / dates before planning | ✅ |
| Ask scope: places / places+stays / day-by-day | ✅ (buttons) |
| Ranked place options, count ∝ num_days, paginated | ✅ |
| "More options" pagination | ✅ |
| Q&A on shown options (LLM grounded by cached options) | ✅ |
| Stays biased to selected place coordinates | ✅ |
| Upsell to day-by-day after places+stays | ✅ |
| Auto-run day-by-day when user chose that scope | ✅ |
| Trip types (staycation / workation / adventure / field trip) | ❌ (later) |

## Graph Topology

Only the FULL path changes; sub-graph internals unchanged.

```
intent_decision
   │ (route=full)
   ▼
hydrate_trip → check_slot_gate
   │ (slots complete)
   ▼
confirm_basics ────interrupt──── user confirms | corrects
   │ (confirmed)
   ▼
elicit_scope ──────interrupt──── button: [Places] [Places+Stays] [Day-by-Day]
   │
   ▼
propose_places (fetch via event_sub + LLM ranker, page 0)
   │
   ▼
present_options ───interrupt──── user: select ids | "more" | free-text Q
   │
   ▼ parse_options_reply
   ├─ more     → propose_places (page+1) → present_options
   ├─ question → answer_from_options (LLM over cached items) → present_options
   └─ select   → record selected_place_ids
                    │
                    ├─ scope=places_only → END (upsell "Want stays too?")
                    └─ scope ∈ {places+stays, day_by_day}
                          ▼
                    propose_stays_for_selection (hotel_sub biased by coords + ranker)
                          │
                          ▼
                    present_options (stays) ── interrupt ── same three actions
                          │
                          ▼ parse_options_reply (stays)
                          └─ select → record selected_stay_ids
                                │
                                ├─ scope=places+stays → ask_day_by_day (button)
                                │      ├─ Yes → itinerary_planning (unchanged)
                                │      └─ No  → END
                                └─ scope=day_by_day  → itinerary_planning (unchanged)
```

`itinerary_planning` is the existing subgraph
(Load → Budget → LLM planner → Conflict → Assemble). It sees the user's
selections via new state fields and can prefer them when composing days.

## State Additions (PlanningState)

```python
planning_scope: Literal["places_only", "places_and_stays", "day_by_day"] | None
basics_confirmed: bool = False
options_payload: dict | None = None
   # {kind: "confirm_basics" | "scope" | "places" | "stays" | "day_by_day",
   #  title, description, items: [{id, name, rank, rationale, meta}],
   #  actions: [{id, label}], select: "single" | "multi" | "none",
   #  page: int, has_more: bool}
selected_place_ids: list[str] = []
selected_stay_ids: list[str] = []
options_page: int = 0
pending_stage: Literal[
    "confirm_basics", "scope", "places", "stays", "day_by_day"
] | None = None
```

`pending_stage` lets the pre-planning router know which node produced the
current `options_payload` so it knows where to resume on the next turn.

## API Additions

### `PlanRequest.option_action`

```python
class OptionAction(BaseModel):
    action: Literal["select", "more", "question", "confirm", "correct"]
    ids: list[str] = []      # for "select"
    text: str | None = None  # for "question", "correct"

class PlanRequest(BaseModel):
    ...
    option_action: OptionAction | None = None
```

If `option_action` is set, the orchestrator forwards a *structured* resume
payload; the router bypasses LLM classification and jumps straight to the
node bound to `pending_stage`.

If the user types free text instead (no `option_action`), a small LLM parses
their reply against the pending payload — this is the fallback path.

### `PlanResponse.options_payload`

Mirrors `PlanningState.options_payload`. The frontend renders it as an
`OptionsCard`. When present, the primary text bubble is suppressed in favor
of the card.

## UI: OptionsCard

- Renders `options_payload.items` as cards (`kind="places" | "stays"`) with
  rank, name, rationale, and a checkbox (or radio for `select="single"`).
- Renders `options_payload.actions` as buttons for the confirm/scope/yesno
  stages (`kind="confirm_basics" | "scope" | "day_by_day"`).
- Submit button posts a `PlanRequest` with `option_action.action="select"`
  and the checked ids.
- Free-typing still works — the input form remains active and sends
  `message` + no `option_action`.

## Ranking

- **Places ranker**: `event_sub` returns `EventOption[]`. A gpt-4o-mini prompt
  ranks and produces a one-sentence rationale per item ("Ideal on day 1 —
  central, low walking, opens early"). Count = `min(3 * num_days, 15)`. Pages
  after page 0 fetch fresh candidates from `run_event_agent`.
- **Stays ranker**: `hotel_sub` biased by the centroid of `selected_place_ids`
  coords. Ranker rationale emphasises travel-time savings ("~15 min to Fushimi
  Inari, ~10 min to Gion — cuts ~40 min/day vs. central Kyoto Station").

## Slot Assumptions

`origin`, `destination`, and `dates` are still gated by `check_slot_gate` /
`ask_missing_slots`. `confirm_basics` runs only after those are present —
the confirmation prompt echoes them back and offers a "Change" button that
maps to `option_action.action="correct"` with free text like "actually
starting from Delhi" and re-enters `intent_decision`.

## Files

```
backend/
├── graph/
│   ├── nodes_preplanning.py   NEW — the seven new node functions + rankers
│   ├── travel_graph.py        MODIFY — wire nodes into FULL path
│   └── nodes_router.py        MODIFY — pre-router that handles pending_stage
├── agent_models.py            MODIFY — state fields, OptionAction, options_payload
├── travel_orchestrator.py     MODIFY — forward option_action into resume payload
└── features/
    └── pre_planning.md        THIS FILE

frontend/src/
├── components/
│   └── OptionsCard.tsx        NEW — renders options_payload
├── types.ts                   MODIFY — options_payload, option_action
├── App.tsx                    MODIFY — render OptionsCard, POST option_action
└── App.css                    MODIFY — card + button styles
```

## Non-Goals

- No `trip_type` yet (staycation / workation / etc.) — a follow-up will add
  a `trip_type` slot on `TripRequest` and branch prompts.
- No booking, no calendar, no payment.
- No auto-rerank on preference edit — user restarts by typing a new message,
  which re-enters `intent_decision`.
- Selections are advisory hints to `itinerary_planning`; the planner may
  still substitute items to resolve conflicts.
