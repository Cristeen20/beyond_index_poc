# Itinerary Agent — Architecture & Design

> This document defines the Itinerary Agent (Planning Engine), a specialized agent that assembles outputs from all other agents into a coherent day-by-day trip itinerary. It is the central reasoning step of the Travel Intelligence Platform.

---

## 1. Role & Position in the System

The Itinerary Agent runs **after** the Route, Hotel, Restaurant, and Event Discovery agents have returned their recommendations, and **before** the Booking Agent executes any confirmed items. It is the last reasoning step before user approval — but only on the **full multi-step path**. Not every request needs it: a specific, narrow query ("find 4-star hotels in Kyoto next weekend") is routed straight to the relevant agent(s) by the **Intake Router** (§1.5) and skips the Itinerary Agent entirely.

The system therefore branches on user intent up front. The diagram below shows both paths — the **Direct (agent-skipping) flow** on the left and the **Full multi-step flow** on the right.

```
                              ┌─────────────────┐
   User Request  ───────────► │  User Memory    │  preferences, history,
   (each turn)                │  Service        │  known slots (defaults)
                              └────────┬────────┘
                                       │ hydrate known slots
                                       ▼
                          ┌──────────────────────────┐
                          │   Travel Orchestrator     │  (entry / turn manager)
                          └────────────┬──────────────┘
                                       ▼
        ┌───────────────────────────────────────────────────────────────┐
        │                 INTAKE ROUTER  (LLM classifier)                │
        │                                                                │
        │   query + history + memory-slots                               │
        │        │                                                       │
        │        ▼                                                       │
        │   ┌──────────────┐   emits IntentClassification:               │
        │   │  Classify    │──►  { route, target_agents[],              │
        │   │  intent +    │      extracted_slots{},                    │
        │   │  extract     │      missing_required_slots[],             │
        │   │  entities    │      confidence, rationale }               │
        │   └──────┬───────┘                                            │
        │          ▼                                                    │
        │   ┌──────────────┐   confidence < τ  ─────────────┐          │
        │   │ Confidence   │   OR spans whole trip          │          │
        │   │ gate (τ)     │   OR vague / "plan a trip"     │          │
        │   └──────┬───────┘                                │          │
        │  route=  │ direct                     route=full  │          │
        └──────────┼───────────────────────────────────────┼──────────┘
                   ▼                                        ▼
   ┌───────────────────────────────┐        ┌─────────────────────────────────┐
   │  DIRECT / AGENT-SKIPPING FLOW │        │      FULL MULTI-STEP FLOW        │
   │                               │        │                                 │
   │  ┌─────────────────────────┐  │        │  §2 Intake (fill required +     │
   │  │ Per-agent SLOT GATE      │  │        │  stylistic questionnaire,       │
   │  │  Hotel  → {dest, dates}  │  │        │  defaulting from memory)        │
   │  │  Restr. → {dest}         │  │        │            │                    │
   │  │  Route  → {orig,dest,dt} │  │        │            ▼                    │
   │  │  Event  → {dest}         │  │        │  Parallel Agents (ALL 4):       │
   │  └───────────┬─────────────┘  │        │   Route ∥ Hotel ∥ Restaurant    │
   │      missing │  complete       │        │        ∥ Event                  │
   │   ┌──────────┘      │          │        │            │                    │
   │   ▼                 │          │        │            ▼                    │
   │ Ask ONE targeted    │          │        │  ┌──────────────────────────┐  │
   │ question (single    │          │        │  │  ITINERARY AGENT (§4)    │  │
   │ slot; NOT full      │          │        │  │  load→budget→schedule→   │  │
   │ intake)             │          │        │  │  conflict→generate       │  │
   │   └──────┐          │          │        │  └────────────┬─────────────┘  │
   │          ▼          ▼          │        │               ▼                 │
   │  ┌─────────────────────────┐   │        │        User Approval            │
   │  │ PARALLEL DISPATCH        │   │        │          │       ▲              │
   │  │ target subset only, e.g. │   │        │   approve│       │ revise       │
   │  │  Hotel ∥ Restaurant      │   │        │          ▼       │ (re-plan,    │
   │  │ (skip Route, Event,      │   │        │      Booking     │  version++)  │
   │  │  Itinerary composition)  │   │        │          │       └──────────────┤
   │  └───────────┬─────────────┘   │        │          ▼                      │
   │              ▼                 │        │     Notification Agent          │
   │  ┌─────────────────────────┐   │        └─────────────────────────────────┘
   │  │ MERGE → targeted answer  │   │
   │  │ (raw HotelOption /        │   │      ┌────────────────────────────────┐
   │  │  RestaurantOption lists,  │   │      │  Per-turn re-classification:   │
   │  │  NOT a full Itinerary)    │   │      │  a follow-up that widens scope │
   │  └───────────┬─────────────┘   │      │  ("now plan the whole trip")   │
   │              ▼                 │      │  re-enters the Router next turn│
   │  Return answer  ── offer ──────┼──────►  and is classified route=full  │
   │  ("want a full itinerary?" —   │      │  (no auto-escalation mid-flow) │
   │   non-escalating suggestion)   │      └────────────────────────────────┘
   └───────────────────────────────┘
```

**Legend** — `τ`: router confidence threshold below which it defaults to the full flow;
`∥`: agents run in parallel; the **Intake Router runs once per conversational turn**, so a
follow-up message is always re-classified fresh (there is no auto-escalation mid-flow).

---

## 1.5 Intake Router (Query Classifier)

The Intake Router is the **first node after the Orchestrator**. It runs on every incoming
turn and decides which of the two flows handles the request.

**Mechanism** — a single **LLM classifier** call. It receives the user's message, the
conversation history, and any slots already known from User Memory, and emits a structured
`IntentClassification` (see §3.4) naming the route, the target agent subset, the slots it
could extract, and a confidence score.

**Routing rules**

| Condition | Route | Handling |
|---|---|---|
| Trip/plan request, spans the whole trip, vague, or asks for a day-by-day schedule | `FULL` | Existing §2 intake → all 4 agents → Itinerary Agent (§4) |
| Query targets specific agent capabilities (hotels, restaurants, routes, events) with enough entities to act | `DIRECT` | Agent-skipping flow (§4.5) — dispatch only the named agent subset |
| Confidence `< τ`, contradictory, or genuinely ambiguous | `FULL` | Safe fallback — better to over-serve than mis-route |

**Missing-slot handling** — if the route is `DIRECT` but a *required* slot for a target
agent is absent (e.g. the Hotel Agent needs `destination` + `dates`), the router asks **one**
targeted question for the single missing slot and then dispatches. It never launches the
full §2 questionnaire from the direct flow — that is the distinguishing property of the
agent-skipping path. Required-slot sets per agent are defined in §3.4.

---

## 2. User Intake: Questions the Agent Needs to Ask

### 2.1 Essential Questions (collected upfront, always required)

| # | Question | Field | Type | Example |
|---|---|---|---|---|
| 1 | Where are you departing from? | `origin` | string (city) | "New York" |
| 2 | What's your destination? | `destination` | string (city/country) | "Tokyo" |
| 3 | What dates are you traveling? | `start_date`, `end_date` | date (ISO 8601) | "2026-09-01"–"2026-09-10" |
| 4 | How many travelers? | `travelers` | int | 2 |
| 5 | What's your total budget? | `budget` | float | 3000.00 |
| 6 | What currency? | `currency` | string (ISO 4217) | "USD" |

### 2.2 Stylistic Questions (can default from User Memory)

| # | Question | Field | Type | Options / Example |
|---|---|---|---|---|
| 7 | What's your travel style? | `travel_style` | enum | relaxed / balanced / packed |
| 8 | Any dietary restrictions or preferred cuisines? | `preferred_foods` | string[] | ["Italian", "Seafood", "Vegetarian"] |
| 9 | What types of activities interest you? | `activity_interests` | string[] | ["museums", "nature", "food", "shopping", "nightlife", "adventure"] |
| 10 | What hotel star rating do you prefer? | `preferred_hotel_rating` | int | 4 |
| 11 | How much walking per day is comfortable? | `walking_tolerance` | enum | low / medium / high |
| 12 | What's your preferred meal schedule? | `meal_timing` | {breakfast, lunch, dinner} | {"breakfast": "08:00", "lunch": "12:30", "dinner": "19:30"} |
| 13 | Preferred transportation mode? | `preferred_transport` | string[] | ["flights", "trains", "buses", "car_rental"] |

### 2.3 Constraint Questions

| # | Question | Field | Type | Example |
|---|---|---|---|---|
| 14 | Are there any must-include places or activities? | `must_include` | string[] | ["Senso-ji Temple", "Tsukiji Market"] |
| 15 | Any must-exclude activities? | `must_exclude` | string[] | ["theme parks"] |
| 16 | Maximum travel time per day? | `max_travel_time_per_day` | int (minutes) | 180 |
| 17 | Any accessibility needs? | `accessibility_needs` | string[] | ["wheelchair_accessible"] |
| 18 | Is this a special occasion? | `special_occasion` | string | "honeymoon", "birthday", "anniversary" |
| 19 | Do you need car rental? | `needs_car_rental` | bool | false |

### 2.4 Repeat / Revision Questions (on subsequent iterations)

| # | Question | Purpose |
|---|---|---|
| 20 | What would you like to change? | "Too packed", "Too expensive", "Swap hotel X for Y", "Add activity Z" |
| 21 | Adjust budget breakdown? | "Spend less on hotels, more on activities" |
| 22 | Remove a specific day's plan and regenerate? | Useful when one day feels off |

---

## 3. Core Data Models

### 3.1 Itinerary Agent Input (collected from all sources)

```python
# --- User Input ---
class UserProfile(BaseModel):
    user_id: str
    name: str
    preferences: UserPreferences
    travel_history: list[TripSummary] = []

class UserPreferences(BaseModel):
    preferred_foods: list[str] = []
    preferred_hotel_rating: int | None = None
    travel_style: Literal["relaxed", "balanced", "packed"] = "balanced"
    meal_timing: MealTiming | None = None
    walking_tolerance: Literal["low", "medium", "high"] = "medium"
    activity_interests: list[str] = []
    accessibility_needs: list[str] = []

class TripRequest(BaseModel):
    origin: str
    destination: str
    start_date: date
    end_date: date
    travelers: int
    total_budget: float
    currency: str
    must_include: list[str] = []
    must_exclude: list[str] = []
    special_occasion: str | None = None

# --- Agent Outputs (inputs to Itinerary Agent) ---
class RouteOption(BaseModel):
    route_id: str
    segments: list[TransportSegment]
    total_cost: float
    total_duration_minutes: int
    mode: Literal["flight", "train", "bus", "car"]

class TransportSegment(BaseModel):
    type: Literal["flight", "train", "bus", "car"]
    departure_location: str
    arrival_location: str
    departure_time: datetime
    arrival_time: datetime
    cost: float
    booking_ref: str | None = None

class HotelOption(BaseModel):
    hotel_id: str
    name: str
    location: str
    latitude: float
    longitude: float
    star_rating: int
    price_per_night: float
    total_cost: float
    score: float  # from hotel ranking strategy
    address: str
    amenities: list[str]
    check_in_time: time = time(15, 0)
    check_out_time: time = time(11, 0)

class RestaurantOption(BaseModel):
    restaurant_id: str
    name: str
    cuisine: str
    meal_type: Literal["breakfast", "lunch", "dinner", "cafe"]
    location: str
    latitude: float
    longitude: float
    avg_cost_per_person: float
    rating: float
    opening_hours: dict[str, str] | None = None

class EventOption(BaseModel):
    event_id: str
    name: str
    type: Literal["museum", "festival", "concert", "landmark", "nature", "shopping", "other"]
    location: str
    latitude: float
    longitude: float
    duration_minutes: int
    cost: float
    typical_hours: str | None = None
    best_time_of_day: Literal["morning", "afternoon", "evening", "flexible"] = "flexible"
    closed_days: list[str] = []
```

### 3.2 Itinerary Output (the agent's product)

```python
class Itinerary(BaseModel):
    trip_id: str
    user_id: str
    title: str
    days: list[DayPlan]
    total_cost: float
    budget_breakdown: BudgetBreakdown
    notes: list[str] = []
    created_at: datetime
    version: int = 1

class DayPlan(BaseModel):
    day_number: int
    date: date
    day_name: str  # e.g. "Monday"
    location: str  # city / area
    accommodation: HotelOption | None = None
    segments: list[TimeSegment]
    total_cost: float
    notes: list[str] = []

    @property
    def is_free_day(self) -> bool:
        return all(s.type == "free_time" for s in self.segments)

class TimeSegment(BaseModel):
    id: str
    start_time: time
    end_time: time
    type: Literal["travel", "activity", "meal", "rest", "free_time", "buffer"]
    title: str
    description: str
    location: str
    latitude: float | None = None
    longitude: float | None = None
    cost: float = 0.0
    item_ref: str | None = None  # references agent option IDs
    booking_status: Literal["pending", "approved", "booked", "cancelled"] = "pending"

class BudgetBreakdown(BaseModel):
    transport: float = 0.0
    accommodation: float = 0.0
    food: float = 0.0
    activities: float = 0.0
    other: float = 0.0
    total: float = 0.0
    remaining: float = 0.0
```

### 3.3 LangGraph State Contract

```python
class PlanningState(BaseModel):
    # User context
    user_profile: UserProfile | None = None
    trip_request: TripRequest | None = None

    # Agent outputs
    route_options: list[RouteOption] = []
    hotel_options: list[HotelOption] = []
    restaurant_options: list[RestaurantOption] = []
    event_options: list[EventOption] = []

    # Itinerary (current state)
    itinerary: Itinerary | None = None
    revision_feedback: str | None = None  # user's change request

    # Routing (set by the Intake Router, §1.5)
    intent: IntentClassification | None = None
    direct_result: list | None = None  # targeted output on the DIRECT path (not an Itinerary)

    # Workflow control
    phase: Literal[
        "routing", "intake", "agent_dispatch", "planning",
        "review", "revision", "approved", "booked", "direct_answer"
    ] = "routing"
    agent_outputs_received: dict[str, bool] = {}
```

### 3.4 Router Contract (Intake Router output)

The Intake Router (§1.5) emits an `IntentClassification`. This is the single object that
decides which flow runs and, on the direct path, which agents to dispatch.

```python
class IntentClassification(BaseModel):
    route: Literal["direct", "full"]
    target_agents: list[Literal["route", "hotel", "restaurant", "event"]] = []
    extracted_slots: dict[str, str] = {}     # e.g. {"destination": "Kyoto", "hotel_rating": "4"}
    missing_required_slots: list[str] = []
    confidence: float
    rationale: str | None = None
```

**Required-slot sets per agent** — used by the direct flow's slot gate (§4.5) to decide
whether it can dispatch immediately or must ask one targeted question:

| Agent | Required slots |
|---|---|
| Hotel | `destination`, `dates` |
| Restaurant | `destination` |
| Route | `origin`, `destination`, `dates` |
| Event | `destination` |

For a direct query spanning multiple agents, the union of their required slots applies;
any slot already present in `extracted_slots` (or hydrated from User Memory) is considered
satisfied.

---

## 4. Planning Engine: Internal Workflow

The Itinerary Agent runs as a multi-step process, implemented as a **LangGraph sub-graph**:

```
┌──────────────────────────────────────────────────────────┐
│                    Itinerary Agent                       │
│                                                          │
│  ┌─────────┐    ┌──────────┐    ┌──────────────┐        │
│  │  Budget  │    │ Schedule │    │   Conflict   │        │
│  │ Allocate │───►│   Days   │───►│   Resolve    │        │
│  └─────────┘    └──────────┘    └──────┬───────┘        │
│         ▲                              │                 │
│         │                              ▼                 │
│  ┌──────┴──────┐              ┌──────────────┐          │
│  │    Load     │              │   Generate   │          │
│  │ Agent Data  │              │  Structured  │          │
│  └─────────────┘              │  Itinerary   │          │
│                               └──────┬───────┘          │
│                                      │                   │
│                                      ▼                   │
│  ┌──────────────────────────────────────────┐            │
│  │   Present to User for Approval           │            │
│  └──────────────────────────────────────────┘            │
│                                      │                   │
│                        ┌─────────────┴─────────┐         │
│                        ▼                       ▼         │
│                  ┌──────────┐          ┌──────────────┐  │
│                  │ Approve  │          │  Revise /    │  │
│                  │ & Book   │          │  Loop Back   │──►
│                  └──────────┘          └──────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### Step-by-step breakdown:

**Step 1 — Load Agent Data**
- Collect validated outputs from Route, Hotel, Restaurant, Event agents
- Validate each against its schema (Pydantic)
- Flag any missing or partial outputs
- If critical data is missing (e.g., no route found), return error to orchestrator

**Step 2 — Budget Allocation**
- Parse `total_budget` from `TripRequest`
- Subtract known fixed costs (already chosen route, etc.)
- Allocate remaining budget across accommodation, food, activities, buffer
- Rules of thumb per travel style:
  - **Relaxed**: 40% accommodation / 25% food / 20% activities / 15% buffer
  - **Balanced**: 30% / 25% / 30% / 15%
  - **Packed**: 20% / 20% / 45% / 15%
- Flag if budget is insufficient for available options

**Step 3 — Day-by-Day Scheduling (the core reasoning)**

For each day of the trip:
1. **Determine location**: Where does the user sleep this night? (from Hotel Agent)
2. **Determine daily anchor**: Is there a fixed event? (flight arrival, must-visit attraction, reservation)
3. **Allocate meal slots**: Based on `meal_timing` preferences
4. **Fill with activities**: From Event Agent recommendations, filtered by location proximity, interest match
5. **Insert travel/buffer time**: Walking time, transit between locations
6. **Check constraints**: Walking tolerance, max travel time, budget per day
7. **Rest / Free time**: Ensure it's not over-scheduled

**Step 4 — Conflict Resolution**

Detect and resolve:
- **Time conflicts**: Museum recommended at the same time as lunch reservation → shift one
- **Location conflicts**: Recommended restaurant is on the other side of the city from the afternoon activity → pick the nearer one
- **Budget conflicts**: Total exceeds budget → suggest cheaper alternatives or reduce activities
- **Availability conflicts**: Attraction is closed on the proposed day → move to another day
- **Logical conflicts**: Flight departs at 6am but a late dinner is scheduled the night before → flag and suggest adjustment
- **Weather conflicts**: Outdoor activity scheduled for a rainy day → swap with indoor activity

**Step 5 — Generate Structured Output**
- Assemble `Itinerary` with all `DayPlan` and `TimeSegment` objects
- Attach `BudgetBreakdown`
- Score the itinerary (optional): "How well does this match user preferences?"
- Return to orchestrator for presentation to user

**Step 6 — Revision Loop**
- User sends natural language feedback: "Too packed on day 2, swap hotel for something cheaper"
- Parse revision request → identify what changed (budget, pace, specific items)
- Re-run affected steps (may re-trigger specific agent calls if new data needed)
- Generate new itinerary version (`version += 1`)

---

## 4.5 Direct (Agent-Skipping) Flow

The counterpart to §4. When the Intake Router (§1.5) returns `route=direct`, the request
bypasses the full intake questionnaire **and** the Itinerary Agent, dispatching only the
target agent subset and returning their raw options.

### Step-by-step breakdown:

**Step 1 — Classify**
- Router returns `route=direct`, `target_agents=[...]`, `extracted_slots={...}`, `confidence`.

**Step 2 — Slot check (gate)**
- For each target agent, compute `required − extracted = missing` (required sets in §3.4).
- If any required slot is missing, ask **one** targeted question for it and resume — never
  the full §2 questionnaire.

**Step 3 — Parallel dispatch**
- Invoke the target agent subset **concurrently** (the same agent contracts as §8).
- Skip non-selected agents and skip the Itinerary Agent composition entirely.

**Step 4 — Assemble targeted response**
- Merge / rank agent outputs (by proximity, rating, price) into a lightweight
  `direct_result` — a list of `HotelOption` / `RestaurantOption` / etc., **not** an `Itinerary`.

**Step 5 — Offer upgrade (non-escalating)**
- The reply may suggest "want a full itinerary?", but per the routing contract it does not
  auto-escalate. A follow-up planning request simply re-enters the Router (§1.5) next turn
  and is classified `route=full`.

```
┌────────────────────────────────────────────────────────────────────┐
│                  Direct (Agent-Skipping) Flow                       │
│                                                                     │
│   IntentClassification { target_agents, extracted_slots }           │
│              │                                                      │
│              ▼                                                      │
│   ┌────────────────────┐   for each target agent:                  │
│   │  Slot Gate          │   required − extracted = missing         │
│   └─────────┬──────────┘                                           │
│    missing? │                                                       │
│      ┌──────┴───────┐                                               │
│      ▼ yes          ▼ no                                            │
│  ┌─────────┐   ┌──────────────────────────────────────────┐        │
│  │ Ask ONE │   │        Parallel Agent Dispatch            │        │
│  │ slot Q  │──►│                                           │        │
│  │ (loop   │   │   ┌────────┐  ┌────────────┐  ┌────────┐  │        │
│  │  until  │   │   │ Hotel   │  │ Restaurant  │  │ Event  │  │      │
│  │  filled)│   │   │ Agent   │∥ │ Agent       │∥ │ Agent  │  │      │
│  └─────────┘   │   └────┬────┘  └──────┬──────┘  └───┬────┘  │      │
│                │        └───────┬──────┴─────────────┘       │      │
│                │                ▼   (only selected agents)   │      │
│                │        ┌───────────────┐                    │      │
│                │        │  Merge / Rank  │  proximity, rating │      │
│                │        └───────┬───────┘                    │      │
│                └────────────────┼───────────────────────────┘      │
│                                 ▼                                   │
│                   direct_result: list[Option]                       │
│                   (NOT an Itinerary)                                 │
│                                 │                                   │
│                                 ▼                                   │
│                   Present targeted answer  +  optional              │
│                   "want a full itinerary?" (does NOT escalate)      │
└────────────────────────────────────────────────────────────────────┘
```

---

## 5. Implementation Phases

### Phase 1 — Foundation (MVP)

**Goal**: Working itinerary generator with manual agent inputs.

```
- Define all Pydantic models (inputs and outputs)
- Build a simple LLM-prompted planner (Option A from architecture.md)
- Implement basic budget allocation
- Produce a structured Itinerary JSON
- No user revision loop yet — single-shot generation
```

**Prompt design for Option A**:

```
You are a travel itinerary planner. Given the following data, create a day-by-day itinerary.

User Profile: {user_profile}
Trip Request: {trip_request}
Available Routes: {route_options}
Available Hotels: {hotel_options}
Available Restaurants: {restaurant_options}
Available Events: {event_options}

Budget Allocation: {budget_breakdown}

Rules:
1. Each day must have a reasonable schedule (not over-packed)
2. Include travel time between locations
3. Respect meal timing preferences
4. Stay within daily budget
5. Output valid JSON matching the Itinerary schema

Itinerary:
```

### Phase 1.5 — Intake Router (dual-path)

The router (§1.5) is added incrementally, on top of the same LLM tool-routing pattern the
POC already uses in `backend/orchestrator.py::chat` (where the model already chooses between
`fetch_places_data`, `return_itinerary`, and a plain reply). It is promoted from an implicit
in-loop decision to an explicit graph node — reuse, not new machinery.

```
- Phase 1: add the LLM router node + DIRECT / FULL split.
  Single-agent direct dispatch only; if a required slot is missing, default to FULL.
- Phase 2: parallel-subset fan-out (Hotel ∥ Restaurant …) + single-slot targeted question.
```

### Phase 2 — Revision & Memory

```
- Add the revision loop (user feedback → re-plan)
- Integrate User Memory to skip re-asking known preferences
- Add conflict detection and resolution logic
- Add "regenerate specific day" capability
```

### Phase 3 — Optimization (Hybrid — Option C)

```
- Replace LLM-only planning with LLM + constraint solver (OR-Tools)
- LLM generates candidate items and preferences
- Solver optimizes: minimize travel time, maximize preference match, respect constraints
- Add itinerary scoring: preference match %, budget efficiency, schedule density
- Add automatic rebalancing: if budget is under-utilized, suggest upgrades
```

### Phase 4 — Proactive & Adaptive

```
- "Surprise me" mode: LLM introduces novel suggestions outside user's stated preferences
- Weather-aware rerouting: detect forecast changes and reschedule
- Crowd-aware timing: suggest popular spots during off-peak hours
- Multi-city / road trip support: optimize across multiple destinations
```

---

## 6. Edge Cases & Failure Modes

| Scenario | Handling |
|---|---|
| No hotels available in budget | Flag to orchestrator, prompt user to adjust budget/location |
| No routes found to destination | Return error, suggest alternatives (nearby airports, multi-leg) |
| All restaurants booked for dinner | Recommend lunch instead, or suggest self-catering |
| Schedule overflows a single day | Split activity across 2 days, or ask user to prioritize |
| User gives contradictory preferences | Flag contradiction (e.g., "relaxed" + "maximize activities"), ask to clarify |
| Budget too low | Estimate minimum viable budget, show gap, ask user to adjust |
| Attractions closed on travel dates | Filter out before planning, suggest alternatives |
| Multi-timezone itinerary | Store all times in UTC + IANA timezone, display in local time |
| Group with conflicting preferences | Generate separate activity blocks, shared meals only |
| Special occasion falls on travel day | Prioritize premium options on that day, allocate extra budget |
| Router: ambiguous intent / low confidence (`< τ`) | Default to `FULL` — safer to over-serve than mis-route |
| Router: direct query missing a required slot | Ask one targeted question for that slot, then dispatch (never full intake) |
| Router: direct query spans agents with no shared location | Treat as `FULL` — no coherent targeted answer |
| Router: user widens scope mid-conversation (direct → "plan the whole trip") | Next turn re-classifies as `FULL`; router runs per turn (no mid-flow auto-escalation) |

---

## 7. Testing Strategy

| Concern | Approach |
|---|---|
| Schema validity | Unit test every Pydantic model with valid + invalid data |
| Budget math | Test allocation ratios, edge cases (zero budget, single day) |
| Prompt output parsing | Test LLM returns valid JSON matching Itinerary schema |
| Conflict detection | Known conflict scenarios → verify all flagged |
| Revision parsing | Natural language feedback → verify state changes correctly |
| Integration | Mock all 4 agent outputs, run full workflow, inspect state transitions |
| Itinerary quality | Human eval on 50+ trip scenarios, track preference match %, constraint satisfaction |
| Router accuracy | Labelled query set → assert correct `route` + `target_agents`; measure mis-route rate, tune `τ` |

---

## 8. Dependencies & Prerequisites

Before the Itinerary Agent can function, these must exist:

1. **Pydantic** — data validation foundation (`pip install pydantic`)
2. **LangGraph** — agent orchestration framework (`pip install langgraph`)
3. **LLM client** — OpenAI / Anthropic SDK or LiteLLM abstraction
4. **Route Agent** — provides `RouteOption` outputs
5. **Hotel Agent** — provides `HotelOption` outputs
6. **Restaurant Agent** — provides `RestaurantOption` outputs
7. **Event Discovery Agent** — provides `EventOption` outputs
8. **User Memory Service** — stores/retrieves `UserPreferences`
9. **Travel Orchestrator** — entry point that calls this agent

The Itinerary Agent can be **built and tested in isolation** using mock data for items 4–8, but it cannot be meaningfully integrated without them.

---

## 9. Key Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Intake routing | LLM intent classifier (`DIRECT` vs `FULL`) | Fits the existing LLM-driven design; handles natural phrasing; formalizes the tool-routing the POC's `/chat` loop already does |
| Missing-info on direct path | Ask one targeted slot question; do **not** fall back to full intake | Keeps the direct flow light — the whole point of skipping agents |
| Direct fan-out | Parallel subset of agents, skip Itinerary composition | Answer multi-agent queries ("hotel + ramen") without a full plan |
| Escalation | None mid-flow; router re-classifies every turn | Predictable, stateless routing; scope changes are just the next turn |
| Planner approach | Phase 1: LLM-driven (Option A) → Phase 3: Hybrid (Option C) | Fastest path to working prototype, then optimize |
| State management | LangGraph typed state (Pydantic) | Built-in validation, serialization, versioning |
| Budget model | Percentage-based allocation by travel style | Simple, tunable, explainable |
| Conflict resolution | LLM detects + proposes fix → structured rules validate | Combines flexibility with safety |
| Output format | Structured Pydantic → JSON | Serialization-safe, frontend-ready |
| Revision model | Full re-plan vs. partial re-plan based on change scope | Simple for MVP (full re-plan), optimize later (partial) |
