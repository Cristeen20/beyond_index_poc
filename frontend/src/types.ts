// Kept simple frontend Itinerary shape (used by ItineraryCard) — separate
// from the backend agent_models.Itinerary. Wire it up when the card UI is
// re-enabled; today the plan flow renders everything as a text bubble.
export interface Stop {
  name: string
  time: string
  duration_minutes: number
  notes: string
}

export interface DayPlan {
  day: number
  theme: string
  stops: Stop[]
  lodging: string
}

export interface Itinerary {
  destination: string
  days: DayPlan[]
  advisories: string[]
}

export interface HistoryItem {
  role: string
  content: string
}

// ---- Backend /plan contract (agent_models.PlanResponse) ------------------- //

export interface IntentClassification {
  route: 'conversational' | 'direct' | 'full' | 'revise'
  target_agents: string[]
  extracted_slots: Record<string, string>
  missing_required_slots: string[]
  confidence: number
  rationale?: string | null
  answer_mode?: 'list' | 'answer'
}

// Untyped payload — direct_result items are agent-shaped dicts from the
// backend (HotelOption / RestaurantOption / RouteOption / EventOption
// model_dump). Rendered via formatPlanResponse in utils/format.ts.
export type DirectResultItem = Record<string, unknown> & { agent?: string; name?: string }

export interface OptionAction {
  action: 'select' | 'more' | 'question' | 'confirm' | 'correct'
  ids?: string[]
  text?: string
}

export interface PlanRequest {
  message: string
  session_id: string
  history?: HistoryItem[]
  option_action?: OptionAction
}

// Structured selection card (features/pre_planning.md). When present the
// UI renders it instead of the plain assistant bubble.
export interface OptionsPayloadItem {
  id: string
  name: string
  rank: number
  rationale: string
  meta?: Record<string, unknown>
}

export interface OptionsPayloadAction {
  id: string
  label: string
}

export interface OptionsPayload {
  kind: 'confirm_basics' | 'scope' | 'places' | 'stays' | 'day_by_day'
  title: string
  description?: string
  items: OptionsPayloadItem[]
  actions: OptionsPayloadAction[]
  select: 'single' | 'multi' | 'none'
  page: number
  has_more: boolean
}

export interface PlanResponse {
  route: 'conversational' | 'direct' | 'full' | 'revise'
  intent: IntentClassification
  itinerary?: unknown | null
  direct_result?: DirectResultItem[] | null
  followup_question?: string | null
  options_payload?: OptionsPayload | null
  message: string
  session_id: string
}

// ---- Chat UI state -------------------------------------------------------- //

export interface Message {
  id: string
  role: 'user' | 'assistant'
  text?: string
  itinerary?: Itinerary
  optionsPayload?: OptionsPayload
  errorText?: string
  isLoading?: boolean
  // Synthetic UI-only bubbles (e.g. "Chose: day_by_day" summarising a
  // button click). Excluded from `history` sent on subsequent requests so
  // they don't pollute the classifier's view of the conversation.
  synthetic?: boolean
}
