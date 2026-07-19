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

export interface PlanRequest {
  message: string
  session_id: string
  history?: HistoryItem[]
}

export interface PlanResponse {
  route: 'conversational' | 'direct' | 'full' | 'revise'
  intent: IntentClassification
  itinerary?: unknown | null
  direct_result?: DirectResultItem[] | null
  followup_question?: string | null
  message: string
  session_id: string
}

// ---- Chat UI state -------------------------------------------------------- //

export interface Message {
  id: string
  role: 'user' | 'assistant'
  text?: string
  itinerary?: Itinerary
  errorText?: string
  isLoading?: boolean
}
