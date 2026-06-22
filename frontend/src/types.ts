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

export interface ChatHistoryItem {
  role: string
  content: string
}

export interface ChatRequest {
  message: string
  history: ChatHistoryItem[]
}

export interface ChatResponse {
  text: string
  itinerary?: Itinerary
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  text?: string
  itinerary?: Itinerary
  errorText?: string
  isLoading?: boolean
}
