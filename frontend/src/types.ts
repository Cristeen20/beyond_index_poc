export interface ItineraryRequest {
  destination: string
  days: number
  interests: string[]
  startDate?: string
}

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

export interface Message {
  id: string
  role: 'user' | 'assistant'
  userRequest?: ItineraryRequest
  itinerary?: Itinerary
  errorText?: string
  isLoading?: boolean
}
