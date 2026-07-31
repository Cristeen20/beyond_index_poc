export interface ChatMessageItem {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface ChatRequest {
  message: string
  conversation_id: string
  history: ChatMessageItem[]
}

export interface ChatResponse {
  reply: string
  conversation_id: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  text?: string
  errorText?: string
  isLoading?: boolean
}
