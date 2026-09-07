import { useState, useRef, useEffect } from 'react'
import type { Message, OptionAction, PlanRequest, PlanResponse } from './types'
import { formatPlanResponse } from './utils/format'
import ChatMessage from './components/ChatMessage'
import InputForm from './components/InputForm'
import Dashboard from './components/Dashboard'
import './App.css'

type Page = 'chat' | 'dashboard'

function getOrCreateUserId(): string {
  const key = 'destinationAgentUserId'
  let id = localStorage.getItem(key)
  if (!id) {
    id = crypto.randomUUID()
    localStorage.setItem(key, id)
  }
  return id
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [page, setPage] = useState<Page>('chat')
  const bottomRef = useRef<HTMLDivElement>(null)
  // One session_id per browser tab — reused across turns so the backend
  // resumes the LangGraph checkpointer instead of restarting from START.
  const sessionIdRef = useRef<string>(crypto.randomUUID())
  // Persistent user ID across sessions — stored in localStorage.
  const userIdRef = useRef<string>(getOrCreateUserId())

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function sendTurn(text: string, optionAction?: OptionAction) {
    // For structured option_action turns with no accompanying text, show a
    // short synthetic user bubble so the conversation reads naturally.
    // These are flagged synthetic=true so they're excluded from the
    // classifier's history — otherwise "Chose: day_by_day" gets treated as
    // real user speech on a subsequent classification.
    const isSynthetic = !text && !!optionAction
    const userText = text || (optionAction ? summarizeAction(optionAction) : '')
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      text: userText,
      synthetic: isSynthetic,
    }
    const loadingMsg: Message = { id: crypto.randomUUID(), role: 'assistant', isLoading: true }

    setMessages((prev) => [...prev, userMsg, loadingMsg])
    setIsLoading(true)

    // Build history from settled, real messages only. Loading, errors, and
    // synthetic button-summary bubbles are excluded.
    const history = messages
      .filter((m) => !m.isLoading && !m.errorText && !m.synthetic && m.text)
      .map((m) => ({ role: m.role, content: m.text! }))

    const req: PlanRequest = {
      message: text,
      session_id: sessionIdRef.current,
      history,
      user_profile: { user_id: userIdRef.current },
      ...(optionAction ? { option_action: optionAction } : {}),
    }

    try {
      const res = await fetch('/plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      })

      if (!res.ok) {
        const errText = await res.text()
        throw new Error(`${res.status}: ${errText}`)
      }

      const data: PlanResponse = await res.json()
      const rendered = formatPlanResponse(data)

      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? {
                ...m,
                isLoading: false,
                text: rendered,
                optionsPayload: data.options_payload ?? undefined,
                hasItinerary: !!data.itinerary,
              }
            : m,
        ),
      )
    } catch (err) {
      const errorText = err instanceof Error ? err.message : 'Unknown error'
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id ? { ...m, isLoading: false, errorText } : m,
        ),
      )
    } finally {
      setIsLoading(false)
    }
  }

  function handleSubmit(text: string) {
    sendTurn(text)
  }

  function handleOptionAction(action: OptionAction) {
    sendTurn(action.text || '', action)
  }

  // Only the most recent assistant message with an options_payload is
  // interactive — earlier cards are frozen history.
  const lastInteractiveId = [...messages]
    .reverse()
    .find((m) => m.role === 'assistant' && m.optionsPayload && !m.isLoading)?.id

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <span className="header-icon" aria-hidden>✈</span>
          <h1>AI Trip Planner</h1>
          <span className="header-badge">POC</span>
        </div>
        <nav className="nav-tabs" aria-label="Main navigation">
          <button
            className={`nav-tab${page === 'chat' ? ' active' : ''}`}
            onClick={() => setPage('chat')}
          >
            Chat
          </button>
          <button
            className={`nav-tab${page === 'dashboard' ? ' active' : ''}`}
            onClick={() => setPage('dashboard')}
          >
            My Trips
          </button>
        </nav>
      </header>

      {page === 'dashboard' ? (
        <Dashboard userId={userIdRef.current} />
      ) : (
        <>
          <main className="chat-window">
            {messages.length === 0 && (
              <div className="welcome-hint">
                Ask about destinations, get place recommendations, or request a full itinerary.
                <br />
                Try: "What are the best spots in Niagara?" or "I'm looking for a historic trip in Ontario — suggest some places."
              </div>
            )}
            {messages.map((msg) => (
              <ChatMessage
                key={msg.id}
                message={msg}
                onOptionAction={msg.id === lastInteractiveId ? handleOptionAction : undefined}
                optionActionDisabled={isLoading}
                onSaveToTrips={msg.hasItinerary ? () => setPage('dashboard') : undefined}
              />
            ))}
            <div ref={bottomRef} />
          </main>

          <footer className="chat-footer">
            <InputForm onSubmit={handleSubmit} disabled={isLoading} />
          </footer>
        </>
      )}
    </div>
  )
}

function summarizeAction(action: OptionAction): string {
  switch (action.action) {
    case 'select': {
      const ids = action.ids ?? []
      // num_days sends a single numeric id; show it as "N day(s)" so the
      // synthetic user bubble reads naturally.
      if (ids.length === 1 && /^\d+$/.test(ids[0])) {
        const n = parseInt(ids[0], 10)
        return `${n} day${n === 1 ? '' : 's'}`
      }
      return `Selected ${ids.length} option(s)`
    }
    case 'more':
      return 'More options'
    case 'confirm':
      return action.ids?.[0] ? `Chose: ${action.ids[0]}` : 'Confirmed'
    case 'correct':
      return action.text || 'Change'
    case 'question':
      return action.text || 'Question'
    default:
      return ''
  }
}
