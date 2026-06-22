import { useState, useRef, useEffect } from 'react'
import type { Message, ItineraryRequest, Itinerary } from './types'
import ChatMessage from './components/ChatMessage'
import InputForm from './components/InputForm'
import './App.css'

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      itinerary: undefined,
      errorText: undefined,
      isLoading: false,
    },
  ])
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleSubmit(req: ItineraryRequest) {
    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      userRequest: req,
    }
    const loadingMsg: Message = {
      id: crypto.randomUUID(),
      role: 'assistant',
      isLoading: true,
    }

    setMessages((prev) => [...prev, userMsg, loadingMsg])

    try {
      const res = await fetch('/itinerary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      })

      if (!res.ok) {
        const text = await res.text()
        throw new Error(`${res.status}: ${text}`)
      }

      const itinerary: Itinerary = await res.json()

      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? { ...m, isLoading: false, itinerary }
            : m,
        ),
      )
    } catch (err) {
      const errorText =
        err instanceof Error ? err.message : 'Unknown error'
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? { ...m, isLoading: false, errorText }
            : m,
        ),
      )
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <span className="header-icon">✈</span>
        <h1>Trip Itinerary Generator</h1>
        <span className="header-badge">POC</span>
      </header>

      <main className="chat-window">
        <div className="welcome-hint">
          Describe a trip — destination, how many days, and what you enjoy.
          The backend will fetch live place data and generate a day-by-day
          itinerary.
        </div>
        {messages.slice(1).map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </main>

      <footer className="chat-footer">
        <InputForm onSubmit={handleSubmit} />
      </footer>
    </div>
  )
}
