import { useState, useRef, useEffect } from 'react'
import type { Message, ChatRequest, ChatResponse } from './types'
import ChatMessage from './components/ChatMessage'
import InputForm from './components/InputForm'
import './App.css'

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const conversationIdRef = useRef<string>(crypto.randomUUID())

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function handleSubmit(text: string) {
    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', text }
    const loadingMsg: Message = { id: crypto.randomUUID(), role: 'assistant', isLoading: true }

    setMessages((prev) => [...prev, userMsg, loadingMsg])
    setIsLoading(true)

    const history = messages
      .filter((m) => !m.isLoading && !m.errorText && m.text)
      .map((m) => ({ role: m.role, content: m.text! }))

    const req: ChatRequest = {
      message: text,
      conversation_id: conversationIdRef.current,
      history,
    }

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(req),
      })

      if (!res.ok) {
        const errText = await res.text()
        throw new Error(`${res.status}: ${errText}`)
      }

      const data: ChatResponse = await res.json()

      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingMsg.id
            ? { ...m, isLoading: false, text: data.reply }
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

  return (
    <div className="app">
      <header className="app-header">
        <span className="header-icon">💬</span>
        <h1>Direct Chat</h1>
        <span className="header-badge">v1</span>
      </header>

      <main className="chat-window">
        {messages.length === 0 && (
          <div className="welcome-hint">
            Chat freely with a travel-savvy assistant — no planning, no agents,
            just answers.
            <br />
            Try: "What's a good time to visit Kyoto?" or "How many days do I need for Iceland?"
          </div>
        )}
        {messages.map((msg) => (
          <ChatMessage key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </main>

      <footer className="chat-footer">
        <InputForm onSubmit={handleSubmit} disabled={isLoading} />
      </footer>
    </div>
  )
}
