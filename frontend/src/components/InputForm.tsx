import { useState, type FormEvent, type KeyboardEvent } from 'react'

interface Props {
  onSubmit: (text: string) => void
  disabled?: boolean
}

export default function InputForm({ onSubmit, disabled }: Props) {
  const [text, setText] = useState('')

  function submit() {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSubmit(trimmed)
    setText('')
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    submit()
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  return (
    <form className="input-form" onSubmit={handleSubmit}>
      <div className="input-row">
        <textarea
          className="input-destination"
          placeholder="Ask about a destination, get recommendations, or request an itinerary… (Shift+Enter for new line)"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={2}
          disabled={disabled}
        />
        <button
          type="submit"
          className="btn-send"
          disabled={!text.trim() || disabled}
        >
          Send ↗
        </button>
      </div>
    </form>
  )
}
