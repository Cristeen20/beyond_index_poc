import { useState, type FormEvent } from 'react'
import type { ItineraryRequest } from '../types'

interface Props {
  onSubmit: (req: ItineraryRequest) => void
}

export default function InputForm({ onSubmit }: Props) {
  const [destination, setDestination] = useState('')
  const [days, setDays] = useState(3)
  const [interests, setInterests] = useState('')
  const [startDate, setStartDate] = useState('')
  const [expanded, setExpanded] = useState(false)

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!destination.trim()) return
    onSubmit({
      destination: destination.trim(),
      days,
      interests: interests
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean),
      startDate: startDate || undefined,
    })
    setDestination('')
    setInterests('')
    setStartDate('')
    setDays(3)
  }

  return (
    <form className="input-form" onSubmit={handleSubmit}>
      <div className="input-row">
        <input
          className="input-destination"
          type="text"
          placeholder="Where to? e.g. Kyoto, Japan"
          value={destination}
          onChange={(e) => setDestination(e.target.value)}
          required
        />
        <button
          type="button"
          className="btn-expand"
          title="More options"
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? '▲' : '▼'}
        </button>
        <button type="submit" className="btn-send" disabled={!destination.trim()}>
          Generate ↗
        </button>
      </div>

      {expanded && (
        <div className="input-extras">
          <label>
            <span>Days</span>
            <input
              type="number"
              min={1}
              max={14}
              value={days}
              onChange={(e) => setDays(Number(e.target.value))}
            />
          </label>
          <label>
            <span>Interests</span>
            <input
              type="text"
              placeholder="temples, food, hiking"
              value={interests}
              onChange={(e) => setInterests(e.target.value)}
            />
          </label>
          <label>
            <span>Start date</span>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </label>
        </div>
      )}
    </form>
  )
}
