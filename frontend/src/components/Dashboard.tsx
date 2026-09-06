import { useEffect, useState } from 'react'
import type { TripSummary } from '../types'

interface Props {
  userId: string
}

interface RawDay {
  day_number: number
  date: string
  location: string
  accommodation?: { name: string } | null
  segments?: Array<{ start_time: string; title: string; type: string; cost: number }>
  notes?: string[]
}

interface RawItinerary {
  title: string
  days: RawDay[]
  total_cost: number
  notes?: string[]
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

function ExpandedItinerary({ data }: { data: RawItinerary }) {
  return (
    <div className="dashboard-itinerary">
      <div className="dash-itin-header">
        <span className="dash-itin-title">{data.title}</span>
        <span className="dash-itin-cost">~${data.total_cost.toFixed(0)} total</span>
      </div>
      <div className="dash-days">
        {data.days.map((d) => (
          <div key={d.day_number} className="dash-day">
            <div className="dash-day-header">
              <strong>Day {d.day_number}</strong>
              <span className="dash-day-meta">{d.date} · {d.location}</span>
              {d.accommodation && (
                <span className="dash-hotel">🛏 {d.accommodation.name}</span>
              )}
            </div>
            <ol className="dash-segments">
              {(d.segments || []).slice(0, 8).map((s, i) => {
                const time = s.start_time?.slice(11, 16) ?? ''
                return (
                  <li key={i} className="dash-segment">
                    {time && <span className="dash-time">{time}</span>}
                    <span className="dash-seg-title">{s.title}</span>
                    {s.cost > 0 && (
                      <span className="dash-seg-cost">${s.cost.toFixed(0)}</span>
                    )}
                  </li>
                )
              })}
            </ol>
            {d.notes && d.notes.length > 0 && (
              <ul className="dash-day-notes">
                {d.notes.map((n, i) => <li key={i}>{n}</li>)}
              </ul>
            )}
          </div>
        ))}
      </div>
      {data.notes && data.notes.length > 0 && (
        <div className="dash-global-notes">
          {data.notes.map((n, i) => <p key={i}>{n}</p>)}
        </div>
      )}
    </div>
  )
}

export default function Dashboard({ userId }: Props) {
  const [trips, setTrips] = useState<TripSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [expandedData, setExpandedData] = useState<Record<string, RawItinerary>>({})
  const [expandLoading, setExpandLoading] = useState<string | null>(null)

  useEffect(() => {
    if (!userId) return
    setLoading(true)
    fetch(`/trips?user_id=${encodeURIComponent(userId)}`)
      .then((r) => r.json())
      .then((data) => { setTrips(data); setLoading(false) })
      .catch(() => { setError('Could not load trips.'); setLoading(false) })
  }, [userId])

  async function toggleExpand(tripId: string) {
    if (expanded === tripId) {
      setExpanded(null)
      return
    }
    setExpanded(tripId)
    if (expandedData[tripId]) return
    setExpandLoading(tripId)
    try {
      const r = await fetch(`/trips/${tripId}`)
      const data: RawItinerary = await r.json()
      setExpandedData((prev) => ({ ...prev, [tripId]: data }))
    } catch {
      // leave expandedData empty — expansion will show nothing
    } finally {
      setExpandLoading(null)
    }
  }

  if (loading) {
    return (
      <div className="dashboard">
        <div className="dashboard-empty">Loading your trips…</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="dashboard">
        <div className="dashboard-empty dashboard-error">{error}</div>
      </div>
    )
  }

  if (trips.length === 0) {
    return (
      <div className="dashboard">
        <div className="dashboard-empty">
          No saved trips yet — plan your first trip in the Chat tab!
        </div>
      </div>
    )
  }

  return (
    <div className="dashboard">
      <h2 className="dashboard-heading">My Trips</h2>
      <div className="dashboard-list">
        {trips.map((t) => (
          <div key={t.trip_id} className="dashboard-card">
            <button
              className="dashboard-card-header"
              onClick={() => toggleExpand(t.trip_id)}
              aria-expanded={expanded === t.trip_id}
            >
              <div className="dash-card-info">
                <span className="dash-card-title">{t.title}</span>
                <span className="dash-card-meta">
                  {t.destination && `${t.destination} · `}
                  {formatDate(t.created_at)}
                  {t.total_cost > 0 && ` · ~$${t.total_cost.toFixed(0)}`}
                </span>
              </div>
              <span className="dash-card-chevron">
                {expanded === t.trip_id ? '▲' : '▼'}
              </span>
            </button>

            {expanded === t.trip_id && (
              <div className="dashboard-card-body">
                {expandLoading === t.trip_id ? (
                  <div className="dash-expand-loading">Loading…</div>
                ) : expandedData[t.trip_id] ? (
                  <ExpandedItinerary data={expandedData[t.trip_id]} />
                ) : (
                  <div className="dash-expand-loading">Could not load details.</div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
