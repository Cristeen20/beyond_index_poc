import { useEffect, useState, type MouseEvent } from 'react'
import type { TripSummary } from '../types'

interface Props {
  userId: string
}

interface Segment {
  start_time: string
  title: string
  type: string
  cost: number
  location?: string
  description?: string
  item_ref?: string
}

interface RawDay {
  day_number: number
  date: string
  location: string
  accommodation?: { name: string; address?: string } | null
  segments?: Segment[]
  notes?: string[]
}

interface RawItinerary {
  title: string
  days: RawDay[]
  total_cost: number
  notes?: string[]
}

type Tab = 'itinerary' | 'stay' | 'dining' | 'options'

const TABS: { id: Tab; icon: string; label: string }[] = [
  { id: 'itinerary', icon: '🗓', label: 'Itinerary' },
  { id: 'stay',      icon: '🛏', label: 'Stay' },
  { id: 'dining',    icon: '🍽', label: 'Dining' },
  { id: 'options',   icon: '⚙',  label: 'Options' },
]

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
  })
}

function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

// ── Tab content panels ─────────────────────────────────────────────────────

function ItineraryTab({ data }: { data: RawItinerary }) {
  return (
    <div className="dash-tab-content">
      {data.days.map((d) => (
        <div key={d.day_number} className="dash-day">
          <div className="dash-day-header">
            <span className="dash-day-badge">Day {d.day_number}</span>
            <span className="dash-day-meta">{shortDate(d.date)} · {d.location}</span>
          </div>
          <ol className="dash-segments">
            {(d.segments || []).map((s, i) => {
              const time = s.start_time?.slice(11, 16) ?? ''
              return (
                <li key={i} className="dash-segment">
                  {time && <span className="dash-time">{time}</span>}
                  <span className="dash-seg-title">{s.title}</span>
                  {s.cost > 0 && <span className="dash-seg-cost">${s.cost.toFixed(0)}</span>}
                </li>
              )
            })}
          </ol>
          {d.notes?.length ? (
            <ul className="dash-day-notes">{d.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
          ) : null}
        </div>
      ))}
    </div>
  )
}

function StayTab({ data }: { data: RawItinerary }) {
  // Collect unique stays grouped by hotel name
  const stayGroups: { name: string; address?: string; nights: string[] }[] = []
  for (const d of data.days) {
    if (!d.accommodation) continue
    const existing = stayGroups.find((g) => g.name === d.accommodation!.name)
    if (existing) {
      existing.nights.push(shortDate(d.date))
    } else {
      stayGroups.push({
        name: d.accommodation.name,
        address: d.accommodation.address,
        nights: [shortDate(d.date)],
      })
    }
  }
  if (!stayGroups.length) {
    return <div className="dash-tab-content dash-tab-empty">No stays recorded.</div>
  }
  return (
    <div className="dash-tab-content">
      {stayGroups.map((g, i) => (
        <div key={i} className="dash-stay-card">
          <span className="dash-stay-icon">🛏</span>
          <div className="dash-stay-info">
            <strong>{g.name}</strong>
            {g.address && <span className="dash-stay-addr">{g.address}</span>}
            <span className="dash-stay-nights">{g.nights.join(', ')}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

function DiningTab({ data }: { data: RawItinerary }) {
  const meals = data.days.flatMap((d) =>
    (d.segments || [])
      .filter((s) => s.type === 'meal')
      .map((s) => ({ ...s, date: d.date, day: d.day_number }))
  )
  if (!meals.length) {
    return <div className="dash-tab-content dash-tab-empty">No dining segments recorded.</div>
  }
  return (
    <div className="dash-tab-content">
      {meals.map((m, i) => (
        <div key={i} className="dash-dining-row">
          <span className="dash-time">{m.start_time?.slice(11, 16) ?? ''}</span>
          <div className="dash-dining-info">
            <span className="dash-seg-title">{m.title}</span>
            {m.location && <span className="dash-dining-loc">{m.location}</span>}
          </div>
          <div className="dash-dining-right">
            {m.cost > 0 && <span className="dash-seg-cost">${m.cost.toFixed(0)}</span>}
            <span className="dash-dining-date">Day {m.day} · {shortDate(m.date)}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

function OptionsTab({ data }: { data: RawItinerary }) {
  return (
    <div className="dash-tab-content">
      <div className="dash-option-row">
        <span className="dash-option-label">Total days</span>
        <span className="dash-option-value">{data.days.length}</span>
      </div>
      <div className="dash-option-row">
        <span className="dash-option-label">Estimated cost</span>
        <span className="dash-option-value">~${data.total_cost.toFixed(0)}</span>
      </div>
      <div className="dash-option-row">
        <span className="dash-option-label">Dates</span>
        <span className="dash-option-value">
          {data.days[0]?.date && shortDate(data.days[0].date)}
          {data.days.length > 1 && ` – ${shortDate(data.days[data.days.length - 1].date)}`}
        </span>
      </div>
      {data.notes?.length ? (
        <div className="dash-option-notes">
          {data.notes.map((n, i) => <p key={i}>{n}</p>)}
        </div>
      ) : null}
    </div>
  )
}

function ExpandedItinerary({ data }: { data: RawItinerary }) {
  const [tab, setTab] = useState<Tab>('itinerary')

  return (
    <div className="dash-itin-layout">
      {/* Left tab column */}
      <nav className="dash-tab-nav">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`dash-tab-btn${tab === t.id ? ' active' : ''}`}
            onClick={() => setTab(t.id)}
          >
            <span className="dash-tab-icon">{t.icon}</span>
            <span className="dash-tab-label">{t.label}</span>
          </button>
        ))}
      </nav>
      {/* Right content */}
      <div className="dash-itin-panel">
        <div className="dash-itin-panel-header">
          <span className="dash-itin-title">{data.title}</span>
          <span className="dash-itin-cost">~${data.total_cost.toFixed(0)}</span>
        </div>
        {tab === 'itinerary' && <ItineraryTab data={data} />}
        {tab === 'stay'      && <StayTab data={data} />}
        {tab === 'dining'    && <DiningTab data={data} />}
        {tab === 'options'   && <OptionsTab data={data} />}
      </div>
    </div>
  )
}

// ── Main Dashboard ─────────────────────────────────────────────────────────

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

  async function handleDelete(e: MouseEvent, tripId: string) {
    e.stopPropagation()
    try {
      await fetch(`/trips/${tripId}`, { method: 'DELETE' })
      setTrips((prev) => prev.filter((t) => t.trip_id !== tripId))
      if (expanded === tripId) setExpanded(null)
    } catch {
      // silent
    }
  }

  async function toggleExpand(tripId: string) {
    if (expanded === tripId) { setExpanded(null); return }
    setExpanded(tripId)
    if (expandedData[tripId]) return
    setExpandLoading(tripId)
    try {
      const r = await fetch(`/trips/${tripId}`)
      const data: RawItinerary = await r.json()
      setExpandedData((prev) => ({ ...prev, [tripId]: data }))
    } catch {
      // leave empty
    } finally {
      setExpandLoading(null)
    }
  }

  if (loading) return <div className="dashboard"><div className="dashboard-empty">Loading your trips…</div></div>
  if (error)   return <div className="dashboard"><div className="dashboard-empty dashboard-error">{error}</div></div>
  if (!trips.length) return (
    <div className="dashboard">
      <div className="dashboard-empty">No saved trips yet — plan your first trip in the Chat tab!</div>
    </div>
  )

  return (
    <div className="dashboard">
      <h2 className="dashboard-heading">My Trips</h2>
      <div className="dashboard-list">
        {trips.map((t) => (
          <div key={t.trip_id} className="dashboard-card">
            <div className="dashboard-card-header">
              <button
                className="dash-card-expand"
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
                <span className="dash-card-chevron">{expanded === t.trip_id ? '▲' : '▼'}</span>
              </button>
              <button
                className="dash-card-delete"
                onClick={(e) => handleDelete(e, t.trip_id)}
                title="Delete trip"
                aria-label="Delete trip"
              >🗑</button>
            </div>

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
