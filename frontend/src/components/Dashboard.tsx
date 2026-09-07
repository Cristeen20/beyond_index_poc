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

function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

// ── Tab panels ──────────────────────────────────────────────────────────────

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
  const groups: { name: string; address?: string; nights: string[] }[] = []
  for (const d of data.days) {
    if (!d.accommodation) continue
    const ex = groups.find((g) => g.name === d.accommodation!.name)
    if (ex) { ex.nights.push(shortDate(d.date)) }
    else { groups.push({ name: d.accommodation.name, address: d.accommodation.address, nights: [shortDate(d.date)] }) }
  }
  if (!groups.length) return <div className="dash-tab-content dash-tab-empty">No stays recorded.</div>
  return (
    <div className="dash-tab-content">
      {groups.map((g, i) => (
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
    (d.segments || []).filter((s) => s.type === 'meal').map((s) => ({ ...s, date: d.date, day: d.day_number }))
  )
  if (!meals.length) return <div className="dash-tab-content dash-tab-empty">No dining segments recorded.</div>
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
      <div className="dash-option-row"><span className="dash-option-label">Total days</span><span className="dash-option-value">{data.days.length}</span></div>
      <div className="dash-option-row"><span className="dash-option-label">Estimated cost</span><span className="dash-option-value">~${data.total_cost.toFixed(0)}</span></div>
      <div className="dash-option-row">
        <span className="dash-option-label">Dates</span>
        <span className="dash-option-value">
          {data.days[0]?.date && shortDate(data.days[0].date)}
          {data.days.length > 1 && ` – ${shortDate(data.days[data.days.length - 1].date)}`}
        </span>
      </div>
      {data.notes?.length ? <div className="dash-option-notes">{data.notes.map((n, i) => <p key={i}>{n}</p>)}</div> : null}
    </div>
  )
}

// ── Single trip detail view ─────────────────────────────────────────────────

function TripDetail({ summary, data, onBack }: { summary: TripSummary; data: RawItinerary; onBack: () => void }) {
  const [tab, setTab] = useState<Tab>('itinerary')
  return (
    <div className="dash-detail">
      <div className="dash-detail-topbar">
        <button className="dash-back-btn" onClick={onBack}>← My Trips</button>
        <div className="dash-detail-meta">
          <span className="dash-detail-title">{data.title}</span>
          <span className="dash-detail-sub">
            {summary.destination && `${summary.destination} · `}
            {formatDate(summary.created_at)}
            {data.total_cost > 0 && ` · ~$${data.total_cost.toFixed(0)}`}
          </span>
        </div>
      </div>
      <div className="dash-itin-layout">
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
        <div className="dash-itin-panel">
          {tab === 'itinerary' && <ItineraryTab data={data} />}
          {tab === 'stay'      && <StayTab data={data} />}
          {tab === 'dining'    && <DiningTab data={data} />}
          {tab === 'options'   && <OptionsTab data={data} />}
        </div>
      </div>
    </div>
  )
}

// ── Main Dashboard ──────────────────────────────────────────────────────────

export default function Dashboard({ userId }: Props) {
  const [trips, setTrips] = useState<TripSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [selected, setSelected] = useState<{ summary: TripSummary; data: RawItinerary } | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)

  useEffect(() => {
    if (!userId) return
    setLoading(true)
    fetch(`/trips?user_id=${encodeURIComponent(userId)}`)
      .then((r) => r.json())
      .then((data) => { setTrips(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [userId])

  async function openTrip(summary: TripSummary) {
    setDetailLoading(true)
    try {
      const r = await fetch(`/trips/${summary.trip_id}`)
      const data: RawItinerary = await r.json()
      setSelected({ summary, data })
    } finally {
      setDetailLoading(false)
    }
  }

  async function handleDelete(e: MouseEvent, tripId: string) {
    e.stopPropagation()
    try {
      await fetch(`/trips/${tripId}`, { method: 'DELETE' })
      setTrips((prev) => prev.filter((t) => t.trip_id !== tripId))
    } catch {
      // silent
    }
  }

  // Show detail view when a trip is selected
  if (selected) {
    return (
      <div className="dashboard">
        <TripDetail summary={selected.summary} data={selected.data} onBack={() => setSelected(null)} />
      </div>
    )
  }

  if (loading || detailLoading) return <div className="dashboard"><div className="dashboard-empty">Loading…</div></div>
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
              <button className="dash-card-expand" onClick={() => openTrip(t)}>
                <div className="dash-card-info">
                  <span className="dash-card-title">{t.title}</span>
                  <span className="dash-card-meta">
                    {t.destination && `${t.destination} · `}
                    {formatDate(t.created_at)}
                    {t.total_cost > 0 && ` · ~$${t.total_cost.toFixed(0)}`}
                  </span>
                </div>
                <span className="dash-card-chevron">›</span>
              </button>
              <button className="dash-card-delete" onClick={(e) => handleDelete(e, t.trip_id)} title="Delete" aria-label="Delete trip">🗑</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
