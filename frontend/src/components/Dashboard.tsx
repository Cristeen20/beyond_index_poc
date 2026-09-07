import { useEffect, useState, type MouseEvent } from 'react'
import type { TripSummary } from '../types'

interface Props { userId: string }

interface Segment {
  start_time: string
  end_time?: string
  title: string
  type: string
  cost: number
  location?: string
  latitude?: number
  longitude?: number
}

interface RawDay {
  day_number: number
  date: string
  day_name?: string
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

const TYPE_ICON: Record<string, string> = {
  travel:    '✈',
  activity:  '📍',
  meal:      '🍽',
  rest:      '😴',
  free_time: '🌿',
  buffer:    '⏱',
}

function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function shortDay(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { weekday: 'short' })
}

function hhmm(t: string): string {
  const s = t?.slice(11, 16)
  if (!s || s === '00:00') return ''
  const [h, m] = s.split(':').map(Number)
  const ampm = h >= 12 ? 'PM' : 'AM'
  return `${h % 12 || 12}:${String(m).padStart(2, '0')} ${ampm}`
}

// ── Day row ─────────────────────────────────────────────────────────────────

function DayRow({ day, open, onToggle }: { day: RawDay; open: boolean; onToggle: () => void }) {
  const activities = (day.segments || []).filter((s) => s.type !== 'buffer')
  const summary = activities.slice(0, 3)

  return (
    <div className={`itin-day-row${open ? ' open' : ''}`}>
      {/* Clickable header */}
      <button className="itin-day-header" onClick={onToggle} aria-expanded={open}>
        <div className="itin-day-timeline">
          <span className="itin-day-dot" />
          <span className="itin-day-line" />
        </div>
        <div className="itin-day-meta">
          <span className="itin-day-num">Day {day.day_number}</span>
          <span className="itin-day-date">{shortDay(day.date)}, {shortDate(day.date)}</span>
        </div>
        <div className="itin-day-body">
          <span className="itin-day-location">{day.location}</span>
          {!open && (
            <div className="itin-day-chips">
              {summary.map((s, i) => (
                <span key={i} className="itin-day-chip">
                  {TYPE_ICON[s.type] || '📍'} {s.title}
                </span>
              ))}
            </div>
          )}
        </div>
        <span className="itin-day-chevron">{open ? '▲' : '▼'}</span>
      </button>

      {/* Expanded segment cards */}
      {open && (
        <div className="itin-segments-list">
          {activities.map((s, i) => (
            <div key={i} className="itin-seg-row">
              <span className="itin-seg-time">{hhmm(s.start_time) || '–'}</span>
              <span className="itin-seg-icon">{TYPE_ICON[s.type] || '📍'}</span>
              <div className="itin-seg-info">
                <span className="itin-seg-name">{s.title}</span>
                {s.location && <span className="itin-seg-loc">{s.location}</span>}
              </div>
              {s.cost > 0 && <span className="itin-seg-cost">${s.cost.toFixed(0)}</span>}
            </div>
          ))}
          {day.accommodation && (
            <div className="itin-seg-row itin-seg-row--stay">
              <span className="itin-seg-time" />
              <span className="itin-seg-icon">🛏</span>
              <div className="itin-seg-info">
                <span className="itin-seg-name">{day.accommodation.name}</span>
                {day.accommodation.address && <span className="itin-seg-loc">{day.accommodation.address}</span>}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── Tab panels ───────────────────────────────────────────────────────────────

function ItineraryTab({ data, tripId }: { data: RawItinerary; tripId: string }) {
  const [openDay, setOpenDay] = useState<number | null>(1)
  const [mapOk, setMapOk] = useState(true)

  return (
    <div className="dash-itin-split">
      {/* Day list */}
      <div className="dash-day-list">
        {data.days.map((d) => (
          <DayRow
            key={d.day_number}
            day={d}
            open={openDay === d.day_number}
            onToggle={() => setOpenDay(openDay === d.day_number ? null : d.day_number)}
          />
        ))}
      </div>
      {/* Map */}
      {mapOk && (
        <div className="dash-map-panel">
          <img
            className="dash-map-img"
            src={`/trips/${tripId}/map`}
            alt="Trip map"
            onError={() => setMapOk(false)}
          />
        </div>
      )}
    </div>
  )
}

function StayTab({ data }: { data: RawItinerary }) {
  const groups: { name: string; address?: string; nights: string[] }[] = []
  for (const d of data.days) {
    if (!d.accommodation) continue
    const ex = groups.find((g) => g.name === d.accommodation!.name)
    if (ex) ex.nights.push(shortDate(d.date))
    else groups.push({ name: d.accommodation.name, address: d.accommodation.address, nights: [shortDate(d.date)] })
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
          <span className="dash-time">{hhmm(m.start_time)}</span>
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

// ── Single trip detail ───────────────────────────────────────────────────────

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
            <button key={t.id} className={`dash-tab-btn${tab === t.id ? ' active' : ''}`} onClick={() => setTab(t.id)}>
              <span className="dash-tab-icon">{t.icon}</span>
              <span className="dash-tab-label">{t.label}</span>
            </button>
          ))}
        </nav>
        <div className="dash-itin-panel">
          {tab === 'itinerary' && <ItineraryTab data={data} tripId={summary.trip_id} />}
          {tab === 'stay'      && <StayTab data={data} />}
          {tab === 'dining'    && <DiningTab data={data} />}
          {tab === 'options'   && <OptionsTab data={data} />}
        </div>
      </div>
    </div>
  )
}

// ── Main Dashboard ───────────────────────────────────────────────────────────

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
      if (selected?.summary.trip_id === tripId) setSelected(null)
    } catch { /* silent */ }
  }

  if (selected) return <div className="dashboard"><TripDetail summary={selected.summary} data={selected.data} onBack={() => setSelected(null)} /></div>
  if (loading || detailLoading) return <div className="dashboard"><div className="dashboard-empty">Loading…</div></div>
  if (!trips.length) return <div className="dashboard"><div className="dashboard-empty">No saved trips yet — plan your first trip in the Chat tab!</div></div>

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
