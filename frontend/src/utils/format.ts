// Formatters for backend PlanResponse. Ports of what used to live in
// backend/main.py (_format_direct_results, _summarise_new_itinerary) so the
// backend can stay a pure JSON API and the UI owns its own presentation.

import type { DirectResultItem, PlanResponse } from '../types'

const AGENT_LABELS: Record<string, string> = {
  hotel: 'Hotels',
  restaurant: 'Restaurants',
  event: 'Things to do',
  route: 'Transport',
}

function num(v: unknown, digits = 0): string | null {
  if (typeof v !== 'number' || Number.isNaN(v)) return null
  return v.toFixed(digits)
}

function formatDirectResults(results: DirectResultItem[]): string {
  if (!results?.length) return "I couldn't find matches — try broadening the search."

  const byAgent: Record<string, DirectResultItem[]> = {}
  for (const r of results) {
    const key = (r.agent as string) || 'result'
    ;(byAgent[key] ||= []).push(r)
  }

  const lines: string[] = []
  for (const [agent, items] of Object.entries(byAgent)) {
    lines.push(`**${AGENT_LABELS[agent] || agent}**`)
    for (const it of items.slice(0, 5)) {
      const name = (it.name as string) || (it.route_id as string) || '(unnamed)'
      const extras: string[] = []
      if (it.star_rating != null) extras.push(`${it.star_rating}★`)
      const rating = num(it.rating, 1)
      if (rating) extras.push(rating)
      const perNight = num(it.price_per_night, 0)
      if (perNight) extras.push(`~$${perNight}/night`)
      const perPerson = num(it.avg_cost_per_person, 0)
      if (perPerson) extras.push(`~$${perPerson}/person`)
      if (it.cuisine) extras.push(String(it.cuisine))
      if (it.mode) {
        const total = num(it.total_cost, 0) || '0'
        extras.push(`${it.mode} ~$${total}`)
      }
      const tail = extras.length ? ` — ${extras.join(' · ')}` : ''
      lines.push(`  • ${name}${tail}`)
    }
    lines.push('')
  }
  return lines.join('\n').trim()
}

interface RawDay {
  day_number: number
  date: string
  location: string
  accommodation?: { name: string } | null
  segments?: Array<{ start_time: string; title: string }>
}

interface RawItinerary {
  title: string
  days: RawDay[]
  total_cost: number
}

function formatItinerary(it: RawItinerary): string {
  const lines: string[] = [
    `**${it.title}** — ${it.days.length} day(s), ~$${it.total_cost.toFixed(0)} total.`,
    '',
  ]
  for (const d of it.days) {
    const hotel = d.accommodation ? ` · 🛏 ${d.accommodation.name}` : ''
    lines.push(`*Day ${d.day_number} (${d.date}) — ${d.location}*${hotel}`)
    for (const s of (d.segments || []).slice(0, 6)) {
      const hhmm = s.start_time.slice(11, 16)
      lines.push(`  • ${hhmm} ${s.title}`)
    }
    lines.push('')
  }
  return lines.join('\n').trim()
}

export function formatPlanResponse(r: PlanResponse): string {
  if (r.followup_question) return r.followup_question
  if (r.route === 'conversational') return r.message
  if (r.route === 'direct') {
    // answer_mode='answer' path — the backend's answer_from_places node
    // produced a natural-language sentence. Prefer it over the bullet list.
    if (r.intent?.answer_mode === 'answer' && r.message) return r.message
    return formatDirectResults(r.direct_result || [])
  }
  // full or revise — itinerary may still be missing if load failed
  if (r.itinerary) return formatItinerary(r.itinerary as RawItinerary)
  return r.message || "Sorry, I couldn't put a plan together."
}
