import { useState } from 'react'
import type { OptionAction, OptionsPayload } from '../types'

interface Props {
  payload: OptionsPayload
  onAction: (action: OptionAction) => void
  disabled?: boolean
}

// Renders the structured pre-planning cards emitted by the backend:
//   confirm_basics / scope / day_by_day → button-only cards
//   places / stays                       → list of items with checkbox/radio
//                                          plus "More options" if paged
export default function OptionsCard({ payload, onAction, disabled }: Props) {
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [q, setQ] = useState('')
  const [customDays, setCustomDays] = useState('')

  const isItemCard = payload.kind === 'places' || payload.kind === 'stays'
  const isNumDaysCard = payload.kind === 'num_days'
  const isNumTravelersCard = payload.kind === 'num_travelers'
  const isBudgetCard = payload.kind === 'budget'
  const isNumericCard = isNumDaysCard || isNumTravelersCard || isBudgetCard

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev)
      if (payload.select === 'single') {
        next.clear()
        next.add(id)
      } else if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  function submitSelect() {
    if (selected.size === 0) return
    onAction({ action: 'select', ids: Array.from(selected) })
  }

  function submitAction(id: string) {
    // Map the clicked button to a semantic action. Non-'more' buttons
    // are single-choice picks — the parser inspects ids[0] to know which
    // button was clicked. 'correct' on the confirm_basics card is its
    // own action so the parser can re-hydrate from the user's next
    // free-text reply. For num_days, the id IS the number ("3") and
    // the parser reads ids[0].
    if (id === 'more') return onAction({ action: 'more' })
    if (id === 'correct') return onAction({ action: 'correct', ids: [id] })
    if (isNumericCard) return onAction({ action: 'select', ids: [id] })
    onAction({ action: 'confirm', ids: [id] })
  }

  function submitCustomDays() {
    const n = parseInt(customDays.trim(), 10)
    if (!Number.isFinite(n) || n < 1) return
    onAction({ action: 'select', ids: [String(n)] })
    setCustomDays('')
  }

  function submitQuestion() {
    const t = q.trim()
    if (!t) return
    onAction({ action: 'question', text: t })
    setQ('')
  }

  return (
    <div className="options-card">
      <div className="options-header">
        <h3>{payload.title}</h3>
        {payload.description && (
          <p className="options-description">{payload.description}</p>
        )}
      </div>

      {isItemCard && (
        <>
          <ul className="options-items">
            {payload.items.map((it) => {
              const meta = it.meta || {}
              return (
                <li key={it.id} className="option-item">
                  <label className="option-item-label">
                    <input
                      type={payload.select === 'single' ? 'radio' : 'checkbox'}
                      name="option"
                      checked={selected.has(it.id)}
                      onChange={() => toggle(it.id)}
                      disabled={disabled}
                    />
                    <span className="option-rank">#{it.rank}</span>
                    <span className="option-body">
                      <span className="option-name">{it.name}</span>
                      {it.rationale && (
                        <span className="option-rationale">{it.rationale}</span>
                      )}
                      {formatItemMeta(payload.kind, meta) && (
                        <span className="option-meta">
                          {formatItemMeta(payload.kind, meta)}
                        </span>
                      )}
                    </span>
                  </label>
                </li>
              )
            })}
            {payload.items.length === 0 && (
              <li className="option-empty">No options to show.</li>
            )}
          </ul>

          <div className="options-actions">
            <button
              type="button"
              className="btn-primary"
              onClick={submitSelect}
              disabled={disabled || selected.size === 0}
            >
              Use {selected.size || ''} selected
            </button>
            {payload.actions.map((a) => (
              <button
                key={a.id}
                type="button"
                className="btn-secondary"
                onClick={() => submitAction(a.id)}
                disabled={disabled}
              >
                {a.label}
              </button>
            ))}
          </div>

          <div className="options-question">
            <input
              type="text"
              placeholder="Or ask a question about these…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submitQuestion()
              }}
              disabled={disabled}
            />
            <button
              type="button"
              className="btn-secondary"
              onClick={submitQuestion}
              disabled={disabled || !q.trim()}
            >
              Ask
            </button>
          </div>
        </>
      )}

      {!isItemCard && !isNumericCard && (
        <div className="options-actions options-actions-buttons">
          {payload.actions.map((a) => (
            <button
              key={a.id}
              type="button"
              className="btn-primary"
              onClick={() => submitAction(a.id)}
              disabled={disabled}
            >
              {a.label}
            </button>
          ))}
        </div>
      )}

      {isNumericCard && (
        <>
          <div className="options-actions options-actions-buttons">
            {payload.actions.map((a) => (
              <button
                key={a.id}
                type="button"
                className="btn-primary"
                onClick={() => submitAction(a.id)}
                disabled={disabled}
              >
                {a.label}
              </button>
            ))}
          </div>
          <div className="options-question">
            <input
              type="number"
              min={1}
              placeholder={
                isBudgetCard
                  ? 'Or type a total budget (e.g. 2000)…'
                  : isNumTravelersCard
                  ? 'Or type a number of travelers…'
                  : 'Or type a number of days…'
              }
              value={customDays}
              onChange={(e) => setCustomDays(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submitCustomDays()
              }}
              disabled={disabled}
            />
            <button
              type="button"
              className="btn-secondary"
              onClick={submitCustomDays}
              disabled={disabled || !customDays.trim()}
            >
              Use
            </button>
          </div>
        </>
      )}
    </div>
  )
}

function formatItemMeta(kind: string, meta: Record<string, unknown>): string {
  const parts: string[] = []
  if (kind === 'places') {
    if (meta.type) parts.push(String(meta.type))
    if (typeof meta.cost === 'number' && meta.cost > 0) {
      parts.push(`~$${(meta.cost as number).toFixed(0)}`)
    }
    if (meta.address) parts.push(String(meta.address))
  } else if (kind === 'stays') {
    if (typeof meta.star_rating === 'number') parts.push(`${meta.star_rating}★`)
    if (typeof meta.price_per_night === 'number') {
      parts.push(`~$${(meta.price_per_night as number).toFixed(0)}/night`)
    }
    if (meta.address) parts.push(String(meta.address))
  }
  return parts.join(' · ')
}
