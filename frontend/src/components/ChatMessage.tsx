import { useState } from 'react'
import type { Message, OptionAction } from '../types'
import ItineraryCard from './ItineraryCard'
import OptionsCard from './OptionsCard'

interface Props {
  message: Message
  onOptionAction?: (action: OptionAction) => void
  optionActionDisabled?: boolean
  onSaveToTrips?: () => Promise<void>
}

export default function ChatMessage({ message, onOptionAction, optionActionDisabled, onSaveToTrips }: Props) {
  const [saved, setSaved] = useState(false)
  const [saving, setSaving] = useState(false)

  if (message.role === 'user') {
    return (
      <div className="message message-user">
        <div className="bubble bubble-user">{message.text}</div>
      </div>
    )
  }

  if (message.isLoading) {
    return (
      <div className="message message-assistant">
        <div className="bubble bubble-assistant loading">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </div>
      </div>
    )
  }

  if (message.errorText) {
    return (
      <div className="message message-assistant">
        <div className="bubble bubble-error">
          <strong>Error</strong> — {message.errorText}
        </div>
      </div>
    )
  }

  // When there's an options card AND the text is just the card title, the
  // card already displays it as a header — skip the duplicate bubble.
  const suppressBubble =
    !!message.optionsPayload && message.text === message.optionsPayload.title

  async function handleSave() {
    if (!onSaveToTrips || saved || saving) return
    setSaving(true)
    await onSaveToTrips()
    setSaved(true)
    setSaving(false)
  }

  return (
    <div className="message message-assistant">
      {message.text && !suppressBubble && (
        <div className="bubble bubble-assistant bubble-with-save">
          {onSaveToTrips && (
            <button
              className={`btn-save-trip-inline${saved ? ' saved' : ''}`}
              onClick={handleSave}
              disabled={saving || saved}
              title={saved ? 'Saved to My Trips' : 'Save to My Trips'}
            >
              {saved ? '✓ Saved' : saving ? '…' : '💾 Save'}
            </button>
          )}
          {message.text}
        </div>
      )}
      {message.itinerary && <ItineraryCard itinerary={message.itinerary} />}
      {message.optionsPayload && onOptionAction && (
        <OptionsCard
          payload={message.optionsPayload}
          onAction={onOptionAction}
          disabled={optionActionDisabled}
        />
      )}
    </div>
  )
}
