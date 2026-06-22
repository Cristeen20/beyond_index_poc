import type { Message } from '../types'
import ItineraryCard from './ItineraryCard'

interface Props {
  message: Message
}

export default function ChatMessage({ message }: Props) {
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

  return (
    <div className="message message-assistant">
      {message.text && (
        <div className="bubble bubble-assistant">{message.text}</div>
      )}
      {message.itinerary && <ItineraryCard itinerary={message.itinerary} />}
    </div>
  )
}
