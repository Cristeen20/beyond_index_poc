import type { Message } from '../types'
import ItineraryCard from './ItineraryCard'

interface Props {
  message: Message
}

export default function ChatMessage({ message }: Props) {
  if (message.role === 'user' && message.userRequest) {
    const { destination, days, interests, startDate } = message.userRequest
    return (
      <div className="message message-user">
        <div className="bubble bubble-user">
          <strong>{destination}</strong> — {days} day{days !== 1 ? 's' : ''}
          {interests.length > 0 && (
            <span className="interests-tag">
              {' · '}
              {interests.join(', ')}
            </span>
          )}
          {startDate && <span className="interests-tag"> · starts {startDate}</span>}
        </div>
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

  if (message.itinerary) {
    return (
      <div className="message message-assistant">
        <ItineraryCard itinerary={message.itinerary} />
      </div>
    )
  }

  return null
}
