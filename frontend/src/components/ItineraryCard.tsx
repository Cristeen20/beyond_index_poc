import type { Itinerary } from '../types'

interface Props {
  itinerary: Itinerary
}

export default function ItineraryCard({ itinerary }: Props) {
  return (
    <div className="itinerary-card">
      <div className="itinerary-header">
        <span className="itinerary-icon">🗺</span>
        <h2>{itinerary.destination}</h2>
        <span className="day-count">{itinerary.days.length} days</span>
      </div>

      {itinerary.advisories.length > 0 && (
        <div className="advisories">
          <h4>Advisories</h4>
          <ul>
            {itinerary.advisories.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="days-list">
        {itinerary.days.map((day) => (
          <div key={day.day} className="day-block">
            <div className="day-header">
              <span className="day-label">Day {day.day}</span>
              <span className="day-theme">{day.theme}</span>
            </div>

            <ol className="stops-list">
              {day.stops.map((stop, i) => (
                <li key={i} className="stop-item">
                  <div className="stop-time">{stop.time}</div>
                  <div className="stop-body">
                    <div className="stop-name">{stop.name}</div>
                    <div className="stop-meta">
                      {stop.duration_minutes} min
                      {stop.notes && (
                        <span className="stop-notes"> · {stop.notes}</span>
                      )}
                    </div>
                  </div>
                </li>
              ))}
            </ol>

            {day.lodging && (
              <div className="lodging">
                <span className="lodging-icon">🏨</span>
                {day.lodging}
              </div>
            )}
          </div>
        ))}
      </div>

      <button
        className="btn-copy"
        onClick={() => navigator.clipboard.writeText(JSON.stringify(itinerary, null, 2))}
        title="Copy raw JSON"
      >
        Copy JSON
      </button>
    </div>
  )
}
