import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { useEffect, useRef } from 'react'

export interface MapMarker {
  lat: number
  lng: number
  title: string
  day: number
  type: 'place' | 'stay'
}

const DAY_COLORS = ['#E53935', '#1E88E5', '#43A047', '#8E24AA', '#F4511E', '#FFB300']

function makeDivIcon(color: string, icon: string, size = 28) {
  return L.divIcon({
    html: `<div style="
      width:${size}px;height:${size}px;border-radius:50%;
      background:${color};color:#fff;
      display:flex;align-items:center;justify-content:center;
      font-size:${size * 0.5}px;
      box-shadow:0 2px 6px rgba(0,0,0,.35);
      border:2px solid #fff;
    ">${icon}</div>`,
    className: '',
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -(size / 2 + 4)],
  })
}

interface Props {
  markers: MapMarker[]
}

export default function TripMap({ markers }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<L.Map | null>(null)

  useEffect(() => {
    if (!containerRef.current || !markers.length) return

    const map = L.map(containerRef.current, {
      zoomControl: true,
      scrollWheelZoom: true,
    })

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© <a href="https://openstreetmap.org">OpenStreetMap</a>',
      maxZoom: 18,
    }).addTo(map)

    const bounds = L.latLngBounds([])

    markers.forEach((m) => {
      const color = DAY_COLORS[(m.day - 1) % DAY_COLORS.length]
      const icon = makeDivIcon(color, m.type === 'stay' ? '🛏' : '📍')
      L.marker([m.lat, m.lng], { icon })
        .addTo(map)
        .bindPopup(`<strong>${m.title}</strong><br><small>Day ${m.day}</small>`)
      bounds.extend([m.lat, m.lng])
    })

    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [24, 24], maxZoom: 14 })
    }

    mapRef.current = map
    return () => {
      map.remove()
      mapRef.current = null
    }
  }, [markers])

  if (!markers.length) {
    return (
      <div className="dash-map-panel dash-map-empty">
        <span>No location data</span>
      </div>
    )
  }

  return <div ref={containerRef} className="dash-map-panel" />
}
