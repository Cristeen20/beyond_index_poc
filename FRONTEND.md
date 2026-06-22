# Frontend UI — Trip Itinerary Generator

A TypeScript/React chatbot-style UI for testing the Python FastAPI backend defined in `poc.md`.

## Stack

- **Vite** + **React 18** + **TypeScript** — no extra UI libraries
- Dev server proxies `/itinerary` → `http://localhost:8000` (the FastAPI backend)
- DM Sans font (matches RouteThis brand)

## Setup

```bash
cd frontend
npm install
npm run dev       # starts on http://localhost:3000
```

The backend must be running on port 8000 before you generate itineraries.

## How it works

1. Type a destination in the input bar (e.g. `Tokyo, Japan`)
2. Click **▼** to expand optional fields: number of days, interests (comma-separated), start date
3. Hit **Generate ↗** — the UI posts to `POST /itinerary` and shows a loading indicator
4. The response renders as a structured itinerary card in the chat window

## API contract

**Request** — `POST /itinerary`

```json
{
  "destination": "Kyoto, Japan",
  "days": 3,
  "interests": ["temples", "food"],
  "startDate": "2026-09-10"
}
```

`startDate` is optional.

**Response** — the JSON schema the backend must return:

```json
{
  "destination": "Kyoto, Japan",
  "days": [
    {
      "day": 1,
      "theme": "Eastern Kyoto temples",
      "stops": [
        {
          "name": "Kiyomizu-dera",
          "time": "9:00 AM",
          "duration_minutes": 90,
          "notes": "Arrive early to avoid crowds"
        }
      ],
      "lodging": "Hotel Kanra Kyoto"
    }
  ],
  "advisories": ["Seasonal note from web search, if relevant"]
}
```

## File structure

```
frontend/
  index.html
  package.json
  tsconfig.json
  vite.config.ts          ← proxy config lives here
  src/
    main.tsx
    App.tsx               ← chat state, fetch logic
    App.css               ← all styles
    index.css             ← reset + CSS variables
    types.ts              ← shared TypeScript types
    components/
      InputForm.tsx       ← destination + expanded options form
      ChatMessage.tsx     ← routes user vs assistant messages
      ItineraryCard.tsx   ← renders the structured itinerary
```

## Changing the backend URL

Edit `vite.config.ts`:

```ts
proxy: {
  '/itinerary': {
    target: 'http://localhost:8000',  // ← change this
    changeOrigin: true,
  },
},
```

## Testing with a mock response

If the backend isn't ready yet, paste this into your browser console to inject a fake itinerary and verify the UI renders correctly:

```js
// In Chrome DevTools console — intercept the next fetch
const orig = window.fetch
window.fetch = (url, ...args) => {
  if (url === '/itinerary') {
    return Promise.resolve(new Response(JSON.stringify({
      destination: "Kyoto, Japan",
      days: [
        {
          day: 1,
          theme: "Eastern temples",
          stops: [
            { name: "Kiyomizu-dera", time: "9:00 AM", duration_minutes: 90, notes: "Go early" },
            { name: "Fushimi Inari", time: "11:30 AM", duration_minutes: 120, notes: "Hat recommended" }
          ],
          lodging: "Hotel Kanra Kyoto"
        }
      ],
      advisories: ["Check for seasonal closures before visiting shrines"]
    }), { headers: { 'Content-Type': 'application/json' } }))
  }
  return orig(url, ...args)
}
```

## What the UI does NOT do

- No auth, no saved history
- No map view (just a list — map view is a post-POC step per `poc.md`)
- No streaming — waits for the full response before rendering
