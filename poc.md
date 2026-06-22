# Trip Itinerary Generator — POC

A minimal proof-of-concept that generates a day-by-day trip itinerary by combining
live place data, web search, and an LLM for synthesis. No vector DB in this version —
that's intentionally deferred (see "What's not in this POC").

## Goal

Given a destination, number of days, and rough interests, return a structured
itinerary: places to visit per day, suggested order/route, and a place to stay.

## Scope (what this POC proves)

- An LLM orchestrator can call external tools (Maps, search) and synthesize a
  usable itinerary.
- Live data (place details, current conditions) beats a static knowledge base
  for this use case.
- The output is structured enough to render in a UI (JSON), not just prose.

## Out of scope for the POC

- User accounts, saved trip history, personalization (this is where a vector DB
  would eventually go).
- Real booking/reservation integration.
- Budget optimization / cost estimation.
- Multi-day route optimization (use a naive "group by area" heuristic instead).

## Architecture

```
User input (destination, days, interests)
        |
        v
Orchestrator (single backend service)
   |-- Google Places API   -> candidate POIs, ratings, hours, coordinates
   |-- Google Directions API -> travel time/distance between POIs
   |-- Web search API      -> current events, closures, weather advisories
        |
        v
LLM call (synthesis step)
   input: structured tool results + user preferences
   output: JSON itinerary
        |
        v
Frontend renders itinerary
```

## Tech stack (POC-level, keep it cheap and fast)

| Layer | Choice | Notes |
|---|---|---|
| Backend | Node.js (Express) or Python (FastAPI) | Either works; FastAPI if you want async tool calls easily |
| LLM | Claude API (`claude-sonnet-4-6`) | Use tool use / function calling for Maps + search |
| Place data | Google Places API + Directions API | Direct REST calls, no DB needed yet |
| Web search | Any search API (Brave Search API, SerpAPI, or Claude's built-in web search tool) | For "is this place currently open / any closures / events" |
| Storage | None, or a single JSON file / SQLite if you want to cache responses | Persistence isn't needed for the POC |
| Frontend | Simple React page or even a CLI script | Just needs to display the JSON itinerary |

No vector DB, no Postgres, no message queue. Keep the POC to "one backend service,
two external APIs, one LLM call."

## API keys needed

- `ANTHROPIC_API_KEY`
- `GOOGLE_MAPS_API_KEY` (enable Places API + Directions API in Google Cloud Console)
- A search API key, if not using Claude's built-in web search tool

## Step-by-step build plan

### 1. Set up the backend skeleton
- One endpoint: `POST /itinerary`
- Request body: `{ destination, days, interests: [], startDate? }`

### 2. Fetch candidate places
- Call Google Places `textSearch` or `nearbySearch` for the destination, filtered
  loosely by `interests` (e.g. "museums in Kyoto", "ramen restaurants in Kyoto").
- Pull 15–30 candidates: name, address, lat/lng, rating, opening hours, place type.

### 3. Check for live conditions
- Run a web search per destination (or per day) for things like:
  `"<destination> travel advisory OR closures OR events <month> <year>"`
- Keep this lightweight — one or two searches, not per-place.

### 4. Synthesize with the LLM
- Single prompt to Claude with:
  - User's request (days, interests)
  - The candidate places list (as JSON)
  - The web search summary
- Ask for a **strict JSON output** matching a schema like:

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
  "advisories": ["Some seasonal note from web search, if relevant"]
}
```

- Group stops by geographic proximity (you can do this naively by clustering
  lat/lng, or just let the LLM order them using the place coordinates you give it).

### 5. Render the output
- Frontend just needs to map over `days[].stops[]` — no special logic needed yet.
- Use `places_map_display_v0`-style mapping later if you build a map view; for the
  POC, a simple list view is enough.

## Example orchestrator pseudocode

```python
def generate_itinerary(destination, days, interests):
    places = call_google_places(destination, interests)       # tool call 1
    advisories = call_web_search(f"{destination} travel advisory")  # tool call 2

    prompt = build_prompt(destination, days, interests, places, advisories)
    response = call_claude(prompt, response_format="json")

    return response
```

## Evaluation checklist (how you'll know the POC works)

- [ ] Returns a valid itinerary for at least 3 different destinations
- [ ] Itinerary respects the requested number of days
- [ ] Places are grouped sensibly by location, not randomly scattered across the city
- [ ] Web search catches at least one real, current closure/event when one exists
- [ ] JSON output validates against your schema every time (test malformed-output handling)

## What's not in this POC (next steps after validating)

- **Personalization / vector DB**: once you have users with saved trip history,
  add a vector DB (e.g. pgvector, Pinecone) to match new requests against past
  liked itineraries.
- **Structured filtering**: budget constraints, date-based pricing — push to a
  real SQL layer once you have inventory data (hotel prices, etc.).
- **Route optimization**: replace naive grouping with an actual TSP-style solver
  if itineraries start feeling inefficient.
- **Caching layer**: cache Places API responses to cut cost once you're past POC.