# Chat improvements

## What changed

The app now handles free-form travel questions, not just itinerary requests.

---

## Backend

### New models (`models.py`)
- `ChatHistoryItem` — a single `{role, content}` turn
- `ChatRequest` — `{message, history[]}` sent from the frontend
- `ChatResponse` — `{text, itinerary?}` returned to the frontend

### New `/chat` endpoint (`main.py`)
Accepts `ChatRequest`, delegates to `orchestrator.chat()`, returns `ChatResponse`.

### New `chat()` function (`orchestrator.py`)
Uses an agentic loop (max 6 iterations) with two tools:

| Tool | When the LLM calls it |
|---|---|
| `fetch_places_data` | Needs real place data — recommendations, spot info, itinerary prep |
| `return_itinerary` | User *explicitly* asks for a day-by-day plan |

**System prompt rules:**
- Recommendations → call `fetch_places_data`, respond in prose
- Specific place questions → answer from knowledge first
- General tips → answer directly, no tools
- Itinerary → ONLY when user explicitly asks; always call `fetch_places_data` first

The loop feeds tool results back into the message history and continues until the LLM produces a plain-text reply or calls `return_itinerary`.

---

## Frontend

### `types.ts`
- Removed `ItineraryRequest` (no longer used in the chat flow)
- `Message.userRequest` replaced by `Message.text` (plain string for both roles)
- Added `ChatHistoryItem`, `ChatRequest`, `ChatResponse`

### `InputForm.tsx`
Replaced the structured form (destination / days / interests / date) with a single `<textarea>`. Enter sends, Shift+Enter adds a new line.

### `ChatMessage.tsx`
- User bubble: renders `message.text`
- Assistant bubble: renders `message.text` (with `white-space: pre-wrap`) and, when present, `<ItineraryCard>`

### `App.tsx`
- Tracks conversation in `messages[]`
- On each submit, builds `history` from settled messages (no loading/error entries) and POSTs to `/chat`
- `isLoading` flag disables input while waiting

### `App.css`
- `textarea` gets `resize: none` and `line-height: 1.5`
- `.bubble-assistant` gets `white-space: pre-wrap` so LLM line breaks render correctly
- Removed unused `.input-extras`, `.btn-expand` styles

---

## Example flows

| User says | LLM does |
|---|---|
| "What are the best spots in Niagara?" | calls `fetch_places_data("Niagara")`, replies in prose |
| "I'm looking for a historic trip in Ontario" | calls `fetch_places_data("Ontario", ["historic"])`, describes places |
| "What would I see at Niagara-on-the-Lake?" | answers from knowledge (no tool call) |
| "Plan me a 3-day historic trip in Ontario" | calls `fetch_places_data`, then `return_itinerary` → renders `ItineraryCard` |
