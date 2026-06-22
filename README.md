# Running the app

Two processes run in parallel: a FastAPI backend (port 8000) and a Vite frontend (port 3000).

## Prerequisites

- Python 3.11+
- Node 18+
- API keys for OpenAI and Google Maps (Brave Search is optional)

---

## 1. Backend

```bash
cd backend
```

Copy the env file and fill in your keys:

```bash
cp .env.example .env
```

```
OPENAI_API_KEY=sk-...
GOOGLE_MAPS_API_KEY=AIza...
BRAVE_API_KEY=          # optional — enables live travel advisories
```

Install dependencies and start the server:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

The API is now running at `http://localhost:8000`.  
Check it with: `curl http://localhost:8000/health`

---

## 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## How the proxy works

Vite proxies `/chat` and `/itinerary` requests from port 3000 → port 8000, so the frontend never needs to know the backend URL directly. Both servers must be running at the same time.

---

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/chat` | Free-form travel chat (used by the UI) |
| POST | `/itinerary` | Direct structured itinerary generation (legacy) |

### `/chat` request body

```json
{
  "message": "What are the best spots in Niagara?",
  "history": []
}
```

### `/chat` response

```json
{
  "text": "Niagara has several great spots...",
  "itinerary": null
}
```

`itinerary` is only populated when the user explicitly asks for a trip plan.
