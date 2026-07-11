from dotenv import load_dotenv

load_dotenv()  # load .env before anything else imports os.environ

from fastapi import FastAPI, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from xml.sax.saxutils import escape as xml_escape

from models import (
    ChatHistoryItem,
    ChatRequest,
    ChatResponse,
    Itinerary,
    ItineraryRequest,
)
from orchestrator import generate_itinerary, chat as chat_handler

from agent_models import PlanRequest, PlanResponse
from travel_orchestrator import plan as plan_handler


app = FastAPI(title="Trip Itinerary Generator", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    try:
        return await chat_handler(req)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Missing env var: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/itinerary", response_model=Itinerary)
async def create_itinerary(req: ItineraryRequest) -> Itinerary:
    try:
        return await generate_itinerary(req)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Missing env var: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/plan", response_model=PlanResponse)
async def plan_endpoint(req: PlanRequest) -> PlanResponse:
    """Entry point for the Itinerary Agent architecture (see itenary_agent.md).

    Runs the Intake Router → DIRECT (targeted lookup) or FULL (full plan).
    """
    try:
        return await plan_handler(req)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Missing env var: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# In-memory chat history per WhatsApp sender. Resets on server restart —
# fine for the POC; swap for Redis/DB if this graduates.
_whatsapp_history: dict[str, list[ChatHistoryItem]] = {}
_MAX_HISTORY = 20
_WHATSAPP_CHAR_LIMIT = 1500  # Twilio caps a single WhatsApp message at 1600


def _format_itinerary(it: Itinerary) -> str:
    lines = [f"*{it.destination}* — {len(it.days)}-day itinerary", ""]
    for day in it.days:
        lines.append(f"*Day {day.day} — {day.theme}*")
        for stop in day.stops:
            lines.append(f"  • {stop.time} {stop.name} ({stop.duration_minutes}m)")
            if stop.notes:
                lines.append(f"    _{stop.notes}_")
        if day.lodging:
            lines.append(f"  🛏 {day.lodging}")
        lines.append("")
    if it.advisories:
        lines.append("*Advisories:*")
        lines.extend(f"• {a}" for a in it.advisories)
    return "\n".join(lines).strip()


def _twiml(body: str) -> Response:
    if len(body) > _WHATSAPP_CHAR_LIMIT:
        body = body[: _WHATSAPP_CHAR_LIMIT - 1] + "…"
    xml = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Message>{xml_escape(body)}</Message></Response>"
    )
    return Response(content=xml, media_type="application/xml")


@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),
    Body: str = Form(...),
) -> Response:
    history = _whatsapp_history.setdefault(From, [])
    try:
        result = await chat_handler(ChatRequest(message=Body, history=history))
    except Exception as exc:  # noqa: BLE001
        return _twiml(f"Sorry — something went wrong: {exc}")

    reply = result.text
    if result.itinerary is not None:
        reply = f"{reply}\n\n{_format_itinerary(result.itinerary)}"

    history.append(ChatHistoryItem(role="user", content=Body))
    history.append(ChatHistoryItem(role="assistant", content=result.text))
    if len(history) > _MAX_HISTORY:
        del history[: len(history) - _MAX_HISTORY]

    return _twiml(reply)
