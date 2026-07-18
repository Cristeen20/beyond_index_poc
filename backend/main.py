import logging

from dotenv import load_dotenv

load_dotenv()  # load .env before anything else imports os.environ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # uvicorn installs its own root handlers; force= wins.
)
for _name in (
    "intake_router", "travel_orchestrator", "itinerary_agent", "sub_agents",
):
    logging.getLogger(_name).setLevel(logging.INFO)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ChatRequest, ChatResponse
from agent_models import PlanRequest, PlanResponse, ReviseRequest, ReviseResponse
from travel_orchestrator import plan as plan_handler, revise as revise_handler


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


def _format_direct_results(results: list[dict]) -> str:
    """Turn PlanResponse.direct_result into a human-readable snippet."""
    if not results:
        return "I couldn't find matches — try broadening the search."
    lines: list[str] = []
    by_agent: dict[str, list[dict]] = {}
    for r in results:
        by_agent.setdefault(r.get("agent", "result"), []).append(r)

    labels = {
        "hotel": "Hotels",
        "restaurant": "Restaurants",
        "event": "Things to do",
        "route": "Transport",
    }
    for agent, items in by_agent.items():
        lines.append(f"**{labels.get(agent, agent.title())}**")
        for it in items[:5]:
            name = it.get("name") or it.get("route_id") or "(unnamed)"
            extras: list[str] = []
            if "star_rating" in it:
                extras.append(f"{it['star_rating']}★")
            if "rating" in it and it["rating"]:
                extras.append(f"{it['rating']:.1f}")
            if "price_per_night" in it:
                extras.append(f"~${it['price_per_night']:.0f}/night")
            if "avg_cost_per_person" in it:
                extras.append(f"~${it['avg_cost_per_person']:.0f}/person")
            if "cuisine" in it and it["cuisine"]:
                extras.append(it["cuisine"])
            if "mode" in it:
                extras.append(f"{it['mode']} ~${it.get('total_cost', 0):.0f}")
            tail = f" — {' · '.join(extras)}" if extras else ""
            lines.append(f"  • {name}{tail}")
        lines.append("")
    return "\n".join(lines).strip()


def _summarise_new_itinerary(it) -> str:
    """One-paragraph text summary of an agent_models.Itinerary."""
    lines = [f"**{it.title}** — {len(it.days)} day(s), ~${it.total_cost:.0f} total.", ""]
    for d in it.days:
        hotel = f" · 🛏 {d.accommodation.name}" if d.accommodation else ""
        lines.append(f"*Day {d.day_number} ({d.date}) — {d.location}*{hotel}")
        for s in d.segments[:6]:
            lines.append(f"  • {s.start_time.strftime('%H:%M')} {s.title}")
        lines.append("")
    return "\n".join(lines).strip()


async def _plan_to_chat_response(req: ChatRequest) -> ChatResponse:
    """Route /chat traffic through the LangGraph flow and adapt PlanResponse
    back to the ChatResponse shape the frontend understands."""
    plan_req = PlanRequest(
        message=req.message,
        history=[m.model_dump() for m in req.history],
    )
    result = await plan_handler(plan_req)

    if result.route == "conversational":
        return ChatResponse(text=result.message)

    if result.followup_question:
        return ChatResponse(text=result.followup_question)

    if result.route == "direct":
        return ChatResponse(text=_format_direct_results(result.direct_result or []))

    # route == "full"
    if result.itinerary is not None:
        return ChatResponse(text=_summarise_new_itinerary(result.itinerary))
    return ChatResponse(text=result.message or "Sorry, I couldn't put a plan together.")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """UI entry point — flows through the LangGraph Intake Router + Itinerary
    sub-graph and is adapted back to ChatResponse for the frontend."""
    try:
        return await _plan_to_chat_response(req)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Missing env var: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/plan", response_model=PlanResponse)
async def plan_endpoint(req: PlanRequest) -> PlanResponse:
    """Full LangGraph plan flow: Intake Router → DIRECT lookup or FULL itinerary."""
    try:
        return await plan_handler(req)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Missing env var: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/revise", response_model=ReviseResponse)
async def revise_endpoint(req: ReviseRequest) -> ReviseResponse:
    """Revision StateGraph: re-plan an existing itinerary from feedback."""
    try:
        return await revise_handler(req)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Missing env var: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
