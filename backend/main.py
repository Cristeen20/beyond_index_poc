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


@app.post("/plan", response_model=PlanResponse)
async def plan_endpoint(req: PlanRequest) -> PlanResponse:
    """Session-checkpointed LangGraph plan flow. First call for a session_id
    runs from START; subsequent calls resume at wait_for_next_message and
    re-enter intent_decision with the new message + persisted trip/itinerary
    context. See itinerary_langgraph_flow.md."""
    try:
        return await plan_handler(req)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Missing env var: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/revise", response_model=ReviseResponse)
async def revise_endpoint(req: ReviseRequest) -> ReviseResponse:
    """Standalone revision subgraph — for callers that already have an
    explicit ReviseRequest (frontend's approve/revise UI). Not
    session-checkpointed; the request payload carries everything."""
    try:
        return await revise_handler(req)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Missing env var: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
