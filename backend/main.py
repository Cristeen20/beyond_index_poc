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

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langfuse.decorators import langfuse_context

from agent_models import PlanRequest, PlanResponse
from graph import close_checkpointer, init_checkpointer
from travel_orchestrator import plan as plan_handler


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Opens the Postgres pool and creates the checkpoint tables when
    # DATABASE_URL is set; no-op for the local MemorySaver path.
    await init_checkpointer()
    yield
    await close_checkpointer()
    # Drain any pending Langfuse spans before the process exits — the SDK
    # batches in the background and would otherwise drop the tail on a
    # serverless cold-stop. No-op when Langfuse env vars are unset.
    langfuse_context.flush()


app = FastAPI(title="Trip Itinerary Generator", version="0.1.0", lifespan=lifespan)

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
