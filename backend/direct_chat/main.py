"""FastAPI app for the Direct Chat feature (features/direct_chat.md).

Self-contained: run with `uvicorn direct_chat.main:app --port 8001 --reload`.
Does not import or touch the existing travel itinerary code.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()  # load .env before anything imports os.environ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
for _name in ("chat_orchestrator", "direct_chat.nodes"):
    logging.getLogger(_name).setLevel(logging.INFO)

from direct_chat.chat_orchestrator import chat as chat_handler
from direct_chat.models import ChatRequest, ChatResponse


app = FastAPI(title="Direct Chat", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest) -> ChatResponse:
    """LangGraph-backed LLM chat with server-side conversation persistence."""
    try:
        return await chat_handler(req.message, req.history, req.conversation_id)
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=f"Missing env var: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
