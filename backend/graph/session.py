"""Session/loop-back plumbing for the top-level travel graph.

The travel graph is compiled with a `MemorySaver` checkpointer so that a
conversation persists across `/plan` calls: `trip_request`, `itinerary`,
and `history` accumulated on turn N are visible to `intent_decision` on
turn N+1.

Rather than ending at each terminal branch and restarting from START on
the next request, every terminal branch (`answer_conversational`,
`merge_direct`, `itinerary_planning`, revise subgraph) edges into
`wait_for_next_message`. That node calls `interrupt()` to pause the run.
The next call resumes with the new user message and edges back to
`intent_decision`.
"""

from __future__ import annotations

import logging
import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from agent_models import PlanningState

logger = logging.getLogger("graph.session")


# Where the checkpointer stores threads depends on the deployment target:
#
#   DATABASE_URL set    → AsyncPostgresSaver. Required on Vercel, where each
#                         /plan call may land on a different (or recycled)
#                         function instance, so an in-process saver would
#                         drop the session mid-conversation.
#   DATABASE_URL unset  → MemorySaver, matching the original local-dev
#                         behaviour: one long-lived uvicorn process, state
#                         held in-process, gone on restart.
#
# Vercel's Neon integration injects DATABASE_URL; POSTGRES_URL is accepted
# as a fallback since some Marketplace Postgres providers use that name.
_DB_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or ""

# Held at module scope so init_checkpointer() / close_checkpointer() can
# open and close it around the FastAPI lifespan. None under MemorySaver.
_POOL = None


def _build_checkpointer():
    global _POOL

    if not _DB_URL:
        logger.warning(
            "DATABASE_URL unset — using in-process MemorySaver. Sessions will "
            "not survive a restart and must not be relied on in serverless."
        )
        return MemorySaver()

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool

    _POOL = AsyncConnectionPool(
        conninfo=_DB_URL,
        min_size=0,      # cold starts shouldn't pay for idle connections
        max_size=4,      # one function instance handles few concurrent turns
        open=False,      # opened in init_checkpointer(), not at import time
        kwargs={
            # AsyncPostgresSaver issues its own transactions.
            "autocommit": True,
            # Neon's pooled endpoint is pgbouncer in transaction mode, which
            # cannot carry server-side prepared statements between requests.
            "prepare_threshold": None,
            "row_factory": dict_row,
        },
    )
    logger.info("Using AsyncPostgresSaver checkpointer")
    return AsyncPostgresSaver(_POOL)


_CHECKPOINTER = None


def get_checkpointer():
    """Return the process-wide checkpointer, building it on first call.

    Deliberately lazy: `AsyncPostgresSaver.__init__` calls
    `asyncio.get_running_loop()`, so it can only be constructed from inside
    a running event loop — which rules out building it at import time. The
    first caller is `init_checkpointer()` from main.py's lifespan; the
    graph compile in travel_orchestrator then reuses the same instance.
    """
    global _CHECKPOINTER
    if _CHECKPOINTER is None:
        _CHECKPOINTER = _build_checkpointer()
    return _CHECKPOINTER


async def init_checkpointer() -> None:
    """Build the checkpointer, open its pool, and ensure its tables exist.

    Idempotent. Under MemorySaver everything after the build is a no-op.
    """
    checkpointer = get_checkpointer()
    if _POOL is None:
        return
    await _POOL.open(wait=True)
    await checkpointer.setup()  # CREATE TABLE IF NOT EXISTS + migrations
    logger.info("Checkpointer ready (postgres)")


async def close_checkpointer() -> None:
    """Release the pool on shutdown. No-op under MemorySaver."""
    if _POOL is None:
        return
    await _POOL.close()


def get_pool():
    """Return the Postgres connection pool, or None under MemorySaver."""
    return _POOL


def wait_for_next_message(state: PlanningState) -> dict:
    """Pause the graph until the next /plan call resumes with a new message.

    Any state written to the current turn (intent, options, itinerary,
    response_message) is persisted to the checkpointer under this thread.
    On resume, the payload returned by interrupt() is the new user message
    which we write back onto `incoming_message` before the graph flows on
    to `intent_decision`.
    """
    logger.info("wait_for_next_message → pausing, awaiting next turn")
    resume_payload = interrupt("awaiting_next_message")
    logger.info(
        "wait_for_next_message → resumed with payload=%r", resume_payload
    )
    # Resume payload is either a plain string (free-text reply) or a dict
    # like {"message": "...", "option_action": {...}} sent by the
    # orchestrator when the frontend used a button/select on an OptionsCard.
    if isinstance(resume_payload, dict):
        next_message = resume_payload.get("message") or ""
        option_action = resume_payload.get("option_action")
    else:
        next_message = resume_payload or ""
        option_action = None

    # Reset per-turn OUTPUT scratch. We deliberately preserve `intent`,
    # `missing_slots`, `followup_question`, `pending_stage`, and
    # `options_payload` so the pre-planning router can see whether we
    # paused mid-stage and route accordingly. Each parse node clears the
    # options_payload / pending_stage once it consumes them.
    return {
        "incoming_message": next_message,
        "option_action": option_action,
        "response_message": "",
        "direct_result": None,
        "error_notes": [],
        "conflict_notes": [],
        "repair_attempts": 0,
    }
