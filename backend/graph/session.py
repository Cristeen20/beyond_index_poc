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

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from agent_models import PlanningState

logger = logging.getLogger("graph.session")


# Single process-wide checkpointer — MemorySaver holds state in-process
# and does not survive server restart. Sessions are cheap to recreate.
CHECKPOINTER = MemorySaver()


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
