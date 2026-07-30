"""Chat orchestrator — invokes the LangGraph chat flow.

Thin entry point that:
  1. Detects new vs. existing conversation (via checkpointer snapshot).
  2. Seeds state for new conversations (system prompt + optional client
     history + user message).
  3. Sends just the user message for resumes (checkpointer has the rest).
  4. Extracts the assistant reply from the final state.

The graph itself (graph.py) does the actual LLM work. This module only
handles the HTTP ↔ graph bridge, mirroring travel_orchestrator.py's role
for the /plan endpoint.
"""

from __future__ import annotations

import logging

from direct_chat.graph import build_chat_graph
from direct_chat.models import ChatMessage, ChatResponse
from direct_chat.nodes import extract_last_reply
from direct_chat.prompts import CHAT_SYSTEM_PROMPT

logger = logging.getLogger("chat_orchestrator")

# Compile once at import time — checkpointed so conversations persist
# across turns per thread_id (conversation_id).
_CHAT_GRAPH = build_chat_graph()


async def chat(message: str, history: list[ChatMessage], conversation_id: str) -> ChatResponse:
    """Run a single chat turn through the LangGraph chat flow.

    Args:
        message: The user's current message.
        history: Optional prior turns — only used to seed a new conversation.
        conversation_id: UUID for session correlation / checkpointer key.

    Returns:
        ChatResponse with the assistant's reply.
    """
    logger.info("chat: conversation=%s message=%r", conversation_id, message)

    config = {"configurable": {"thread_id": conversation_id}}

    # Is this a fresh conversation or a resume?
    snapshot = _CHAT_GRAPH.get_state(config)
    is_new = not (snapshot and snapshot.values)

    if is_new:
        # Seed: system prompt + optional client history + user message.
        # The system message is only added once — on resume the
        # checkpointer already has it and add_messages appends only
        # the new user message.
        initial_messages: list[dict] = [
            {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        ]
        for m in (history or []):
            if m.role in ("user", "assistant"):
                initial_messages.append({"role": m.role, "content": m.content})
        initial_messages.append({"role": "user", "content": message})

        logger.info("chat: new conversation=%s (%d seed messages)",
                     conversation_id, len(initial_messages))
        result = await _CHAT_GRAPH.ainvoke({"messages": initial_messages}, config=config)
    else:
        # Resume: checkpointer has full history, just append the new message.
        logger.info("chat: resuming conversation=%s", conversation_id)
        result = await _CHAT_GRAPH.ainvoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
        )

    reply = extract_last_reply(result.get("messages", []))
    logger.info("chat: conversation=%s reply=%d chars", conversation_id, len(reply))
    return ChatResponse(reply=reply, conversation_id=conversation_id)
