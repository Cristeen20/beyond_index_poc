"""Graph nodes for the Direct Chat feature.

v1 has a single node (`chat_node`) that calls the LLM. Future nodes
(tool execution, RAG retrieval, moderation) get added here and wired
in `graph.py`.
"""

from __future__ import annotations

import logging
import os

from langfuse.openai import openai
from langchain_core.messages import AIMessage, BaseMessage

from direct_chat.state import ChatState

logger = logging.getLogger("direct_chat.nodes")

_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_MAX_TOKENS = 800

_llm_client: openai.AsyncOpenAI | None = None


def _get_llm() -> openai.AsyncOpenAI:
    global _llm_client
    if _llm_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise KeyError("OPENAI_API_KEY")
        _llm_client = openai.AsyncOpenAI(api_key=api_key)
    return _llm_client


def _to_openai_messages(messages: list) -> list[dict]:
    """Convert state messages (BaseMessage or dict) to OpenAI API format."""
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    result: list[dict] = []
    for m in messages:
        if isinstance(m, dict):
            result.append(m)
        elif isinstance(m, BaseMessage):
            result.append({
                "role": role_map.get(m.type, m.type),
                "content": m.content,
            })
    return result


async def chat_node(state: ChatState) -> dict:
    """Call the LLM with the full conversation history and return the reply.

    The `add_messages` reducer in the state schema ensures this node sees
    all prior messages (system + history + current user message). The
    returned dict is merged back via the same reducer, appending the
    assistant reply.
    """
    messages = _to_openai_messages(state["messages"])
    resp = await _get_llm().chat.completions.create(
        model=_DEFAULT_MODEL,
        max_tokens=_DEFAULT_MAX_TOKENS,
        messages=messages,
    )
    reply = (resp.choices[0].message.content or "").strip()
    logger.info("chat_node → %d chars", len(reply))
    # Returned as a dict — add_messages converts to AIMessage and appends.
    return {"messages": [{"role": "assistant", "content": reply}]}


def extract_last_reply(messages: list) -> str:
    """Pull the last AIMessage content from the state's message list."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m.content
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "")
    return ""
