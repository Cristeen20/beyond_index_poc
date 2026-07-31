"""State schema for the Direct Chat graph.

Uses LangGraph's `add_messages` reducer so each invocation appends
messages to the state instead of overwriting. This is what makes
server-side conversation persistence work: the checkpointer saves
the full message list after each turn, and the next turn loads it,
appends the new user message, and runs the chat node.

Future fields (RAG context, tool results, provider selection) get
added here without touching existing nodes.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import add_messages


class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
