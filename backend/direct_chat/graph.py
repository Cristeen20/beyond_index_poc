"""Chat graph builder + process-wide checkpointer.

v1 topology:  START → chat → END

Compiled with a MemorySaver checkpointer so conversation state
(message history) persists across /chat calls, keyed by
conversation_id (thread_id). Same pattern as the travel graph's
graph/session.py.

Future: add nodes (retrieve, tool_executor, moderation) and
conditional edges here. The state schema and existing nodes don't
change — the graph just grows.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from direct_chat.nodes import chat_node
from direct_chat.state import ChatState

# Single process-wide checkpointer — MemorySaver holds state in-process
# and does not survive server restart. Swap to PostgresSaver /
# RedisSaver for durability (see features/direct_chat.md §Scalability).
CHECKPOINTER = MemorySaver()


def build_chat_graph():
    """Compile and return the Direct Chat StateGraph."""
    g = StateGraph(ChatState)
    g.add_node("chat", chat_node)
    g.add_edge(START, "chat")
    g.add_edge("chat", END)
    return g.compile(checkpointer=CHECKPOINTER)
