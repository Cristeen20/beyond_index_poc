# Direct Chat — Feature Spec (v1, LangGraph)

## What & Why

A standalone chat endpoint that lets the user talk freely with a travel-savvy LLM. Built on **LangGraph** — same framework as the itinerary pipeline — so the chat feature gets session checkpointing, message accumulation, and a clean path to add streaming, tools, RAG, and multi-provider support later.

v1's graph is intentionally minimal (`START → chat → END`), but the LangGraph foundation means every future feature is just adding nodes and edges.

---

## v1 Scope

| Capability | In v1? |
|---|---|
| Send message, get LLM reply | ✅ |
| Server-side conversation persistence (MemorySaver) | ✅ |
| Travel-specialized system prompt | ✅ |
| Multiple independent conversations (by `conversation_id`) | ✅ |
| Client-provided history seeding (for new conversations) | ✅ |
| Streaming (SSE) | ❌ (next) |
| Persistent storage (Postgres/Redis checkpointer) | ❌ (next) |
| Multiple LLM providers | ❌ (later) |
| Tool calls (weather, flights, etc.) | ❌ (later) |
| RAG / knowledge retrieval | ❌ (later) |

---

## Architecture

### Graph Topology (v1)

```
START ──► chat ──► END
            │
            │  (compiled with MemorySaver checkpointer,
            │   keyed by conversation_id / thread_id)
            ▼
      state persists across turns
```

One node, one edge. The `chat` node:
1. Reads all messages from state (system + history + current user message)
2. Calls `openai.chat.completions.create(model="gpt-4o")`
3. Returns the assistant reply — `add_messages` reducer appends it to state

### State Schema

```python
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
```

Uses LangGraph's `add_messages` reducer — each invocation appends messages instead of overwriting. This is what makes server-side conversation persistence work: the checkpointer saves the full message list after each turn, and the next turn loads it, appends the new user message, and runs the chat node.

### Session Model

- **MemorySaver** checkpointer (in-memory, process-wide) — same pattern as the travel graph's `graph/session.py`
- Keyed by `conversation_id` (UUID, generated client-side)
- **New conversation**: orchestrator seeds state with `[SystemMessage, user_message]` (+ optional client history)
- **Resume**: orchestrator sends just `[user_message]` — the checkpointer already has the full history, and `add_messages` appends
- Server restart loses state (MemorySaver is in-process) — swap to a persistent checkpointer for durability

### Why LangGraph instead of raw OpenAI calls?

| Concern | Raw OpenAI | LangGraph |
|---|---|---|
| Conversation persistence | Client sends full history every turn | Server stores via checkpointer |
| Adding tools | Rewrite the call loop | Add a `tool_node` + conditional edge |
| Adding RAG | Rewrite the call loop | Add a `retrieve` node before `chat` |
| Streaming | Manual SSE plumbing | `astream_events()` built in |
| Multi-step reasoning | Custom orchestration | Just more nodes + edges |
| Consistency with /plan | Different patterns | Same framework, same mental model |

---

## API

### `POST /chat`

**Request:**
```json
{
  "message": "what's a good time to visit Kyoto?",
  "conversation_id": "uuid-here",
  "history": [
    {"role": "user", "content": "I'm planning a trip to Japan"},
    {"role": "assistant", "content": "Great! When are you thinking of going?"}
  ]
}
```

- `message` — required, the user's current input
- `conversation_id` — required, UUID for session correlation
- `history` — **optional**, only used to seed a new conversation. On resume, the server already has the history from the checkpointer; this field is ignored.

**Response:**
```json
{
  "reply": "Late March to early April for cherry blossom season...",
  "conversation_id": "uuid-here"
}
```

---

## File Structure

```
backend/direct_chat/
├── __init__.py              # Exports app
├── main.py                  # FastAPI app + POST /chat endpoint
├── models.py                # ChatRequest, ChatResponse, ChatMessage
├── chat_orchestrator.py     # Invokes the graph (new vs. resume logic)
├── graph.py                 # build_chat_graph() + CHECKPOINTER
├── state.py                 # ChatState TypedDict (add_messages reducer)
├── nodes.py                 # chat_node (LLM call)
└── prompts.py               # System prompt(s)
```

### File Responsibilities

| File | Role |
|---|---|
| `prompts.py` | System prompt constant. Future: tool descriptions, RAG context templates. |
| `state.py` | `ChatState` TypedDict with `add_messages` reducer. Future: add fields for retrieved context, tool outputs, etc. |
| `nodes.py` | `chat_node()` — converts state messages to OpenAI format, calls LLM, returns reply. Future: `tool_node()`, `retrieve_node()`, `moderation_node()`. |
| `graph.py` | `build_chat_graph()` — wires nodes + edges, compiles with checkpointer. Future: add nodes/edges here. |
| `chat_orchestrator.py` | `chat()` — detects new vs. resume, invokes graph, extracts reply. Thin entry point. |
| `models.py` | Pydantic request/response models. |
| `main.py` | FastAPI app, CORS, endpoint registration. |

---

## How to Run

**Backend** (port 8001, from `backend/`):
```bash
uvicorn direct_chat.main:app --port 8001 --reload
```

**Frontend** (port 3001, from `frontend/direct_chat/`):
```bash
npm install && npm run dev
```

Ensure `OPENAI_API_KEY` is set in `backend/.env`.

---

## Scalability — How Future Features Layer On

Each feature is a new node and/or edge. The v1 graph and state don't change — they grow.

### Streaming (SSE)

Switch `chat_orchestrator.chat()` from `ainvoke()` to `astream_events()`. The FastAPI endpoint returns a `StreamingResponse` that yields tokens as they arrive. No graph changes needed — LangGraph streams node output natively.

```
# orchestrator change:
async for event in _CHAT_GRAPH.astream_events(input, config=config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        yield event["data"]["chunk"].content
```

### Persistent storage

Swap `MemorySaver` for a durable backend in `graph.py`:

```python
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

CHECKPOINTER = PostgresSaver.from_conn_string(os.environ["DATABASE_URL"])
```

Zero changes to nodes, state, or orchestrator.

### Tool calls (weather, flights, currency)

Add a `tool_node` and a conditional edge after `chat`:

```
START ──► chat ──► should_call_tools? ──► tool_node ──► chat ──► END
                           └── no tools ──► END
```

- `state.py`: add `tool_results: list[dict]` field
- `nodes.py`: add `tool_node()` that executes tool calls and `should_call_tools()` edge function
- `graph.py`: wire the new nodes + conditional edges
- `prompts.py`: add tool schemas/descriptions

### RAG / knowledge retrieval

Add a `retrieve` node before `chat`:

```
START ──► retrieve ──► chat ──► END
```

- `state.py`: add `retrieved_context: str` field
- `nodes.py`: add `retrieve_node()` that queries a vector store
- `graph.py`: add the node + edge
- `chat_node` reads `state["retrieved_context"]` and includes it in the prompt

### Multiple providers

Add a provider field to state and a routing node:

```
START ──► route_provider ──► openai_chat ──► END
                      └──► anthropic_chat ──► END
                      └──► local_chat ──────► END
```

Or simpler: a provider registry in `nodes.py` that `chat_node` looks up based on `state["provider"]`.

### Per-conversation system prompt

Add `system_prompt: str` to `ChatState` (no reducer — set once, preserved across turns). Orchestrator sets it on new conversations; `chat_node` reads it from state instead of importing the constant. v1 uses the default from `prompts.py`; future versions let the client override via `ChatRequest`.

### Conversation management endpoints

```
GET    /chat/{conversation_id}   → full message history
DELETE /chat/{conversation_id}   → clear conversation
GET    /chat                     → list conversations
```

All read from / write to the checkpointer. No graph changes.
