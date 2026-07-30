"""Pydantic models for the Direct Chat feature (features/direct_chat.md)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in the conversation history."""

    role: str = Field(..., description="One of: 'user', 'assistant', 'system'")
    content: str = Field(..., description="The message text")


class ChatRequest(BaseModel):
    """Request body for POST /chat.

    `history` is optional — only used to seed a new conversation. On
    resume, the server already has the full history from the LangGraph
    checkpointer; this field is ignored.
    """

    message: str = Field(..., description="The user's current message")
    conversation_id: str = Field(default="", description="UUID for conversation correlation")
    history: list[ChatMessage] = Field(default_factory=list, description="Prior turns (new conversations only)")


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    reply: str = Field(..., description="The assistant's reply text")
    conversation_id: str = Field(default="", description="Echoed back for client correlation")
