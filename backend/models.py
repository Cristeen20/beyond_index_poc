from pydantic import BaseModel


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatHistoryItem] = []


class ChatResponse(BaseModel):
    text: str
