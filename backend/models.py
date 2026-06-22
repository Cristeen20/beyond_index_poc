from pydantic import BaseModel, Field
from typing import Optional


class ItineraryRequest(BaseModel):
    destination: str
    days: int = Field(default=3, ge=1, le=14)
    interests: list[str] = []
    startDate: Optional[str] = None


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatHistoryItem] = []


class Stop(BaseModel):
    name: str
    time: str
    duration_minutes: int
    notes: str


class DayPlan(BaseModel):
    day: int
    theme: str
    stops: list[Stop]
    lodging: str


class Itinerary(BaseModel):
    destination: str
    days: list[DayPlan]
    advisories: list[str]


class ChatResponse(BaseModel):
    text: str
    itinerary: Optional[Itinerary] = None
