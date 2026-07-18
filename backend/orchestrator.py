import json
import logging

import openai

from models import ItineraryRequest, Itinerary, ChatRequest, ChatResponse
from places import fetch_places
from directions import get_travel_times
from search import fetch_advisories

logger = logging.getLogger("chat_orchestrator")


_client: openai.AsyncOpenAI | None = None

def _get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI()
    return _client

_ITINERARY_TOOL = {
    "type": "function",
    "function": {
        "name": "return_itinerary",
        "description": "Return the complete structured trip itinerary.",
        "parameters": {
            "type": "object",
            "required": ["destination", "days", "advisories"],
            "properties": {
                "destination": {"type": "string"},
                "advisories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Current travel advisories, closures, or seasonal notes.",
                },
                "days": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["day", "theme", "stops", "lodging"],
                        "properties": {
                            "day": {"type": "integer"},
                            "theme": {
                                "type": "string",
                                "description": "Short thematic description for this day, e.g. 'Eastern temples'.",
                            },
                            "lodging": {"type": "string"},
                            "stops": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["name", "time", "duration_minutes", "notes"],
                                    "properties": {
                                        "name": {"type": "string"},
                                        "time": {
                                            "type": "string",
                                            "description": "Suggested arrival time, e.g. '9:00 AM'.",
                                        },
                                        "duration_minutes": {"type": "integer"},
                                        "notes": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


async def generate_itinerary(req: ItineraryRequest) -> Itinerary:
    # 1. Fetch candidate places from Google Places API
    places = await fetch_places(req.destination, req.interests)

    # 2. Compute walking times between the top places via Directions API
    travel_matrix: list[list[str]] = []
    place_names: list[str] = []
    try:
        travel_matrix, place_names = await get_travel_times(places)
    except Exception:
        pass  # non-fatal — Claude will still group by coordinates

    # 3. Optional web advisory search
    advisories_text = ""
    try:
        advisories_text = await fetch_advisories(req.destination)
    except Exception:
        pass

    # 4. Build prompt
    prompt = _build_prompt(req, places, travel_matrix, place_names, advisories_text)

    # 5. Synthesise with GPT-4o — forced function call guarantees structured output
    response = await _get_client().chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        tools=[_ITINERARY_TOOL],
        tool_choice={"type": "function", "function": {"name": "return_itinerary"}},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert travel planner. Use the provided place data and travel times "
                    "to create a practical, geographically sensible itinerary. Group stops by proximity "
                    "to minimise transit time between them. Assign realistic start times and visit durations. "
                    "Suggest appropriate lodging near each day's stops. "
                    "Include any relevant advisories; if none exist write an empty array."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    for choice in response.choices:
        for tool_call in choice.message.tool_calls or []:
            if tool_call.function.name == "return_itinerary":
                return Itinerary(**json.loads(tool_call.function.arguments))

    raise RuntimeError("OpenAI did not return a structured itinerary")


_FETCH_PLACES_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_places_data",
        "description": (
            "Fetch real place data from Google Places for a destination. "
            "Call this when recommending specific places, answering what to see/do somewhere, "
            "or gathering place data before building an itinerary."
        ),
        "parameters": {
            "type": "object",
            "required": ["destination"],
            "properties": {
                "destination": {"type": "string", "description": "City, region, or area to search"},
                "interests": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional themes, e.g. 'historic', 'food', 'nature', 'hiking'",
                },
            },
        },
    },
}

_CHAT_SYSTEM = (
    "You are a friendly, knowledgeable travel assistant. Respond conversationally.\n\n"
    "- For 'best spots in X' or 'what to see/do in X': call fetch_places_data, then highlight "
    "the top places naturally in prose — no rigid lists unless the user asks for them.\n"
    "- For themed trip suggestions ('historic', 'foodie', 'nature'): call fetch_places_data with "
    "relevant interests, then describe fitting places in that theme.\n"
    "- For questions about a specific place the user just mentioned: answer from your knowledge; "
    "only call fetch_places_data if you need fresh data.\n"
    "- For general tips, weather, packing, visa: answer directly — no tools needed.\n"
    "- ONLY call return_itinerary when the user explicitly asks for an itinerary, trip plan, or "
    "day-by-day schedule. Never generate one just for a recommendations request.\n"
    "- When generating an itinerary, call fetch_places_data first to gather real place data."
)


async def chat(req: ChatRequest) -> ChatResponse:
    tools = [_FETCH_PLACES_TOOL, _ITINERARY_TOOL]

    messages: list[dict] = [{"role": "system", "content": _CHAT_SYSTEM}]
    for m in req.history:
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": req.message})

    logger.info("chat: incoming message=%r history_len=%d",
                req.message, len(req.history))

    client = _get_client()

    for turn in range(6):
        response = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=2048,
            tools=tools,
            messages=messages,
        )
        choice = response.choices[0]

        if not choice.message.tool_calls:
            logger.info("chat: turn=%d → plain reply (no tool calls)", turn)
            return ChatResponse(text=choice.message.content or "")

        messages.append(choice.message)
        logger.info(
            "chat: turn=%d tool_calls=%s",
            turn,
            [tc.function.name for tc in choice.message.tool_calls],
        )

        itinerary_result: Itinerary | None = None
        tool_results: list[dict] = []

        for tc in choice.message.tool_calls:
            if tc.function.name == "return_itinerary":
                args = json.loads(tc.function.arguments)
                logger.info(
                    "chat: tool_call=return_itinerary destination=%r days=%d",
                    args.get("destination"), len(args.get("days") or []),
                )
                itinerary_result = Itinerary(**args)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Itinerary created successfully.",
                })

            elif tc.function.name == "fetch_places_data":
                args = json.loads(tc.function.arguments)
                logger.info(
                    "chat: tool_call=fetch_places_data destination=%r interests=%s",
                    args["destination"], args.get("interests", []),
                )
                places = await fetch_places(args["destination"], args.get("interests", []))
                logger.info("chat: fetch_places_data → %d places", len(places))

                travel_matrix: list[list[str]] = []
                place_names: list[str] = []
                try:
                    travel_matrix, place_names = await get_travel_times(places)
                except Exception:
                    pass

                advisories = ""
                try:
                    advisories = await fetch_advisories(args["destination"])
                except Exception:
                    pass

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({
                        "places": places,
                        "travel_matrix": travel_matrix,
                        "place_names": place_names,
                        "advisories": advisories,
                    }),
                })

        messages.extend(tool_results)

        if itinerary_result is not None:
            return ChatResponse(text="Here's your itinerary!", itinerary=itinerary_result)

    return ChatResponse(text="Sorry, I wasn't able to complete that request.")


def _build_prompt(
    req: ItineraryRequest,
    places: list[dict],
    travel_matrix: list[list[str]],
    place_names: list[str],
    advisories_text: str,
) -> str:
    lines = [
        f"Destination: {req.destination}",
        f"Duration: {req.days} day{'s' if req.days != 1 else ''}",
    ]
    if req.interests:
        lines.append(f"Traveller interests: {', '.join(req.interests)}")
    if req.startDate:
        lines.append(f"Start date: {req.startDate}")

    lines.append(f"\n## Candidate places ({len(places)} found)\n")
    lines.append(json.dumps(places, indent=2))

    if travel_matrix and place_names:
        lines.append("\n## Walking travel times (top places)\n")
        header = ["from \\ to"] + place_names
        lines.append(" | ".join(header))
        lines.append(" | ".join(["---"] * len(header)))
        for i, row in enumerate(travel_matrix):
            lines.append(" | ".join([place_names[i]] + row))

    if advisories_text:
        lines.append("\n## Live travel advisories\n")
        lines.append(advisories_text)

    lines.append(
        f"\nPlan a {req.days}-day itinerary with 3–5 stops per day. "
        "Group stops that are geographically close together within each day. "
        "Use the travel time matrix above to sequence stops logically. "
        "Call return_itinerary with the complete result."
    )

    return "\n".join(lines)
