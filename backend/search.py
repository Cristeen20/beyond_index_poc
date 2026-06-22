import os
import httpx


BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"


async def fetch_advisories(destination: str) -> str:
    """Return a short text block of travel advisories. Empty string if no key is set."""
    api_key = os.getenv("BRAVE_API_KEY", "").strip()
    if not api_key:
        return ""

    query = f"{destination} travel advisory closures events 2026"

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            BRAVE_URL,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": api_key,
            },
            params={"q": query, "count": 5, "freshness": "pm"},
        )
        resp.raise_for_status()
        data = resp.json()

    snippets: list[str] = []
    for result in data.get("web", {}).get("results", []):
        title = result.get("title", "")
        desc = result.get("description", "")
        if title and desc:
            snippets.append(f"- {title}: {desc}")

    return "\n".join(snippets[:5])
