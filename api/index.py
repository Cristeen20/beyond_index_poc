"""Vercel serverless entrypoint for the FastAPI backend.

Vercel's Python runtime looks for an ASGI app named `app` in this module.
The actual application lives in `backend/` (unchanged, so `uvicorn main:app`
still works locally per README.md); this shim only puts `backend/` on
`sys.path` and re-exports the app.

`vercel.json` bundles `backend/**` alongside this file via `includeFiles`
and rewrites `/plan` and `/health` here.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import app  # noqa: E402  (import must follow the sys.path setup)

__all__ = ["app"]
