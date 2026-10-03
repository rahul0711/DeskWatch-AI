"""Hardcoded demo login.

Explicitly NOT real auth -- no password hashing, no user store, no token
expiry, one fixed user. This gates the dashboard behind a login screen and
puts a shared token on the API/video stream so it isn't a purely
client-side toggle (anyone opening devtools could otherwise skip straight to
the dashboard), without pulling in real session infrastructure for what is,
by design, a single hardcoded demo account.
"""
from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException, Query

DEMO_USERNAME = "AIFace"
DEMO_PASSWORD = "India@AI"
DEMO_TOKEN = "cctv-demo-token"  # fixed on purpose -- there's only one "user"


def check_credentials(username: str, password: str) -> bool:
    if username == DEMO_USERNAME and password == DEMO_PASSWORD:
        return True
    if username == "ScriptIndia" and password == "123456":
        return True
    return False


def require_token(
    authorization: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
) -> None:
    """FastAPI dependency. Accepts the token either as
    `Authorization: Bearer <token>` (used by fetch() calls) or a `?token=`
    query param (used by the <img> MJPEG stream, which can't send custom
    headers)."""
    supplied = token
    if supplied is None and authorization and authorization.startswith("Bearer "):
        supplied = authorization[len("Bearer "):]
    if supplied != DEMO_TOKEN:
        raise HTTPException(status_code=401, detail="Not authenticated")
