"""Login check for the Python servers.

Two modes, chosen by .env:

  JWT_SECRET set (normal)  -- people log in through the .NET backend
      (CCTV_Backend_Project, POST /api/db/auth/login), which issues an HS256
      JWT. Here we verify that token with the same secret, then look the
      login up in MySQL: the session must not be revoked, the login must be
      active, and its role and cameras come from app_users /
      v_app_user_cameras. The old demo token is refused.

  JWT_SECRET unset (legacy / tests) -- the old hardcoded demo login and
      shared demo token, treated as a Super Admin on every camera.

Every route depends on require_token (any logged-in role), require_roles
(only some roles) or require_camera (the login must have that camera).
"""
from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, Query

logger = logging.getLogger(__name__)

DEMO_USERNAME = "AIFace"
DEMO_PASSWORD = "India@AI"
DEMO_TOKEN = "cctv-demo-token"  # legacy mode only

SUPER_ADMIN = "SUPER_ADMIN"
ADMIN = "ADMIN"
VIEWER = "VIEWER"
ADMIN_OR_ABOVE = (SUPER_ADMIN, ADMIN)

_LEEWAY_S = 60
_CACHE_S = 30.0


@dataclass(frozen=True)
class Principal:
    login_id: Optional[int]
    username: str
    role: str
    cameras: Optional[frozenset[str]]  # None = every camera

    def can_see(self, camera_id: str) -> bool:
        return self.cameras is None or camera_id in self.cameras


LEGACY_PRINCIPAL = Principal(None, DEMO_USERNAME, SUPER_ADMIN, None)


def jwt_enabled() -> bool:
    return bool(os.getenv("JWT_SECRET"))


def check_credentials(username: str, password: str) -> bool:
    """Legacy hardcoded logins (only used when JWT_SECRET is unset)."""
    if username == DEMO_USERNAME and password == DEMO_PASSWORD:
        return True
    if username == "ScriptIndia" and password == "123456":
        return True
    return False


# --- JWT (HS256) -------------------------------------------------------------

def _b64url_decode(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def decode_jwt(token: str) -> Optional[dict]:
    """Claims of a valid token from the .NET backend, else None. Checks the
    HS256 signature, issuer, audience and expiry."""
    secret = os.getenv("JWT_SECRET", "")
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg") != "HS256":
            return None
        expected = hmac.new(secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(sig_b64)):
            return None
        claims = json.loads(_b64url_decode(payload_b64))
    except (ValueError, TypeError, json.JSONDecodeError):
        return None

    now = time.time()
    if claims.get("iss") != os.getenv("JWT_ISSUER", claims.get("iss")):
        return None
    aud = claims.get("aud")
    want_aud = os.getenv("JWT_AUDIENCE")
    if want_aud and want_aud not in (aud if isinstance(aud, list) else [aud]):
        return None
    if "exp" not in claims or now > float(claims["exp"]) + _LEEWAY_S:
        return None
    if "nbf" in claims and now < float(claims["nbf"]) - _LEEWAY_S:
        return None
    return claims


# --- Login lookup in MySQL (cached per session) --------------------------------

_cache: dict[int, tuple[float, Optional[Principal], str]] = {}
_cache_lock = threading.Lock()


def _load_principal(login_id: int, session_id: int, token_role: str) -> tuple[Optional[Principal], str]:
    from sqlalchemy import text

    from app.attendance.db import get_session  # lazy: no database needed at import

    with get_session() as db:
        row = db.execute(text("""
            SELECT a.id, a.username, a.role_code, a.is_active, a.all_cameras, a.must_change_password,
                   s.expires_at, s.revoked_at
            FROM app_user_sessions s JOIN app_users a ON a.id = s.app_user_id
            WHERE s.id = :sid
        """), {"sid": session_id}).mappings().first()
        if row is None or row["id"] != login_id:
            return None, "Session not found, log in again"
        if row["revoked_at"] is not None:
            return None, "Session ended, log in again"
        if row["expires_at"] <= dt.datetime.now(dt.timezone.utc).replace(tzinfo=None):
            return None, "Session expired, log in again"
        if not row["is_active"]:
            return None, "This login is disabled"
        if row["role_code"] != token_role:
            return None, "Role changed, log in again"
        if row["must_change_password"]:
            return None, "Change your password first"
        cameras: Optional[frozenset[str]] = None
        if row["role_code"] != SUPER_ADMIN and not row["all_cameras"]:
            cameras = frozenset(db.execute(
                text("SELECT camera_id FROM v_app_user_cameras WHERE app_user_id = :id"), {"id": login_id}
            ).scalars().all())
    return Principal(login_id, row["username"], row["role_code"], cameras), ""


def principal_from_token(token: Optional[str]) -> Principal:
    """The login behind a token, or HTTPException(401)."""
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    if not jwt_enabled():
        if token != DEMO_TOKEN:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return LEGACY_PRINCIPAL

    claims = decode_jwt(token)
    if claims is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        login_id, session_id, role = int(claims["sub"]), int(claims["sid"]), str(claims["role"])
    except (KeyError, ValueError):
        raise HTTPException(status_code=401, detail="Not authenticated")

    now = time.monotonic()
    with _cache_lock:
        hit = _cache.get(session_id)
    if hit is None or hit[0] < now:
        try:
            principal, reason = _load_principal(login_id, session_id, role)
        except Exception as exc:  # database unreachable
            logger.error("Login lookup failed: %s", exc)
            raise HTTPException(status_code=503, detail="Login check unavailable, try again")
        hit = (now + _CACHE_S, principal, reason)
        with _cache_lock:
            _cache[session_id] = hit
    principal, reason = hit[1], hit[2]
    if principal is None:
        raise HTTPException(status_code=401, detail=reason or "Not authenticated")
    return principal


# --- FastAPI dependencies ---------------------------------------------------------

def require_token(
    authorization: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
) -> Principal:
    """Any logged-in role. Token as `Authorization: Bearer <token>` (fetch calls)
    or `?token=` (the <img> MJPEG stream and the WebSocket can't send headers)."""
    supplied = token
    if supplied is None and authorization and authorization.startswith("Bearer "):
        supplied = authorization[len("Bearer "):]
    return principal_from_token(supplied)


def require_roles(*roles: str):
    """Only these roles, e.g. Depends(require_roles(SUPER_ADMIN, ADMIN))."""
    def dependency(principal: Principal = Depends(require_token)) -> Principal:
        if principal.role not in roles:
            raise HTTPException(status_code=403, detail="Your role is not allowed to do this")
        return principal
    return dependency


def require_camera(camera_id: str, principal: Principal = Depends(require_token)) -> Principal:
    """Any role, but only on cameras assigned to the login."""
    if not principal.can_see(camera_id):
        raise HTTPException(status_code=403, detail=f"You do not have access to camera '{camera_id}'")
    return principal
