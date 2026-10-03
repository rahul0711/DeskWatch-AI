"""Client for the external ScriptIndia /Recognize API.

This project's own pipeline only detects and tracks faces locally --
identity recognition/matching happens server-side on that external service.
A captured face crop is POSTed as multipart/form-data; whatever JSON it
returns (match found, unknown, error, ...) is handed back as-is, since the
exact response shape isn't something this codebase should assume.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)

RECOGNIZE_URL = "https://demo.scriptindia.in:8060/identifyFace"

# Single-site deployment for now -- fixed to this camera's location/company
# rather than read from .env, since there's currently only one site.
DEFAULT_IN_OUT_FLAG = "1"
DEFAULT_LATITUDE = "20.396390874233457"
DEFAULT_LONGITUDE = "72.91259635464365"
DEFAULT_COMPANY_ID = "2"

_TIMEOUT_S = 10.0


def recognize_face(image: np.ndarray) -> dict:
    """POST a BGR face crop to /Recognize and return its parsed JSON
    response. Never raises -- this runs off the render loop's hot path in a
    background thread, so a network hiccup should just show up as an
    {"error": ...} row in the recognitions table, not crash anything."""
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return {"error": "failed to encode face crop as JPEG"}

    files = {"file": ("face.jpg", buf.tobytes(), "image/jpeg")}
    data = {
        "InOutFlag": DEFAULT_IN_OUT_FLAG,
        "Latitude": DEFAULT_LATITUDE,
        "Longitude": DEFAULT_LONGITUDE,
        "CompanyId": DEFAULT_COMPANY_ID,
    }
    try:
        resp = requests.post(RECOGNIZE_URL, files=files, data=data, timeout=_TIMEOUT_S)
    except requests.RequestException as exc:
        logger.warning("Recognize API call failed: %s", exc)
        return {"error": str(exc)}

    try:
        return resp.json()
    except ValueError:
        return {
            "error": f"non-JSON response (status {resp.status_code})",
            "raw": resp.text[:500],
        }
