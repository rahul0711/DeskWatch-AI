"""Admin-selected activity list, persisted across restarts.

The admin picks which activities are analysed and stored from the
dashboard (PUT /api/activity/settings). That choice lives in
data/activity_settings.json and overrides ACTIVITY_TYPES from .env, which
is only the default for a fresh install.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from app.activity.config import ROOT, normalize_activity_types

logger = logging.getLogger(__name__)

_SETTINGS_PATH = ROOT / "data" / "activity_settings.json"
_lock = threading.Lock()


def load_enabled_activities(default: tuple[str, ...], path: Path = _SETTINGS_PATH) -> tuple[str, ...]:
    with _lock:
        try:
            data = json.loads(path.read_text())
        except FileNotFoundError:
            return tuple(default)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read %s (%s) -- using ACTIVITY_TYPES", path, exc)
            return tuple(default)
    return normalize_activity_types(data.get("enabled_activities", default))


def save_enabled_activities(activities, path: Path = _SETTINGS_PATH) -> tuple[str, ...]:
    enabled = normalize_activity_types(activities)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"enabled_activities": list(enabled)}, indent=2))
        tmp.replace(path)
    return enabled
