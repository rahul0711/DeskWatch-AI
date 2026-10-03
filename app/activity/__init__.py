"""Human activity analysis layer.

Sits on top of the existing capture/detection pipeline without replacing any
of it: ActivityDetector (app/detection/) produces per-frame observations,
and this package turns them into tracked states, durations and events.

Public surface:
    load_activity_config()   -> ActivityConfig  (env-driven)
    CameraActivityAnalyzer   -> one per camera, call .observe() per pass
    ActivityRepository       -> queued event persistence
    Activity                 -> the activity-type string constants
"""
from app.activity.analyzer import CameraActivityAnalyzer
from app.activity.config import ActivityConfig, load_activity_config
from app.activity.identity_binder import Identity
from app.activity.repository import ActivityRepository
from app.activity.state_manager import Activity, ActivityEvent

__all__ = [
    "Activity",
    "ActivityConfig",
    "ActivityEvent",
    "ActivityRepository",
    "CameraActivityAnalyzer",
    "Identity",
    "load_activity_config",
]
