"""Per-camera activity analysis orchestrator.

This is the only part of app/activity/ that the rest of the app talks to.
One instance per camera, living in the parent process.

Data flow for one activity detection pass:

    ActivityDetector (subprocess)          face pipeline (subprocess)
      person tracks + posture                 face tracks + /Recognize
      + phone/monitor + keypoints                      |
                 |                                     |
                 v                                     v
           MovementAnalyzer  ---------->  identity_binder.bind()
                 |                                     |
                 +------------------+------------------+
                                    v
                        ActivityStateManager.update()
                                    v
                    events -> ActivityRepository (queued)
                           -> recent-events ring (API)
                    tracks <- annotated for the video overlay

Everything stateful lives here rather than in the detector because the
detector runs in a worker subprocess that may be restarted; losing a
person's accumulated sitting time to a worker restart would be a data bug,
not a hiccup.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from collections import deque
from typing import Optional

from app.activity.config import ActivityConfig
from app.activity.identity_binder import Identity, bind
from app.activity.movement import MovementAnalyzer
from app.activity.repository import ActivityRepository
from app.activity.state_manager import ActivityStateManager

logger = logging.getLogger(__name__)


class CameraActivityAnalyzer:
    def __init__(
        self,
        camera_id: str,
        config: ActivityConfig,
        repository: Optional[ActivityRepository] = None,
    ) -> None:
        self.camera_id = camera_id
        self._cfg = config
        self._repo = repository
        self._movement = MovementAnalyzer(
            ema_alpha=config.movement_ema_alpha,
            weights=config.movement_weights,
            keypoint_min_conf=config.keypoint_min_conf,
        )
        self._states = ActivityStateManager(
            camera_id=camera_id,
            confirmation_seconds=config.confirmation_seconds,
            identity_grace_seconds=config.identity_grace_seconds,
            track_grace_seconds=config.track_grace_seconds,
            walking_threshold=config.walking_threshold,
            stationary_threshold=config.stationary_threshold,
            stationary_min_seconds=config.stationary_min_seconds,
            min_event_seconds=config.min_event_seconds,
            min_track_observations=config.min_track_observations,
            max_tracks=config.max_tracks,
        )
        self._recent: deque = deque(maxlen=config.max_recent_events)

    # -- main entry point ----------------------------------------------

    def observe(
        self,
        activity_tracks: list,
        face_tracks: Optional[list] = None,
        face_identities: Optional[dict] = None,
        now: Optional[float] = None,
    ) -> list:
        """Process one activity detection pass.

        activity_tracks -- TrackedFace list from ActivityDetector.
        face_tracks     -- TrackedFace list from the face/attendance detector,
                           used only to locate faces for identity binding.
        face_identities -- {face_track_id: Identity} learned by recognition.

        Returns the events that closed on this pass. Also annotates
        activity_tracks in place with movement score, current activity and
        live timers for the overlay.
        """
        now = now if now is not None else time.monotonic()
        now_wall = dt.datetime.now(dt.timezone.utc)

        scores: dict[int, float] = {}
        for t in activity_tracks:
            scores[t.track_id] = self._movement.update(
                t.track_id, t.face.bbox, t.keypoints, t.keypoints_conf, now
            )
        self._movement.keep_only(scores.keys())

        events = self._states.update(activity_tracks, scores, now, now_wall)

        if face_tracks and face_identities:
            for person_track_id, identity in bind(
                face_tracks, activity_tracks, face_identities
            ).items():
                self._states.apply_identity(
                    person_track_id,
                    identity.person_id,
                    identity.person_name,
                    identity.confidence,
                    now,
                )

        if events:
            for event in events:
                self._recent.append(event.as_dict())
            if self._repo is not None:
                self._repo.record_many(events)

        self._annotate(activity_tracks, now)
        return events

    def _annotate(self, activity_tracks: list, now: float) -> None:
        states = self._states.states
        for t in activity_tracks:
            state = states.get(t.track_id)
            if state is None:
                continue
            t.movement_score = state.movement_score
            t.activity = state.display_activity
            t.primary_activity = (
                str(state.primary.value) if state.primary is not None else "UNKNOWN"
            )
            t.activity_seconds = state.display_since(now)
            t.stationary_seconds = state.stationary_seconds(now)
            if state.person_name:
                t.name = state.person_name
                t.name_confidence = state.identity_confidence

    # -- reads ----------------------------------------------------------

    def rows(self, now: Optional[float] = None) -> list[dict]:
        """Live per-person activity rows."""
        return self._states.rows(now if now is not None else time.monotonic())

    def recent_events(self, limit: int = 100) -> list[dict]:
        """Chronological in-memory event log, newest last."""
        events = list(self._recent)
        return events[-max(1, limit):]

    def needs_identity(self, track_id: int, now: Optional[float] = None) -> bool:
        now = now if now is not None else time.monotonic()
        return self._states.identity_is_stale(track_id, now)

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Flush every open interval so a restart doesn't silently lose the
        time people had already accumulated."""
        try:
            events = self._states.close_all(time.monotonic())
        except Exception:
            logger.exception("[%s] Failed closing activity states", self.camera_id)
            return
        for event in events:
            self._recent.append(event.as_dict())
        if self._repo is not None and events:
            self._repo.record_many(events)
        logger.info("[%s] Activity analyzer closed (%d open interval(s) flushed)",
                    self.camera_id, len(events))
