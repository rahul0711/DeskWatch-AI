"""Per-camera activity analysis orchestrator.

This is the only part of app/activity/ that the rest of the app talks to.
One instance per camera, living in the parent process.

Data flow for one activity detection pass:

    RTSP -> ActivityDetector (subprocess)        face worker (subprocess)
              detector -> tracker -> pose          SCRFD -> ByteTrack ->
              person tracks + phone/monitor        InsightFace embedding
                       |                           (scheduled, cached)
                       |                           + /Recognize result
                       v                                   |
                 MovementAnalyzer                          v
                       |                IdentityCache: track_id -> USER_ID
                       |                (explicit id, or embedding re-id)
                       +--------------------+--------------+
                                            v
                         ActivityStateManager.update()   (state per USER)
                                            v
                          ActivitySession per user x activity
                                            v
                 closed sessions -> ActivityRepository -> activity_sessions
                                 -> recent-events ring (API)
                 tracks <- annotated from the USER's session state (overlay)

A track id is only short-term continuity. Activities, sessions and totals
belong to the user the IdentityCache resolves it to.

Everything stateful lives here rather than in the detector because the
detector runs in a worker subprocess that may be restarted; losing a
person's accumulated sitting time to a worker restart would be a data bug,
not a hiccup.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from collections import deque
from typing import Optional

import copy

from app.activity.config import ActivityConfig
from app.activity.identity_cache import IdentityCache
from app.activity.movement import MovementAnalyzer
from app.activity.repository import ActivityRepository
from app.activity.state_manager import ActivityStateManager

logger = logging.getLogger(__name__)



# On-screen only: how long after its last real sighting a phone stays drawn.
_PHONE_DISPLAY_HOLD_S = 3.0

class CameraActivityAnalyzer:
    def __init__(
        self,
        camera_id: str,
        config: ActivityConfig,
        repository: Optional[ActivityRepository] = None,
        enabled_activities: Optional[tuple] = None,
    ) -> None:
        self.camera_id = camera_id
        self._cfg = config
        self._repo = repository
        self._identities = IdentityCache(
            match_threshold=config.reid_threshold,
            match_margin=config.reid_margin,
            binding_ttl_seconds=config.track_grace_seconds,
        )
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
            activation_seconds=config.activation_seconds,
            session_grace_seconds=config.session_grace_seconds,
            phone_grace_seconds=config.phone_grace_seconds,
            phone_activation_seconds=config.phone_activation_seconds,
            screen_grace_seconds=config.screen_grace_seconds,
            enabled_activities=enabled_activities if enabled_activities is not None else config.activity_types,
            baseline_loader=self._load_baseline if (repository is not None and config.persist_events) else None,
        )
        self._recent: deque = deque(maxlen=config.max_recent_events)
        # Last drawn copy of each track, so a person the detector misses for
        # a pass or two stays on screen instead of blinking (see _coast).
        self._last_drawn: dict[int, tuple[float, object]] = {}
        # observe() runs on the camera's render thread; a view or settings
        # change (set_enabled_activities) and the status/rows endpoints come
        # from other threads. One lock keeps them from changing the per-person
        # state while it is being read or updated.
        self._lock = threading.RLock()

    # -- main entry point ----------------------------------------------

    def observe(self, *args, **kwargs) -> list:
        with self._lock:
            return self._observe(*args, **kwargs)

    def _observe(
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

        # track_id -> USER_ID, before any time is accumulated.
        bindings = self._identities.update(
            face_tracks or [], activity_tracks, face_identities or {}, now,
        )
        events = self._states.update(activity_tracks, scores, now, now_wall, identities=bindings)
        self._record(events)

        self._annotate(activity_tracks, now)
        self._coast(activity_tracks, now)
        return events

    def _record(self, events: list) -> None:
        if not events:
            return
        # A phone use whose phone-use confidence never reached the violation
        # threshold is too doubtful to count -- not stored, so the Violations
        # page, Phone usage page and live alert all agree.
        kept = []
        for event in events:
            if event.activity_type == "PHONE_INTERACTION":
                state = self._states.state_for_track(event.track_id)
                peak = getattr(state, "phone_peak_confidence", 0.0) if state is not None else 0.0
                if peak < self._cfg.phone_violation_conf:
                    logger.info("[%s] Phone use of %.1fs not stored: confidence peaked at %.0f%%",
                                self.camera_id, event.duration_seconds, 100 * peak)
                    continue
            kept.append(event)
        events = kept
        if not events:
            return
        for event in events:
            self._recent.append(event.as_dict())
        if self._repo is not None:
            self._repo.record_many(events)

    def _load_baseline(self, person_id: str) -> dict:
        return ActivityRepository.user_totals(person_id, camera_id=self.camera_id)

    def _annotate(self, activity_tracks: list, now: float) -> None:
        for t in activity_tracks:
            state = self._states.state_for_track(t.track_id)
            if state is None:
                continue
            # Draw the USER's session state, not this pass's raw detection:
            # a session in its grace window stays on, so one missed phone
            # detection does not make the overlay blink.
            # ...except the phone, which is drawn from its first sighting
            # (PENDING) so the Phone view reacts at once, and only while it
            # was actually seen recently: the session's 10s grace keeps one
            # phone use as ONE recorded session, but on screen it made the
            # label linger after the phone was put down (e.g. drinking from
            # a bottle right after). Recording is unchanged.
            phone = state.session("phone")
            state.phone_use_confidence = getattr(t, "phone_use_confidence", 0.0)
            # Highest confidence during the current phone use (reset when it ends).
            in_use = state.is_on("phone") or state.is_starting("phone")
            state.phone_peak_confidence = (
                max(getattr(state, "phone_peak_confidence", 0.0), state.phone_use_confidence) if in_use else 0.0
            )
            seen_recently = (
                phone is not None and phone.last_confirmed is not None
                and now - phone.last_confirmed <= _PHONE_DISPLAY_HOLD_S
            )
            t.phone_in_use = state.is_starting("phone") or (state.is_on("phone") and seen_recently)
            t.phone_session_seconds = phone.current_duration(now) if phone is not None else 0.0
            t.phone_total_seconds = self._states.total_seconds(state, "PHONE_INTERACTION", now)
            # Same rule for screens: drawn while recently seen, recorded
            # with the longer grace.
            for stream, attr, key in (("monitor", "monitor_facing", "MONITOR_INTERACTION"),
                                      ("computer", "computer_in_use", "COMPUTER_INTERACTION")):
                sess = state.session(stream)
                recent = sess is not None and sess.last_confirmed is not None and now - sess.last_confirmed <= _PHONE_DISPLAY_HOLD_S
                setattr(t, attr, state.is_starting(stream) or (state.is_on(stream) and recent))
                setattr(t, f"{stream}_session_seconds", sess.current_duration(now) if sess is not None else 0.0)
                setattr(t, f"{stream}_total_seconds", self._states.total_seconds(state, key, now))
            t.movement_score = state.movement_score
            t.primary_activity = (
                str(state.primary.value) if state.primary is not None else "UNKNOWN"
            )
            # The main label follows the same on-screen rule as above: phone,
            # monitor or computer only while actually seen in the last few
            # seconds -- not through the recording grace, which kept "PHONE"
            # on someone who had already put the phone down -- else posture.
            for flag, stream, act in ((t.phone_in_use, "phone", "PHONE_INTERACTION"),
                                      (t.monitor_facing, "monitor", "MONITOR_INTERACTION"),
                                      (t.computer_in_use, "computer", "COMPUTER_INTERACTION")):
                if flag:
                    sess = state.session(stream)
                    t.activity = act
                    t.activity_seconds = sess.current_duration(now) if sess is not None else 0.0
                    break
            else:
                t.activity = t.primary_activity
                t.activity_seconds = (now - state.primary.since) if state.primary is not None else 0.0
            t.stationary_seconds = state.stationary_seconds(now)
            if state.person_name:
                t.name = state.person_name
                t.name_confidence = state.identity_confidence
            self._last_drawn[t.track_id] = (now, t)

    def _coast(self, activity_tracks: list, now: float) -> None:
        """Keep a person drawn for up to the session grace window after the
        detector loses them for a pass, at their last box. Display only --
        coasted tracks are appended AFTER the state update, so they never
        count as evidence."""
        hold = self._cfg.session_grace_seconds
        present = {t.track_id for t in activity_tracks}
        for tid, (seen_at, t) in list(self._last_drawn.items()):
            if tid in present:
                continue
            if now - seen_at > hold:
                del self._last_drawn[tid]
                continue
            state = self._states.state_for_track(tid)
            # Don't resurrect a box whose user is already drawn via another track.
            if state is not None and any(self._states.state_for_track(p) is state for p in present):
                continue
            ghost = copy.copy(t)
            ghost.frames_since_seen = max(1, getattr(t, "frames_since_seen", 0))
            activity_tracks.append(ghost)

    # -- reads ----------------------------------------------------------

    def rows(self, now: Optional[float] = None) -> list[dict]:
        """Live per-person activity rows."""
        with self._lock:
            return self._states.rows(now if now is not None else time.monotonic())

    def recent_events(self, limit: int = 100) -> list[dict]:
        """Chronological in-memory event log, newest last."""
        events = list(self._recent)
        return events[-max(1, limit):]

    def phone_violations(self, now: Optional[float] = None) -> list[dict]:
        """People using a phone right now. A confirmed phone use is a
        violation once its phone-use confidence has reached
        ACTIVITY_PHONE_VIOLATION_CONF, and is shown while the phone was seen
        in the last few seconds (not through the whole recording grace).
        For the live alert banner; the stored phone sessions are the record."""
        now = now if now is not None else time.monotonic()
        out = []
        with self._lock:
            for state in self._states.states.values():
                if not state.present or not state.is_on("phone"):
                    continue
                sess = state.session("phone")
                if sess.last_confirmed is None or now - sess.last_confirmed > _PHONE_DISPLAY_HOLD_S:
                    continue
                if getattr(state, "phone_peak_confidence", 0.0) < self._cfg.phone_violation_conf:
                    continue
                out.append({
                    "track_id": state.track_id,
                    "employee_id": state.person_id,
                    "name": state.person_name or "Unidentified person",
                    "since": sess.session_start_wall.isoformat() if sess.session_start_wall else None,
                    "seconds": round(sess.current_duration(now), 1),
                    "confidence": round(float(getattr(state, "phone_use_confidence", 0.0)), 2),
                })
        return sorted(out, key=lambda v: -v["seconds"])

    def needs_identity(self, track_id: int, now: Optional[float] = None) -> bool:
        """True when this person track has no user yet, or its identity is
        due for revalidation."""
        now = now if now is not None else time.monotonic()
        with self._lock:
            return self._identities.needs_recognition(track_id, now, self._cfg.reid_revalidate_seconds)

    @property
    def enabled_activities(self) -> list[str]:
        return sorted(self._states.enabled_activities)

    def set_enabled_activities(self, activities) -> None:
        """Admin changed the selection: closes (and stores) sessions of any
        activity that was just deselected."""
        with self._lock:
            self._record(self._states.set_enabled_activities(activities, time.monotonic()))

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        """Flush every open interval so a restart doesn't silently lose the
        time people had already accumulated."""
        try:
            with self._lock:
                events = self._states.close_all(time.monotonic())
        except Exception:
            logger.exception("[%s] Failed closing activity states", self.camera_id)
            return
        self._record(events)
        logger.info("[%s] Activity analyzer closed (%d open interval(s) flushed)",
                    self.camera_id, len(events))
