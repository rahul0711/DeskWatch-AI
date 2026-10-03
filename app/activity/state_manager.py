"""Temporal state engine: per-track activity states, timers and events.

This is the piece that turns per-frame observations into durations. The
detector is stateless by design; everything with a clock lives here.

Four INDEPENDENT streams are tracked per person, not one exclusive state:

    primary      SITTING | STANDING | WALKING | UNKNOWN   (posture + motion)
    stationary   on/off                                   (low movement)
    phone        on/off                                   (phone interaction)
    monitor      on/off                                   (monitor facing)

They overlap on purpose -- someone can be sitting, stationary and facing a
monitor at the same time, and reporting that as a single winner-takes-all
state would throw away most of the information. It also matches the event
vocabulary (SITTING_STARTED and PHONE_INTERACTION_STARTED are separate
events) and makes per-activity totals that legitimately sum to more than
the presence window.

Every stream is debounced: a change must hold for ACTIVITY_CONFIRMATION_SECONDS
before it is committed. Debouncing symmetrically (on->off as well as
off->on) is what stops one phone call from being chopped into five events by
a couple of frames where the hand occluded the phone.

Confidence on a closed event is measured, not invented: it is the fraction
of observations during that interval which actually voted for the state.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Callable, Hashable, Optional

logger = logging.getLogger(__name__)


class Activity:
    UNKNOWN = "UNKNOWN"
    SITTING = "SITTING"
    STANDING = "STANDING"
    WALKING = "WALKING"
    STATIONARY = "STATIONARY"
    PHONE_INTERACTION = "PHONE_INTERACTION"
    MONITOR_INTERACTION = "MONITOR_INTERACTION"
    # Hands on a laptop/keyboard/mouse. Kept distinct from
    # MONITOR_INTERACTION: a screen you face and a keyboard you touch are
    # different evidence, and the pre-existing dashboard already reported
    # this one.
    COMPUTER_INTERACTION = "COMPUTER_INTERACTION"
    PRESENCE = "PRESENCE"


# Label shown on the overlay / "current activity" column when several
# streams are active at once. Order is significance, not exclusivity --
# all of them keep their own timers regardless of what is displayed.
_DISPLAY_PRIORITY = (
    Activity.PHONE_INTERACTION,
    Activity.MONITOR_INTERACTION,
    Activity.COMPUTER_INTERACTION,
    Activity.WALKING,
)


@dataclass
class ActivityEvent:
    """One completed interval of one activity stream."""
    camera_id: str
    track_id: int
    activity_type: str
    start_time: dt.datetime
    end_time: dt.datetime
    duration_seconds: float
    confidence: float
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "activity_type": self.activity_type,
            "person_id": self.person_id,
            "person_name": self.person_name or "Unknown",
            "start_time": _iso(self.start_time),
            "end_time": _iso(self.end_time),
            "duration_seconds": round(self.duration_seconds, 1),
            "confidence": round(self.confidence, 3),
            "metadata": self.metadata,
        }


def _iso(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    return ts.isoformat()


class _Stream:
    """A debounced value with an open interval and vote accounting.

    `value` is what has been committed. A different value replaces it only
    once it has been the MAJORITY of observations across a window at least
    `confirm_seconds` long.

    Majority-over-a-window rather than an unbroken run: posture on a
    high-mounted camera genuinely alternates frame to frame when hips and
    knees are occluded by a desk, and a strict "must hold continuously"
    rule never commits at all in that case -- the candidate timer resets on
    every disagreeing frame, so the stream can sit on a stale value
    indefinitely even while the new one is observed most of the time.

    Votes are counted against the committed value so a closed interval can
    report how much of the time the evidence actually agreed with it.
    """

    __slots__ = (
        "value", "since", "since_wall", "_confirm", "_window",
        "_votes_for", "_votes_total",
    )

    # Share of the window a challenger must win to take over.
    _MAJORITY = 0.6

    def __init__(self, value: Hashable, now: float, now_wall: dt.datetime, confirm_seconds: float) -> None:
        self.value = value
        self.since = now
        self.since_wall = now_wall
        self._confirm = confirm_seconds
        # (timestamp, observed value) for the recent confirmation window.
        self._window: list[tuple[float, Hashable]] = []
        self._votes_for = 0
        self._votes_total = 0

    @property
    def confidence(self) -> float:
        if self._votes_total <= 0:
            return 0.0
        return self._votes_for / self._votes_total

    def observe(
        self, observed: Hashable, now: float, now_wall: dt.datetime
    ) -> Optional[tuple[Hashable, float, dt.datetime, float]]:
        """Record one observation.

        Returns None, or (old_value, duration_seconds, start_wall, confidence)
        when the committed value just changed, i.e. an interval closed.
        """
        self._votes_total += 1
        if observed == self.value:
            self._votes_for += 1

        self._window.append((now, observed))
        # Keep a little more than the confirmation window so the span test
        # below has something to measure.
        cutoff = now - max(self._confirm * 1.5, self._confirm + 1e-9)
        self._window = [w for w in self._window if w[0] >= cutoff]

        if observed == self.value:
            return None

        span = now - self._window[0][0]
        if span < self._confirm:
            return None

        challengers = [v for _, v in self._window if v == observed]
        if len(challengers) / len(self._window) < self._MAJORITY:
            return None

        # Back-date the switch to the challenger's first appearance inside
        # the window, so neither interval is credited with the other's time.
        cand_since = next(ts for ts, v in self._window if v == observed)

        # The interval ended when the NEW value was first observed, not now:
        # billing it up to the moment the debounce confirmed would add
        # confirm_seconds to every interval and overlap it with the next one.
        closed = (self.value, max(0.0, cand_since - self.since), self.since_wall, self.confidence)
        self.value = observed
        # ...and the new interval began there too, so the timeline has no
        # gap between them.
        self.since = cand_since
        self.since_wall = now_wall - dt.timedelta(seconds=max(0.0, now - cand_since))
        self._votes_for = 1
        self._votes_total = 1
        self._window = [(now, observed)]
        return closed

    def close(self, now: float) -> tuple[Hashable, float, dt.datetime, float]:
        return (self.value, now - self.since, self.since_wall, self.confidence)


@dataclass
class PersonState:
    """Live state for one tracked person. Never reset per frame."""
    track_id: int
    camera_id: str

    first_seen: float
    first_seen_wall: dt.datetime
    last_seen: float

    # Identity, attached from the face pipeline with a grace period so a
    # turned head or brief occlusion does not flip a person to Unknown.
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    identity_confidence: float = 0.0
    identity_updated_at: float = 0.0

    movement_score: float = 0.0
    last_movement_at: float = 0.0

    primary: Optional[_Stream] = None
    stationary: Optional[_Stream] = None
    phone: Optional[_Stream] = None
    monitor: Optional[_Stream] = None
    computer: Optional[_Stream] = None

    totals: dict = field(default_factory=lambda: {
        Activity.SITTING: 0.0,
        Activity.STANDING: 0.0,
        Activity.WALKING: 0.0,
        Activity.STATIONARY: 0.0,
        Activity.PHONE_INTERACTION: 0.0,
        Activity.MONITOR_INTERACTION: 0.0,
        Activity.COMPUTER_INTERACTION: 0.0,
    })
    posture: str = "unknown"
    head_orientation: str = "UNKNOWN"
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    monitor_bbox: Optional[tuple] = None
    monitor_distance: float = -1.0
    monitor_confidence: float = 0.0
    phone_hand_distance: float = -1.0
    phone_bbox: Optional[tuple] = None
    phone_confidence: float = 0.0
    # How many detection passes this track has been seen in. Used to
    # suppress events from ghost tracks: on a busy scene the person
    # detector briefly fires on chairs, reflections and partial bodies,
    # and BoT-SORT dutifully gives each one a track id. Those vanish after
    # a few passes, and without this gate each one writes a handful of
    # low-confidence rows.
    observations: int = 0

    @property
    def display_activity(self) -> str:
        if self.phone is not None and self.phone.value:
            return Activity.PHONE_INTERACTION
        if self.monitor is not None and self.monitor.value:
            return Activity.MONITOR_INTERACTION
        if self.computer is not None and self.computer.value:
            return Activity.COMPUTER_INTERACTION
        if self.primary is not None:
            return str(self.primary.value)
        return Activity.UNKNOWN

    def display_since(self, now: float) -> float:
        label = self.display_activity
        if label == Activity.PHONE_INTERACTION and self.phone is not None:
            return now - self.phone.since
        if label == Activity.MONITOR_INTERACTION and self.monitor is not None:
            return now - self.monitor.since
        if label == Activity.COMPUTER_INTERACTION and self.computer is not None:
            return now - self.computer.since
        if self.primary is not None:
            return now - self.primary.since
        return 0.0

    def stationary_seconds(self, now: float) -> float:
        if self.stationary is not None and self.stationary.value:
            return now - self.stationary.since
        return 0.0


class ActivityStateManager:
    """Owns every PersonState for one camera.

    update() is called once per activity detection pass with that pass's
    observations; it returns the events that closed on this pass.
    """

    def __init__(
        self,
        camera_id: str,
        confirmation_seconds: float = 2.0,
        identity_grace_seconds: float = 5.0,
        track_grace_seconds: float = 10.0,
        walking_threshold: float = 0.25,
        stationary_threshold: float = 0.06,
        stationary_min_seconds: float = 3.0,
        min_event_seconds: float = 2.0,
        min_track_observations: int = 8,
        max_tracks: int = 200,
        on_event: Optional[Callable[[ActivityEvent], None]] = None,
    ) -> None:
        self.camera_id = camera_id
        self._confirm = confirmation_seconds
        self._identity_grace = identity_grace_seconds
        self._track_grace = track_grace_seconds
        self._walking_threshold = walking_threshold
        self._stationary_threshold = stationary_threshold
        self._stationary_min = stationary_min_seconds
        self._min_event = min_event_seconds
        self._min_observations = min_track_observations
        self._max_tracks = max_tracks
        self._on_event = on_event
        self._states: dict[int, PersonState] = {}

    # -- public ---------------------------------------------------------

    @property
    def states(self) -> dict[int, PersonState]:
        return self._states

    def update(
        self,
        observations: list,
        movement_scores: dict[int, float],
        now: float,
        now_wall: Optional[dt.datetime] = None,
    ) -> list[ActivityEvent]:
        """Feed one detection pass. `observations` are TrackedFace objects
        from ActivityDetector (already carrying posture/phone/monitor), and
        `movement_scores` maps track_id -> smoothed movement score.
        """
        now_wall = now_wall or dt.datetime.now(dt.timezone.utc)
        events: list[ActivityEvent] = []

        for obs in observations:
            tid = obs.track_id
            state = self._states.get(tid)
            if state is None:
                state = self._create(tid, now, now_wall)
                events.extend(self._emit_entered(state, now_wall))

            state.last_seen = now
            state.observations += 1
            state.posture = obs.posture
            state.head_orientation = getattr(obs, "head_orientation", "UNKNOWN")
            state.head_yaw = getattr(obs, "head_yaw", 0.0)
            state.head_pitch = getattr(obs, "head_pitch", 0.0)
            state.monitor_bbox = obs.monitor_bbox
            state.monitor_distance = getattr(obs, "monitor_distance", -1.0)
            state.monitor_confidence = getattr(obs, "monitor_confidence", 0.0)
            state.phone_hand_distance = getattr(obs, "phone_hand_distance", -1.0)
            state.phone_bbox = getattr(obs, "phone_bbox", None)
            state.phone_confidence = getattr(obs, "phone_confidence", 0.0)
            score = movement_scores.get(tid, 0.0)
            state.movement_score = score
            if score >= self._stationary_threshold:
                state.last_movement_at = now

            # --- primary posture/motion stream --------------------------
            if score >= self._walking_threshold:
                primary_obs = Activity.WALKING
            elif obs.posture == "sitting":
                primary_obs = Activity.SITTING
            elif obs.posture == "standing":
                primary_obs = Activity.STANDING
            else:
                primary_obs = Activity.UNKNOWN
            events.extend(self._observe(state, "primary", primary_obs, now, now_wall))

            # --- stationary stream --------------------------------------
            # Only becomes true once movement has stayed low for
            # stationary_min_seconds; the timer then runs until real
            # movement resumes. Tolerance-based, never requiring zero
            # pixel movement.
            low_for = now - state.last_movement_at
            stationary_obs = (
                score < self._stationary_threshold and low_for >= self._stationary_min
            )
            events.extend(self._observe(state, "stationary", stationary_obs, now, now_wall))

            # --- interaction streams ------------------------------------
            events.extend(self._observe(state, "phone", bool(obs.phone_in_use), now, now_wall))
            events.extend(self._observe(state, "monitor", bool(obs.monitor_facing), now, now_wall))
            events.extend(self._observe(state, "computer", bool(obs.computer_in_use), now, now_wall))

        events.extend(self._expire(now, now_wall))
        self._cap()
        return events

    def apply_identity(
        self,
        track_id: int,
        person_id: Optional[str],
        person_name: str,
        confidence: float,
        now: float,
    ) -> None:
        """Attach/refresh an identity on a track (see identity_binder)."""
        state = self._states.get(track_id)
        if state is None:
            return
        if state.person_name != person_name and person_name:
            logger.info(
                "[%s] Identity assigned: track=%s -> %s (conf=%.2f)",
                self.camera_id, track_id, person_name, confidence,
            )
        state.person_id = person_id or state.person_id
        state.person_name = person_name or state.person_name
        state.identity_confidence = confidence
        state.identity_updated_at = now

    def identity_is_stale(self, track_id: int, now: float) -> bool:
        state = self._states.get(track_id)
        if state is None or state.person_name is None:
            return True
        return (now - state.identity_updated_at) > self._identity_grace

    def close_all(self, now: float, now_wall: Optional[dt.datetime] = None) -> list[ActivityEvent]:
        """Flush every open interval -- called on shutdown."""
        now_wall = now_wall or dt.datetime.now(dt.timezone.utc)
        events: list[ActivityEvent] = []
        for tid in list(self._states):
            events.extend(self._close_track(tid, now, now_wall))
        return events

    def rows(self, now: float) -> list[dict]:
        """Live per-person rows for the API/table."""
        out = []
        for state in self._states.values():
            out.append({
                "track_id": state.track_id,
                "person_id": state.person_id,
                "person_name": state.person_name or "Unknown",
                "identity_confidence": round(state.identity_confidence, 3),
                "activity": state.display_activity,
                "activity_seconds": round(state.display_since(now), 1),
                "posture": state.posture,
                "head_orientation": state.head_orientation,
                "head_yaw": round(state.head_yaw, 3),
                "head_pitch": round(state.head_pitch, 3),
                "movement_score": round(state.movement_score, 4),
                "phone": bool(state.phone and state.phone.value),
                "phone_hand_distance": round(state.phone_hand_distance, 1),
                "phone_confidence": round(state.phone_confidence, 3),
                "monitor": bool(state.monitor and state.monitor.value),
                "monitor_distance": round(state.monitor_distance, 1),
                "monitor_confidence": round(state.monitor_confidence, 3),
                "computer": bool(state.computer and state.computer.value),
                "stationary_seconds": round(state.stationary_seconds(now), 1),
                "first_seen": _iso(state.first_seen_wall),
                "present_seconds": round(now - state.first_seen, 1),
                "sitting_seconds": round(self._total(state, Activity.SITTING, now), 1),
                "standing_seconds": round(self._total(state, Activity.STANDING, now), 1),
                "walking_seconds": round(self._total(state, Activity.WALKING, now), 1),
                "low_movement_seconds": round(self._total(state, Activity.STATIONARY, now), 1),
                "phone_seconds": round(self._total(state, Activity.PHONE_INTERACTION, now), 1),
                "monitor_seconds": round(self._total(state, Activity.MONITOR_INTERACTION, now), 1),
                "computer_seconds": round(self._total(state, Activity.COMPUTER_INTERACTION, now), 1),
            })
        return out

    # -- internal -------------------------------------------------------

    def _create(self, track_id: int, now: float, now_wall: dt.datetime) -> PersonState:
        state = PersonState(
            track_id=track_id,
            camera_id=self.camera_id,
            first_seen=now,
            first_seen_wall=now_wall,
            last_seen=now,
            last_movement_at=now,
        )
        state.primary = _Stream(Activity.UNKNOWN, now, now_wall, self._confirm)
        state.stationary = _Stream(False, now, now_wall, self._confirm)
        state.phone = _Stream(False, now, now_wall, self._confirm)
        state.monitor = _Stream(False, now, now_wall, self._confirm)
        state.computer = _Stream(False, now, now_wall, self._confirm)
        self._states[track_id] = state
        return state

    _STREAM_ACTIVITY = {
        "stationary": Activity.STATIONARY,
        "phone": Activity.PHONE_INTERACTION,
        "monitor": Activity.MONITOR_INTERACTION,
        "computer": Activity.COMPUTER_INTERACTION,
    }

    def _observe(
        self, state: PersonState, stream_name: str, observed, now: float, now_wall: dt.datetime
    ) -> list[ActivityEvent]:
        stream: _Stream = getattr(state, stream_name)
        closed = stream.observe(observed, now, now_wall)
        if closed is None:
            return []
        old_value, duration, start_wall, confidence = closed
        activity = self._activity_name(stream_name, old_value)
        if activity is None:
            return []
        # Totals count every second observed, including intervals too short
        # to be worth storing as an event.
        state.totals[activity] = state.totals.get(activity, 0.0) + duration
        event = self._make_event(
            state, activity, start_wall, duration, confidence,
            metadata={"stream": stream_name},
        )
        if event is None:
            return []
        logger.info(
            "[%s] %s: %s %s -> %s (%.0fs)",
            self.camera_id, state.person_name or f"track={state.track_id}",
            "activity changed" if stream_name == "primary" else stream_name,
            activity, observed, duration,
        )
        return [event]

    def _activity_name(self, stream_name: str, value) -> Optional[str]:
        if stream_name == "primary":
            return None if value == Activity.UNKNOWN else str(value)
        # Boolean streams only produce an event for the "on" interval.
        return self._STREAM_ACTIVITY[stream_name] if value else None

    def _make_event(
        self,
        state: PersonState,
        activity: str,
        start_wall: dt.datetime,
        duration: float,
        confidence: float,
        metadata: dict,
    ) -> Optional[ActivityEvent]:
        if duration < self._min_event:
            return None
        if state.observations < self._min_observations:
            # Ghost track -- time still counts toward its live totals, but
            # nothing is written or broadcast for it.
            return None
        event = ActivityEvent(
            camera_id=self.camera_id,
            track_id=state.track_id,
            activity_type=activity,
            start_time=start_wall,
            end_time=start_wall + dt.timedelta(seconds=duration),
            duration_seconds=duration,
            confidence=confidence,
            person_id=state.person_id,
            person_name=state.person_name,
            metadata=metadata,
        )
        if self._on_event is not None:
            self._on_event(event)
        return event

    def _total(self, state: PersonState, activity: str, now: float) -> float:
        """Completed time for this activity plus whatever its currently-open
        interval has accrued, so a live row counts the ongoing sit."""
        total = state.totals.get(activity, 0.0)
        if state.primary is not None and str(state.primary.value) == activity:
            total += max(0.0, now - state.primary.since)
        for name, act in self._STREAM_ACTIVITY.items():
            if act != activity:
                continue
            stream: _Stream = getattr(state, name)
            if stream is not None and stream.value:
                total += max(0.0, now - stream.since)
        return total

    def _emit_entered(self, state: PersonState, now_wall: dt.datetime) -> list[ActivityEvent]:
        logger.info("[%s] Person entered: track=%s", self.camera_id, state.track_id)
        return []

    def _expire(self, now: float, now_wall: dt.datetime) -> list[ActivityEvent]:
        events: list[ActivityEvent] = []
        for tid, state in list(self._states.items()):
            if now - state.last_seen > self._track_grace:
                events.extend(self._close_track(tid, now, now_wall))
        return events

    def _close_track(self, track_id: int, now: float, now_wall: dt.datetime) -> list[ActivityEvent]:
        state = self._states.pop(track_id, None)
        if state is None:
            return []
        events: list[ActivityEvent] = []
        # Durations end at last_seen, not now -- the grace period is for
        # deciding the person left, not time they were observed.
        end = state.last_seen
        for name in ("primary", "stationary", "phone", "monitor", "computer"):
            stream: _Stream = getattr(state, name)
            if stream is None:
                continue
            value, _duration, start_wall, confidence = stream.close(end)
            activity = self._activity_name(name, value)
            if activity is None:
                continue
            duration = max(0.0, end - stream.since)
            state.totals[activity] = state.totals.get(activity, 0.0) + duration
            event = self._make_event(
                state, activity, start_wall, duration, confidence,
                metadata={"stream": name, "closed_by": "track_lost"},
            )
            if event is not None:
                events.append(event)

        presence = max(0.0, end - state.first_seen)
        presence_event = self._make_event(
            state, Activity.PRESENCE, state.first_seen_wall, presence, 1.0,
            metadata={"closed_by": "track_lost"},
        )
        if presence_event is not None:
            events.append(presence_event)
        logger.info(
            "[%s] Person left: track=%s person=%s present=%.0fs",
            self.camera_id, track_id, state.person_name or "Unknown", presence,
        )
        return events

    def _cap(self) -> None:
        while len(self._states) > self._max_tracks:
            oldest = min(self._states, key=lambda t: self._states[t].first_seen)
            del self._states[oldest]
