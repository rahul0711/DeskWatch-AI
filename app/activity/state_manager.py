"""Temporal state engine: per-USER activity states, sessions and totals.

This is the piece that turns per-frame observations into durations. The
detector is stateless by design; everything with a clock lives here.

Activities belong to a USER, not to a tracker id. A track id is only
short-term visual continuity -- it dies on occlusion and comes back as a
new number -- so state is keyed by a *subject*:

    "user:<person_id>"  once identity is known (identity_cache.py), or
    "track:<track_id>"  for a person not identified (yet).

Each track id is an alias pointing at a subject. When an anonymous track is
identified it is merged into the user's existing state, and when a new
track (after occlusion) is re-identified as the same user it simply points
at the same state -- totals keep accumulating and are never reset because
the tracker handed out a new number.

Per subject, INDEPENDENT streams are tracked, not one exclusive state:

    primary      SITTING | STANDING | WALKING | UNKNOWN   (posture + motion)
    stationary   ActivitySession                          (low movement)
    phone        ActivitySession                          (phone interaction)
    monitor      ActivitySession                          (monitor facing)
    computer     ActivitySession                          (hands on a PC)

They overlap on purpose -- someone can be sitting, stationary and facing a
monitor at the same time. The interaction streams use ActivitySession
(sessions.py): ~1.5s of stable evidence to activate, a grace window that
tolerates brief detection loss, and an accumulated total that survives
every new session. Posture keeps its majority-over-window debounce
(_Stream), which suits a mutually exclusive state on a camera where
hips/knees are often occluded.

Only the admin-selected activities (settings.py) produce sessions, totals
and stored rows.

Confidence on a closed session is measured, not invented: it is the
fraction of observations during that session which actually voted for it.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Callable, Hashable, Iterable, Optional

from app.activity.sessions import ActivitySession, ClosedSession, SessionState

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


# Every activity that can be selected for tracking/storage.
ALL_ACTIVITIES = (
    Activity.SITTING,
    Activity.STANDING,
    Activity.WALKING,
    Activity.STATIONARY,
    Activity.PHONE_INTERACTION,
    Activity.MONITOR_INTERACTION,
    Activity.COMPUTER_INTERACTION,
)
_POSTURE_ACTIVITIES = (Activity.SITTING, Activity.STANDING, Activity.WALKING)
# Boolean streams -> activity type, all backed by an ActivitySession.
_SESSION_STREAMS = {
    "stationary": Activity.STATIONARY,
    "phone": Activity.PHONE_INTERACTION,
    "monitor": Activity.MONITOR_INTERACTION,
    "computer": Activity.COMPUTER_INTERACTION,
}

_ANON_PREFIX = "track:"
_USER_PREFIX = "user:"


def anon_subject(track_id: int) -> str:
    return f"{_ANON_PREFIX}{track_id}"


def user_subject(person_id: str) -> str:
    return f"{_USER_PREFIX}{person_id}"


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
    """Live state for one SUBJECT (a user, or a not-yet-identified track).
    Never reset per frame, and for users never reset when the tracker hands
    out a new track id."""
    subject_id: str
    camera_id: str
    # Most recent track id seen for this subject (display / debugging only).
    track_id: int

    first_seen: float
    first_seen_wall: dt.datetime
    last_seen: float

    # Every track id currently aliased to this subject.
    track_ids: set = field(default_factory=set)
    # False once the subject has been unseen past track_grace_seconds. A
    # user's state (and totals) is kept while absent; an anonymous one is
    # dropped.
    present: bool = True

    person_id: Optional[str] = None
    person_name: Optional[str] = None
    identity_confidence: float = 0.0
    identity_updated_at: float = 0.0
    identity_source: Optional[str] = None

    movement_score: float = 0.0
    last_movement_at: float = 0.0

    primary: Optional[_Stream] = None
    sessions: dict = field(default_factory=dict)   # activity type -> ActivitySession

    # Closed posture time this process has observed.
    totals: dict = field(default_factory=lambda: {a: 0.0 for a in _POSTURE_ACTIVITIES})
    # Time already stored in the database before this process first saw the
    # user today, so a restart does not reset the dashboard totals.
    baseline: dict = field(default_factory=dict)

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
    # How many detection passes this subject has been seen in. Used to
    # suppress events from ghost tracks: on a busy scene the person
    # detector briefly fires on chairs, reflections and partial bodies,
    # and the tracker dutifully gives each one an id.
    observations: int = 0

    @property
    def is_user(self) -> bool:
        return self.subject_id.startswith(_USER_PREFIX)

    def session(self, stream_name: str) -> Optional[ActivitySession]:
        return self.sessions.get(_SESSION_STREAMS[stream_name])

    # Stream-name accessors, kept for callers that read e.g. state.phone.
    @property
    def stationary(self) -> Optional[ActivitySession]:
        return self.session("stationary")

    @property
    def phone(self) -> Optional[ActivitySession]:
        return self.session("phone")

    @property
    def monitor(self) -> Optional[ActivitySession]:
        return self.session("monitor")

    @property
    def computer(self) -> Optional[ActivitySession]:
        return self.session("computer")

    def is_on(self, stream_name: str) -> bool:
        s = self.session(stream_name)
        return bool(s is not None and s.is_on)

    def is_starting(self, stream_name: str) -> bool:
        """Evidence seen, session not confirmed yet (nothing recorded)."""
        s = self.session(stream_name)
        return bool(s is not None and s.current_state == SessionState.PENDING)

    @property
    def display_activity(self) -> str:
        if self.is_on("phone"):
            return Activity.PHONE_INTERACTION
        if self.is_on("monitor"):
            return Activity.MONITOR_INTERACTION
        if self.is_on("computer"):
            return Activity.COMPUTER_INTERACTION
        if self.primary is not None:
            return str(self.primary.value)
        return Activity.UNKNOWN

    def display_since(self, now: float) -> float:
        label = self.display_activity
        for name, act in _SESSION_STREAMS.items():
            if label == act and self.is_on(name):
                return self.session(name).current_duration(now)
        if self.primary is not None:
            return now - self.primary.since
        return 0.0

    def stationary_seconds(self, now: float) -> float:
        s = self.stationary
        return s.current_duration(now) if s is not None and s.is_on else 0.0


class ActivityStateManager:
    """Owns every subject's state for one camera.

    update() is called once per activity detection pass with that pass's
    observations (and the identities known for its tracks); it returns the
    events (closed sessions/intervals) produced on this pass.
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
        activation_seconds: float = 1.5,
        session_grace_seconds: float = 2.0,
        phone_grace_seconds: Optional[float] = None,
        phone_activation_seconds: Optional[float] = None,
        screen_grace_seconds: Optional[float] = None,
        enabled_activities: Optional[Iterable[str]] = None,
        baseline_loader: Optional[Callable[[str], dict]] = None,
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
        self._activation = activation_seconds
        self._session_grace = session_grace_seconds
        self._phone_grace = session_grace_seconds if phone_grace_seconds is None else phone_grace_seconds
        self._phone_activation = activation_seconds if phone_activation_seconds is None else phone_activation_seconds
        self._screen_grace = session_grace_seconds if screen_grace_seconds is None else screen_grace_seconds
        self._enabled: set[str] = set(enabled_activities) if enabled_activities is not None else set(ALL_ACTIVITIES)
        # person_id -> {activity: seconds already stored today}
        self._baseline_loader = baseline_loader
        self._states: dict[str, PersonState] = {}
        self._track_subject: dict[int, str] = {}

    # -- public ---------------------------------------------------------

    @property
    def states(self) -> dict[str, PersonState]:
        return self._states

    @property
    def enabled_activities(self) -> set[str]:
        return set(self._enabled)

    def subject_for_track(self, track_id: int) -> Optional[str]:
        return self._track_subject.get(track_id)

    def state_for_track(self, track_id: int) -> Optional[PersonState]:
        subject = self._track_subject.get(track_id)
        return self._states.get(subject) if subject is not None else None

    def update(
        self,
        observations: list,
        movement_scores: dict[int, float],
        now: float,
        now_wall: Optional[dt.datetime] = None,
        identities: Optional[dict] = None,
    ) -> list[ActivityEvent]:
        """Feed one detection pass.

        observations    -- TrackedFace objects from ActivityDetector.
        movement_scores -- track_id -> smoothed movement score.
        identities      -- track_id -> binding (anything with person_id,
                           person_name, confidence[, source]) for the tracks
                           identity_cache.py could identify.
        """
        now_wall = now_wall or dt.datetime.now(dt.timezone.utc)
        events: list[ActivityEvent] = []

        # 1. Resolve identity BEFORE accumulating, so this pass's evidence
        #    already lands on the user's state.
        for tid, b in (identities or {}).items():
            events.extend(self._bind_track(
                tid, str(b.person_id), b.person_name, float(b.confidence or 0.0),
                getattr(b, "source", None), now, now_wall,
            ))

        # 1b. A track still pointing at a user that another track in this
        #     pass is now bound to has lost that identity (identity_cache.py
        #     keeps one track per user). Detach it, or step 2 would fold a
        #     different person's activity -- their phone, their posture --
        #     into that user's state.
        if identities:
            observed = {obs.track_id for obs in observations}
            owners: dict[str, set[int]] = {}
            for tid, b in identities.items():
                if tid in observed:
                    owners.setdefault(user_subject(str(b.person_id)), set()).add(tid)
            for tid in observed:
                subject = self._track_subject.get(tid)
                if subject in owners and tid not in owners[subject]:
                    state = self._states.get(subject)
                    if state is not None:
                        state.track_ids.discard(tid)
                    self._track_subject[tid] = anon_subject(tid)
                    logger.info("[%s] track=%s detached from %s", self.camera_id, tid, subject)

        # 2. Group this pass's tracks by subject: two tracks of the same
        #    user in one pass (duplicate box, overlap during an id switch)
        #    must count once, not twice.
        grouped: dict[str, list] = {}
        for obs in observations:
            subject = self._track_subject.get(obs.track_id)
            if subject is None:
                subject = anon_subject(obs.track_id)
                self._track_subject[obs.track_id] = subject
            grouped.setdefault(subject, []).append(obs)

        for subject, obs_list in grouped.items():
            state = self._states.get(subject)
            if state is None:
                state = self._create(subject, obs_list[0].track_id, now, now_wall)
            elif not state.present:
                self._revisit(state, now, now_wall)
            events.extend(self._observe_subject(state, obs_list, movement_scores, now, now_wall))

        # 3. Subjects with no track this pass: absence for grace purposes.
        for subject, state in list(self._states.items()):
            if subject in grouped or not state.present:
                continue
            for act, sess in state.sessions.items():
                closed = sess.tick(now)
                if closed is not None:
                    events.extend(self._emit(state, closed, {"closed_by": "absent"}))

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
        now_wall: Optional[dt.datetime] = None,
    ) -> list[ActivityEvent]:
        """Attach/refresh an identity on a track (moves it onto the user)."""
        if not person_name:
            return []
        return self._bind_track(
            track_id, str(person_id or person_name), person_name, confidence, None,
            now, now_wall or dt.datetime.now(dt.timezone.utc),
        )

    def identity_is_stale(self, track_id: int, now: float) -> bool:
        state = self.state_for_track(track_id)
        if state is None or state.person_name is None:
            return True
        return (now - state.identity_updated_at) > self._identity_grace

    def set_enabled_activities(
        self, activities: Iterable[str], now: float, now_wall: Optional[dt.datetime] = None,
    ) -> list[ActivityEvent]:
        """Change which activities are tracked. Newly disabled activities
        have their open sessions closed (and stored); newly enabled ones
        start accumulating on the next pass."""
        now_wall = now_wall or dt.datetime.now(dt.timezone.utc)
        new = {a for a in activities if a in ALL_ACTIVITIES}
        removed = self._enabled - new
        self._enabled = new
        events: list[ActivityEvent] = []
        for state in self._states.values():
            for act in removed:
                sess = state.sessions.pop(act, None)
                if sess is not None:
                    closed = sess.close()
                    if closed is not None:
                        events.extend(self._emit(state, closed, {"closed_by": "disabled"}))
                if act in _POSTURE_ACTIVITIES and state.primary is not None and str(state.primary.value) == act:
                    events.extend(self._close_primary(state, state.last_seen, {"closed_by": "disabled"}))
                    state.primary = _Stream(Activity.UNKNOWN, now, now_wall, self._confirm)
            self._ensure_sessions(state)
        return events

    def close_all(self, now: float, now_wall: Optional[dt.datetime] = None) -> list[ActivityEvent]:
        """Flush every open session -- called on shutdown."""
        now_wall = now_wall or dt.datetime.now(dt.timezone.utc)
        events: list[ActivityEvent] = []
        for state in list(self._states.values()):
            if state.present:
                events.extend(self._leave(state, "shutdown"))
        return events

    def rows(self, now: float) -> list[dict]:
        """Live per-USER rows for the API/table (anonymous subjects too)."""
        out = []
        for state in self._states.values():
            row = {
                "subject_id": state.subject_id,
                "track_id": state.track_id,
                "track_ids": sorted(state.track_ids),
                "user_id": state.person_id,
                "person_id": state.person_id,
                "person_name": state.person_name or "Unknown",
                "identity_confidence": round(state.identity_confidence, 3),
                "identity_source": state.identity_source,
                "present": state.present,
                "activity": state.display_activity if state.present else "ABSENT",
                "activity_seconds": round(state.display_since(now), 1) if state.present else 0.0,
                "posture": state.posture,
                # Posture/motion alone (what "Body tracking" shows), without
                # phone/monitor/computer masking it.
                "primary_activity": (
                    str(state.primary.value) if state.present and state.primary is not None else "ABSENT"
                ),
                "primary_seconds": (
                    round(now - state.primary.since, 1) if state.present and state.primary is not None else 0.0
                ),
                "head_orientation": state.head_orientation,
                "head_yaw": round(state.head_yaw, 3),
                "head_pitch": round(state.head_pitch, 3),
                "movement_score": round(state.movement_score, 4),
                "phone": state.present and state.is_on("phone"),
                "phone_hand_distance": round(state.phone_hand_distance, 1),
                "phone_confidence": round(state.phone_confidence, 3),
                "monitor": state.present and state.is_on("monitor"),
                "monitor_distance": round(state.monitor_distance, 1),
                "monitor_confidence": round(state.monitor_confidence, 3),
                "computer": state.present and state.is_on("computer"),
                "stationary_seconds": round(state.stationary_seconds(now), 1) if state.present else 0.0,
                "first_seen": _iso(state.first_seen_wall),
                "present_seconds": round(now - state.first_seen, 1) if state.present else 0.0,
                "enabled_activities": sorted(self._enabled),
                "sitting_seconds": round(self._total(state, Activity.SITTING, now), 1),
                "standing_seconds": round(self._total(state, Activity.STANDING, now), 1),
                "walking_seconds": round(self._total(state, Activity.WALKING, now), 1),
                "low_movement_seconds": round(self._total(state, Activity.STATIONARY, now), 1),
                "phone_seconds": round(self._total(state, Activity.PHONE_INTERACTION, now), 1),
                "monitor_seconds": round(self._total(state, Activity.MONITOR_INTERACTION, now), 1),
                "computer_seconds": round(self._total(state, Activity.COMPUTER_INTERACTION, now), 1),
            }
            row["sessions"] = {
                act: {
                    "current_state": sess.current_state,
                    "session_start": _iso(sess.session_start_wall) if sess.session_start_wall else None,
                    "last_confirmed": _iso(sess.last_confirmed_wall) if sess.last_confirmed_wall else None,
                    "accumulated_duration": round(sess.accumulated_duration + state.baseline.get(act, 0.0), 1),
                    "sessions_closed": sess.sessions_closed,
                }
                for act, sess in state.sessions.items()
            }
            out.append(row)
        return out

    # -- identity / subjects ----------------------------------------------

    def _bind_track(
        self, track_id: int, person_id: str, person_name: str, confidence: float,
        source: Optional[str], now: float, now_wall: dt.datetime,
    ) -> list[ActivityEvent]:
        events: list[ActivityEvent] = []
        target = user_subject(person_id)
        current = self._track_subject.get(track_id)

        if current != target:
            cur_state = self._states.get(current) if current is not None else None
            user_state = self._states.get(target)
            if cur_state is not None and not cur_state.is_user:
                if user_state is None:
                    # First time we know who this is: the anonymous state
                    # simply becomes the user's state.
                    del self._states[current]
                    cur_state.subject_id = target
                    self._states[target] = cur_state
                    self._load_baseline(cur_state, person_id)
                    for tid in cur_state.track_ids:
                        self._track_subject[tid] = target
                    logger.info("[%s] track=%s identified as %s", self.camera_id, track_id, person_name)
                else:
                    # A new track for a user we already have (after
                    # occlusion / re-detection): fold it into that user.
                    if not user_state.present:
                        self._revisit(user_state, now, now_wall)
                    events.extend(self._merge(cur_state, user_state, now, now_wall))
                    del self._states[current]
                    for tid in cur_state.track_ids:
                        self._track_subject[tid] = target
                    logger.info(
                        "[%s] track=%s reconnected to existing user %s (tracks %s)",
                        self.camera_id, track_id, person_name, sorted(user_state.track_ids),
                    )
            else:
                # Unknown track, or a tracker id that switched from one
                # user onto another: re-point only this track id.
                if cur_state is not None:
                    cur_state.track_ids.discard(track_id)
                if user_state is None:
                    user_state = self._create(target, track_id, now, now_wall)
                    self._load_baseline(user_state, person_id)
                elif not user_state.present:
                    self._revisit(user_state, now, now_wall)
                user_state.track_ids.add(track_id)
                self._track_subject[track_id] = target

        state = self._states[target]
        if state.person_name != person_name:
            logger.info(
                "[%s] Identity assigned: track=%s -> %s (conf=%.2f, %s)",
                self.camera_id, track_id, person_name, confidence, source or "explicit",
            )
        state.person_id = person_id
        state.person_name = person_name
        state.identity_confidence = confidence
        state.identity_updated_at = now
        state.identity_source = source
        return events

    def _merge(self, anon: PersonState, user: PersonState, now: float, now_wall: dt.datetime) -> list[ActivityEvent]:
        """Fold an anonymous subject into a user's state."""
        events: list[ActivityEvent] = []
        user.track_ids |= anon.track_ids
        user.observations += anon.observations
        user.first_seen = min(user.first_seen, anon.first_seen)
        if anon.first_seen_wall < user.first_seen_wall:
            user.first_seen_wall = anon.first_seen_wall
        if anon.last_seen > user.last_seen:
            user.last_seen = anon.last_seen
            user.track_id = anon.track_id

        for act, sess in anon.sessions.items():
            if act in user.sessions:
                user.sessions[act].absorb(sess)
            else:
                user.sessions[act] = sess
        for act, secs in anon.totals.items():
            user.totals[act] = user.totals.get(act, 0.0) + secs

        # Posture: keep the user's running interval unless the new track
        # has already committed to a different posture.
        if anon.primary is not None and anon.primary.value not in (Activity.UNKNOWN,) and (
            user.primary is None or anon.primary.value != user.primary.value
        ):
            if user.primary is not None:
                events.extend(self._close_primary(user, anon.primary.since, {"closed_by": "reconnected"}))
            user.primary = anon.primary
        return events

    def _create(self, subject: str, track_id: int, now: float, now_wall: dt.datetime) -> PersonState:
        state = PersonState(
            subject_id=subject,
            camera_id=self.camera_id,
            track_id=track_id,
            first_seen=now,
            first_seen_wall=now_wall,
            last_seen=now,
            last_movement_at=now,
        )
        state.track_ids.add(track_id)
        state.primary = _Stream(Activity.UNKNOWN, now, now_wall, self._confirm)
        self._ensure_sessions(state)
        self._states[subject] = state
        logger.info("[%s] Person entered: %s", self.camera_id, subject)
        return state

    def _revisit(self, state: PersonState, now: float, now_wall: dt.datetime) -> None:
        """A user who had left is back: new visit, same accumulated totals."""
        state.present = True
        state.first_seen, state.first_seen_wall = now, now_wall
        state.last_seen = now
        state.last_movement_at = now
        state.primary = _Stream(Activity.UNKNOWN, now, now_wall, self._confirm)
        logger.info("[%s] %s is back", self.camera_id, state.person_name or state.subject_id)

    def _ensure_sessions(self, state: PersonState) -> None:
        for act in _SESSION_STREAMS.values():
            if act in self._enabled and act not in state.sessions:
                if act == Activity.PHONE_INTERACTION:
                    state.sessions[act] = ActivitySession(act, self._phone_activation, self._phone_grace)
                elif act in (Activity.MONITOR_INTERACTION, Activity.COMPUTER_INTERACTION):
                    state.sessions[act] = ActivitySession(act, self._activation, self._screen_grace)
                else:
                    state.sessions[act] = ActivitySession(act, self._activation, self._session_grace)

    def _load_baseline(self, state: PersonState, person_id: str) -> None:
        if self._baseline_loader is None or state.baseline:
            return
        try:
            state.baseline = dict(self._baseline_loader(person_id) or {})
        except Exception as exc:
            logger.warning("[%s] Could not load stored totals for %s: %s", self.camera_id, person_id, exc)

    # -- per-pass accumulation ---------------------------------------------

    def _observe_subject(
        self, state: PersonState, obs_list: list, movement_scores: dict, now: float, now_wall: dt.datetime,
    ) -> list[ActivityEvent]:
        events: list[ActivityEvent] = []
        # The most confident box speaks for posture/geometry; interaction
        # evidence from any of the subject's tracks counts.
        obs = max(obs_list, key=lambda o: getattr(getattr(o, "face", None), "confidence", 0.0))
        state.track_id = obs.track_id
        state.track_ids.update(o.track_id for o in obs_list)
        state.last_seen = now
        state.observations += 1
        state.posture = obs.posture
        state.head_orientation = getattr(obs, "head_orientation", "UNKNOWN")
        state.head_yaw = getattr(obs, "head_yaw", 0.0)
        state.head_pitch = getattr(obs, "head_pitch", 0.0)
        state.monitor_bbox = obs.monitor_bbox
        state.monitor_distance = getattr(obs, "monitor_distance", -1.0)
        state.monitor_confidence = getattr(obs, "monitor_confidence", 0.0)
        phone_obs = next((o for o in obs_list if o.phone_in_use), obs)
        state.phone_hand_distance = getattr(phone_obs, "phone_hand_distance", -1.0)
        state.phone_bbox = getattr(phone_obs, "phone_bbox", None)
        state.phone_confidence = getattr(phone_obs, "phone_confidence", 0.0)

        score = max(movement_scores.get(o.track_id, 0.0) for o in obs_list)
        state.movement_score = score
        if score >= self._stationary_threshold:
            state.last_movement_at = now

        # --- primary posture/motion stream ------------------------------
        if score >= self._walking_threshold:
            primary_obs = Activity.WALKING
        elif obs.posture == "sitting":
            primary_obs = Activity.SITTING
        elif obs.posture == "standing":
            primary_obs = Activity.STANDING
        else:
            primary_obs = Activity.UNKNOWN
        closed = state.primary.observe(primary_obs, now, now_wall)
        if closed is not None:
            old_value, duration, start_wall, confidence = closed
            events.extend(self._credit_primary(state, old_value, duration, start_wall, confidence, {"stream": "primary"}))

        # --- session streams ----------------------------------------------
        # Stationary only counts once movement has stayed low for
        # stationary_min_seconds -- tolerance-based, never zero movement.
        low_for = now - state.last_movement_at
        evidence = {
            Activity.STATIONARY: score < self._stationary_threshold and low_for >= self._stationary_min,
            Activity.PHONE_INTERACTION: any(o.phone_in_use for o in obs_list),
            Activity.MONITOR_INTERACTION: any(o.monitor_facing for o in obs_list),
            Activity.COMPUTER_INTERACTION: any(o.computer_in_use for o in obs_list),
        }
        for act, sess in state.sessions.items():
            closed_s = sess.observe(evidence.get(act, False), now, now_wall)
            if closed_s is not None:
                events.extend(self._emit(state, closed_s, {"stream": act}))
        return events

    def _credit_primary(
        self, state: PersonState, value, duration: float, start_wall: dt.datetime, confidence: float, metadata: dict,
    ) -> list[ActivityEvent]:
        activity = None if value == Activity.UNKNOWN else str(value)
        if activity is None or activity not in self._enabled:
            return []
        # Totals count every second observed, including intervals too short
        # to be worth storing.
        state.totals[activity] = state.totals.get(activity, 0.0) + duration
        closed = ClosedSession(activity, start_wall, start_wall + dt.timedelta(seconds=duration), duration, confidence)
        return self._emit(state, closed, metadata, add_to_totals=False)

    def _close_primary(self, state: PersonState, end: float, metadata: dict) -> list[ActivityEvent]:
        if state.primary is None:
            return []
        value, _d, start_wall, confidence = state.primary.close(end)
        duration = max(0.0, end - state.primary.since)
        return self._credit_primary(state, value, duration, start_wall, confidence, metadata)

    def _emit(self, state: PersonState, closed: ClosedSession, metadata: dict, add_to_totals: bool = True) -> list[ActivityEvent]:
        """Turn a closed session into a stored event, unless it is too short
        or comes from a ghost track. (Session totals were already updated by
        ActivitySession itself.)"""
        if closed.duration_seconds < self._min_event:
            return []
        if state.observations < self._min_observations:
            return []
        event = ActivityEvent(
            camera_id=self.camera_id,
            track_id=state.track_id,
            activity_type=closed.activity_type,
            start_time=closed.start_wall,
            end_time=closed.end_wall,
            duration_seconds=closed.duration_seconds,
            confidence=closed.confidence,
            person_id=state.person_id,
            person_name=state.person_name,
            metadata={**metadata, "track_ids": sorted(state.track_ids)},
        )
        logger.info(
            "[%s] %s: %s session closed (%.0fs, conf=%.2f)",
            self.camera_id, state.person_name or state.subject_id,
            closed.activity_type, closed.duration_seconds, closed.confidence,
        )
        if self._on_event is not None:
            self._on_event(event)
        return [event]

    def total_seconds(self, state: PersonState, activity: str, now: float) -> float:
        """Running total of one activity for this person (stored + live)."""
        return self._total(state, activity, now)

    def _total(self, state: PersonState, activity: str, now: float) -> float:
        """Stored-before-today baseline + closed time + whatever the open
        session/interval has accrued, so the live row counts the ongoing
        activity and never drops when a new session starts."""
        if activity not in self._enabled:
            return 0.0
        total = state.baseline.get(activity, 0.0)
        if activity in _POSTURE_ACTIVITIES:
            total += state.totals.get(activity, 0.0)
            if state.present and state.primary is not None and str(state.primary.value) == activity:
                total += max(0.0, now - state.primary.since)
            return total
        sess = state.sessions.get(activity)
        if sess is not None:
            total += sess.total(now)
        return total

    # -- leaving ----------------------------------------------------------

    def _expire(self, now: float, now_wall: dt.datetime) -> list[ActivityEvent]:
        events: list[ActivityEvent] = []
        for subject, state in list(self._states.items()):
            if state.present and now - state.last_seen > self._track_grace:
                events.extend(self._leave(state, "track_lost"))
                if not state.is_user:
                    del self._states[subject]
        # Forget track aliases nobody points at any more.
        live = {tid for s in self._states.values() if s.present for tid in s.track_ids}
        for tid in [t for t in self._track_subject if t not in live]:
            del self._track_subject[tid]
        return events

    def _leave(self, state: PersonState, reason: str) -> list[ActivityEvent]:
        """Close everything open for a subject that left. Durations end at
        last_seen -- the grace period is for deciding the person left, not
        time they were observed."""
        events: list[ActivityEvent] = []
        end = state.last_seen
        events.extend(self._close_primary(state, end, {"stream": "primary", "closed_by": reason}))
        state.primary = None
        for act, sess in state.sessions.items():
            closed = sess.close()
            if closed is not None:
                events.extend(self._emit(state, closed, {"stream": act, "closed_by": reason}))
        presence = max(0.0, end - state.first_seen)
        events.extend(self._emit(
            state,
            ClosedSession(Activity.PRESENCE, state.first_seen_wall,
                          state.first_seen_wall + dt.timedelta(seconds=presence), presence, 1.0),
            {"closed_by": reason},
        ))
        state.present = False
        state.track_ids.clear()
        logger.info(
            "[%s] Person left: %s present=%.0fs",
            self.camera_id, state.person_name or state.subject_id, presence,
        )
        return events

    def _cap(self) -> None:
        while len(self._states) > self._max_tracks:
            # Drop absent users first, then the oldest anonymous subject.
            absent = [s for s, st in self._states.items() if not st.present]
            pool = absent or list(self._states)
            oldest = min(pool, key=lambda s: self._states[s].last_seen)
            del self._states[oldest]
