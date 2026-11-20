"""Per-activity session state machine with temporal confirmation + hysteresis.

One ActivitySession exists per (user, activity type). It turns a noisy
per-pass boolean ("phone seen in this person's hand on this pass?") into
clean sessions:

    IDLE --evidence--> PENDING --stable for activation_seconds--> ACTIVE
      ^                   |                                        |  ^
      |        evidence lost for activation_seconds       no evidence |evidence
      +-------------------+                                        v  |
      +------------- absent for grace_seconds ------------------- GRACE

* Activation needs `activation_seconds` of mostly-positive evidence
  (majority over the window, not an unbroken run -- a hand briefly
  occluding the phone must not restart the clock). The session is
  back-dated to the first positive observation, so the confirmation delay
  is not lost from the duration.
* A session survives up to `grace_seconds` without evidence (GRACE). The UI
  keeps showing it as active during GRACE, which is what stops the overlay
  flickering when one detection pass misses the phone. Evidence returning
  inside the grace window resumes the SAME session.
* A session only closes once evidence has been absent for the whole grace
  window. It ends at `last_confirmed` -- the grace period itself is never
  billed as activity time.
* `accumulated_duration` is the sum of every closed session and is never
  reset when a new session starts: phone 25s + phone 20s -> 45s total.

No clock of its own and no I/O: callers pass monotonic `now` and wall-clock
`now_wall`, which keeps this fully unit-testable.
"""
from __future__ import annotations

import datetime as dt
from collections import deque
from dataclasses import dataclass
from typing import Optional


class SessionState:
    IDLE = "IDLE"
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    GRACE = "GRACE"


@dataclass(frozen=True)
class ClosedSession:
    activity_type: str
    start_wall: dt.datetime
    end_wall: dt.datetime
    duration_seconds: float
    # Fraction of observations during the session that were positive -- a
    # measured agreement rate, not a model probability.
    confidence: float


class ActivitySession:
    # Share of the activation window that must be positive to activate.
    _ACTIVATION_RATIO = 0.6

    __slots__ = (
        "activity_type", "activation_seconds", "grace_seconds",
        "current_state", "session_start", "session_start_wall",
        "last_confirmed", "last_confirmed_wall", "accumulated_duration",
        "sessions_closed", "_window", "_pos", "_total",
    )

    def __init__(self, activity_type: str, activation_seconds: float = 1.5, grace_seconds: float = 2.0) -> None:
        self.activity_type = activity_type
        self.activation_seconds = max(0.0, activation_seconds)
        self.grace_seconds = max(0.0, grace_seconds)
        self.current_state = SessionState.IDLE
        self.session_start: Optional[float] = None
        self.session_start_wall: Optional[dt.datetime] = None
        self.last_confirmed: Optional[float] = None
        self.last_confirmed_wall: Optional[dt.datetime] = None
        self.accumulated_duration = 0.0
        self.sessions_closed = 0
        # (timestamp, evidence) since the PENDING candidate started.
        self._window: deque[tuple[float, bool]] = deque()
        self._pos = 0
        self._total = 0

    # -- reads ----------------------------------------------------------

    @property
    def is_on(self) -> bool:
        """True while a session is open, including its grace window -- what
        the UI should show, so a missed detection never blinks the label."""
        return self.current_state in (SessionState.ACTIVE, SessionState.GRACE)

    def current_duration(self, now: float) -> float:
        """Length of the open session so far (0 when none). While ACTIVE it
        runs live; in GRACE it is frozen at last_confirmed, so a session
        that ends up closing never shows time it will not be credited."""
        if not self.is_on or self.session_start is None:
            return 0.0
        end = now if self.current_state == SessionState.ACTIVE else (self.last_confirmed or now)
        return max(0.0, end - self.session_start)

    def total(self, now: float) -> float:
        return self.accumulated_duration + self.current_duration(now)

    @property
    def confidence(self) -> float:
        return self._pos / self._total if self._total else 0.0

    # -- writes ---------------------------------------------------------

    def observe(self, evidence: bool, now: float, now_wall: dt.datetime) -> Optional[ClosedSession]:
        """Feed one observation. Returns the session that closed, if any."""
        evidence = bool(evidence)
        state = self.current_state

        if state == SessionState.IDLE:
            if evidence:
                self._start_pending(now, now_wall)
                if self.activation_seconds <= 0:
                    self._activate(now, now_wall)
            return None

        if state == SessionState.PENDING:
            self._window.append((now, evidence))
            self._total += 1
            if evidence:
                self._pos += 1
                self.last_confirmed, self.last_confirmed_wall = now, now_wall
            # Evidence dried up before confirming: drop the candidate.
            if now - (self.last_confirmed or now) > self.activation_seconds:
                self._reset_to_idle()
                return None
            span = now - self._window[0][0]
            if span >= self.activation_seconds and self._pos / max(1, len(self._window)) >= self._ACTIVATION_RATIO:
                self._activate(now, now_wall)
            return None

        # ACTIVE or GRACE
        self._total += 1
        if evidence:
            self._pos += 1
            self.last_confirmed, self.last_confirmed_wall = now, now_wall
            self.current_state = SessionState.ACTIVE
            return None
        self.current_state = SessionState.GRACE
        return self._close_if_expired(now)

    def tick(self, now: float) -> Optional[ClosedSession]:
        """No observation at all this pass (the person is not in frame, e.g.
        occluded). Counts as absence for grace purposes but casts no vote."""
        if self.current_state == SessionState.PENDING:
            if now - (self.last_confirmed or now) > self.activation_seconds:
                self._reset_to_idle()
            return None
        if self.current_state == SessionState.ACTIVE:
            self.current_state = SessionState.GRACE
        if self.current_state == SessionState.GRACE:
            return self._close_if_expired(now)
        return None

    def close(self) -> Optional[ClosedSession]:
        """Force-close any open session at last_confirmed (shutdown, the
        person left, or the activity was disabled by the admin)."""
        if not self.is_on:
            self._reset_to_idle()
            return None
        return self._finish()

    def absorb(self, other: "ActivitySession") -> None:
        """Merge another session's history into this one. Used when an
        anonymous track turns out to be a user who already has state: their
        time is one person's time, so totals add and overlapping open
        sessions become one."""
        self.accumulated_duration += other.accumulated_duration
        self.sessions_closed += other.sessions_closed
        if not other.is_on:
            return
        if not self.is_on:
            self.current_state = other.current_state
            self.session_start, self.session_start_wall = other.session_start, other.session_start_wall
            self.last_confirmed, self.last_confirmed_wall = other.last_confirmed, other.last_confirmed_wall
            self._pos, self._total = other._pos, other._total
            return
        if other.session_start is not None and (self.session_start is None or other.session_start < self.session_start):
            self.session_start, self.session_start_wall = other.session_start, other.session_start_wall
        if other.last_confirmed is not None and (self.last_confirmed is None or other.last_confirmed > self.last_confirmed):
            self.last_confirmed, self.last_confirmed_wall = other.last_confirmed, other.last_confirmed_wall
            self.current_state = other.current_state
        self._pos += other._pos
        self._total += other._total

    # -- internal -------------------------------------------------------

    def _start_pending(self, now: float, now_wall: dt.datetime) -> None:
        self.current_state = SessionState.PENDING
        self.session_start, self.session_start_wall = now, now_wall
        self.last_confirmed, self.last_confirmed_wall = now, now_wall
        self._window.clear()
        self._window.append((now, True))
        self._pos, self._total = 1, 1

    def _activate(self, now: float, now_wall: dt.datetime) -> None:
        # session_start stays back-dated to the first positive observation.
        self.current_state = SessionState.ACTIVE
        self._window.clear()

    def _close_if_expired(self, now: float) -> Optional[ClosedSession]:
        if self.last_confirmed is not None and now - self.last_confirmed >= self.grace_seconds:
            return self._finish()
        return None

    def _finish(self) -> ClosedSession:
        start = self.session_start if self.session_start is not None else 0.0
        end = self.last_confirmed if self.last_confirmed is not None else start
        duration = max(0.0, end - start)
        start_wall = self.session_start_wall or dt.datetime.now(dt.timezone.utc)
        closed = ClosedSession(
            activity_type=self.activity_type,
            start_wall=start_wall,
            end_wall=start_wall + dt.timedelta(seconds=duration),
            duration_seconds=duration,
            confidence=self.confidence,
        )
        self.accumulated_duration += duration
        self.sessions_closed += 1
        self._reset_to_idle()
        return closed

    def _reset_to_idle(self) -> None:
        self.current_state = SessionState.IDLE
        self.session_start = self.session_start_wall = None
        self.last_confirmed = self.last_confirmed_wall = None
        self._window.clear()
        self._pos = self._total = 0
