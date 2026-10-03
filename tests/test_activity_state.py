"""Temporal-logic tests for the activity state engine.

These drive the state manager with synthetic observations on a controlled
clock -- the point is to prove the debounce/duration/total behaviour without
a camera, a GPU or a model, so the rules stay pinned down.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.activity.movement import MovementAnalyzer
from app.activity.state_manager import Activity, ActivityStateManager


@dataclass
class Obs:
    """Minimal stand-in for the fields state_manager reads off TrackedFace."""
    track_id: int
    posture: str = "sitting"
    phone_in_use: bool = False
    computer_in_use: bool = False
    monitor_facing: bool = False
    monitor_bbox: tuple | None = None


def _mgr(**kw) -> ActivityStateManager:
    defaults = dict(
        camera_id="camera-01",
        confirmation_seconds=2.0,
        identity_grace_seconds=5.0,
        track_grace_seconds=10.0,
        walking_threshold=0.25,
        stationary_threshold=0.06,
        stationary_min_seconds=3.0,
        min_event_seconds=2.0,
    )
    defaults.update(kw)
    return ActivityStateManager(**defaults)


def test_single_frame_flicker_does_not_switch_state():
    m = _mgr()
    t = 0.0
    # Settle into SITTING.
    for _ in range(10):
        m.update([Obs(17, posture="sitting")], {17: 0.0}, t)
        t += 0.25
    assert m.states[17].primary.value == Activity.SITTING

    # One bad frame says standing -- must NOT flip (needs 2s of agreement).
    events = m.update([Obs(17, posture="standing")], {17: 0.0}, t)
    t += 0.25
    m.update([Obs(17, posture="sitting")], {17: 0.0}, t)
    assert events == []
    assert m.states[17].primary.value == Activity.SITTING


def test_sustained_change_switches_and_closes_event_with_duration():
    m = _mgr()
    t = 0.0
    for _ in range(40):  # 10s sitting
        m.update([Obs(17, posture="sitting")], {17: 0.0}, t)
        t += 0.25

    closed = []
    for _ in range(20):  # 5s standing -> confirms after 2s
        closed += m.update([Obs(17, posture="standing")], {17: 0.0}, t)
        t += 0.25

    sitting_events = [e for e in closed if e.activity_type == Activity.SITTING]
    assert len(sitting_events) == 1
    ev = sitting_events[0]
    # The sitting interval ran ~10s and must not be shortened by the 2s
    # confirmation window.
    assert 9.0 <= ev.duration_seconds <= 11.0
    assert ev.end_time > ev.start_time
    assert m.states[17].primary.value == Activity.STANDING


def test_stationary_requires_min_duration_then_runs_until_movement():
    m = _mgr()
    t = 0.0
    # Movement below threshold from the start, but stationary must not latch
    # until stationary_min_seconds (3s) of low movement has passed.
    m.update([Obs(17)], {17: 0.0}, t)
    t += 1.0
    m.update([Obs(17)], {17: 0.0}, t)
    assert m.states[17].stationary.value is False

    for _ in range(40):  # plenty of low-movement time
        t += 0.25
        m.update([Obs(17)], {17: 0.0}, t)
    assert m.states[17].stationary.value is True
    assert m.states[17].stationary_seconds(t) > 3.0

    # Real movement ends it (after the debounce).
    for _ in range(20):
        t += 0.25
        m.update([Obs(17)], {17: 0.5}, t)
    assert m.states[17].stationary.value is False


def test_phone_interaction_event_has_duration_and_confidence():
    m = _mgr()
    t = 0.0
    for _ in range(20):
        m.update([Obs(17, phone_in_use=False)], {17: 0.0}, t)
        t += 0.25

    for _ in range(40):  # 10s of phone use
        m.update([Obs(17, phone_in_use=True)], {17: 0.0}, t)
        t += 0.25

    closed = []
    for _ in range(20):  # put it down
        closed += m.update([Obs(17, phone_in_use=False)], {17: 0.0}, t)
        t += 0.25

    phone = [e for e in closed if e.activity_type == Activity.PHONE_INTERACTION]
    assert len(phone) == 1
    assert 9.0 <= phone[0].duration_seconds <= 11.0
    assert 0.0 < phone[0].confidence <= 1.0


def test_short_blip_is_not_stored_but_still_counts_toward_totals():
    m = _mgr(min_event_seconds=5.0)
    t = 0.0
    for _ in range(20):
        m.update([Obs(17, phone_in_use=False)], {17: 0.0}, t)
        t += 0.25
    # ~3s of phone use: longer than the 2s debounce, shorter than the 5s
    # storage floor.
    for _ in range(12):
        m.update([Obs(17, phone_in_use=True)], {17: 0.0}, t)
        t += 0.25
    closed = []
    for _ in range(20):
        closed += m.update([Obs(17, phone_in_use=False)], {17: 0.0}, t)
        t += 0.25

    assert [e for e in closed if e.activity_type == Activity.PHONE_INTERACTION] == []
    rows = {r["track_id"]: r for r in m.rows(t)}
    assert rows[17]["phone_seconds"] > 2.0


def test_totals_accumulate_across_multiple_intervals():
    m = _mgr()
    t = 0.0

    def run(posture, seconds):
        nonlocal t
        for _ in range(int(seconds / 0.25)):
            m.update([Obs(17, posture=posture)], {17: 0.0}, t)
            t += 0.25

    run("sitting", 10)
    run("standing", 10)
    run("sitting", 10)
    rows = {r["track_id"]: r for r in m.rows(t)}
    # Two sitting intervals of ~10s each, minus debounce slack.
    assert rows[17]["sitting_seconds"] > 17.0
    assert rows[17]["standing_seconds"] > 7.0


def test_track_loss_closes_open_events_and_emits_presence():
    m = _mgr()
    t = 0.0
    for _ in range(40):
        m.update([Obs(17, posture="sitting")], {17: 0.0}, t)
        t += 0.25

    # Nothing observed for longer than track_grace_seconds.
    t += 11.0
    closed = m.update([], {}, t)
    kinds = {e.activity_type for e in closed}
    assert Activity.PRESENCE in kinds
    assert Activity.SITTING in kinds
    assert 17 not in m.states

    presence = next(e for e in closed if e.activity_type == Activity.PRESENCE)
    # Presence ends at last_seen, not at the expiry check -- the grace
    # period must not be billed as observed time.
    assert 9.0 <= presence.duration_seconds <= 11.0


def test_identity_grace_and_staleness():
    m = _mgr()
    t = 0.0
    m.update([Obs(17)], {17: 0.0}, t)
    assert m.identity_is_stale(17, t) is True

    m.apply_identity(17, person_id="42", person_name="Rahul", confidence=0.9, now=t)
    assert m.identity_is_stale(17, t) is False
    assert m.identity_is_stale(17, t + 3.0) is False   # inside 5s grace
    assert m.identity_is_stale(17, t + 7.0) is True    # past it

    rows = {r["track_id"]: r for r in m.rows(t)}
    assert rows[17]["person_name"] == "Rahul"


def test_walking_overrides_posture_when_movement_is_high():
    m = _mgr()
    t = 0.0
    for _ in range(40):
        m.update([Obs(17, posture="standing")], {17: 0.9}, t)
        t += 0.25
    assert m.states[17].primary.value == Activity.WALKING


def test_display_priority_phone_over_monitor_over_primary():
    m = _mgr()
    t = 0.0
    for _ in range(40):
        m.update([Obs(17, posture="sitting", monitor_facing=True)], {17: 0.0}, t)
        t += 0.25
    assert m.states[17].display_activity == Activity.MONITOR_INTERACTION

    for _ in range(40):
        m.update(
            [Obs(17, posture="sitting", monitor_facing=True, phone_in_use=True)],
            {17: 0.0}, t,
        )
        t += 0.25
    assert m.states[17].display_activity == Activity.PHONE_INTERACTION


# --- movement analyzer -------------------------------------------------


def test_movement_score_is_scale_invariant():
    """Same motion as a fraction of body size must score the same whether
    the person is near (large bbox) or far (small bbox)."""
    a = MovementAnalyzer(ema_alpha=1.0)
    # Far person: 50x100 box, moves 5px in 1s (5% of a 111px diagonal).
    a.update(1, (0, 0, 50, 100), None, None, 0.0)
    far = a.update(1, (5, 0, 55, 100), None, None, 1.0)
    # Near person: 200x400 box, moves 20px in 1s (same fraction).
    a.update(2, (0, 0, 200, 400), None, None, 0.0)
    near = a.update(2, (20, 0, 220, 400), None, None, 1.0)
    assert far == near != 0.0


def test_movement_score_first_frame_is_zero_and_wrists_contribute():
    a = MovementAnalyzer(ema_alpha=1.0)
    bbox = (0, 0, 100, 200)
    kp = np.zeros((17, 2), dtype=np.float32)
    conf = np.full(17, 0.9, dtype=np.float32)
    assert a.update(1, bbox, kp, conf, 0.0) == 0.0

    # Only the wrists move; the box centre is unchanged.
    kp2 = kp.copy()
    kp2[9] = [30, 0]
    kp2[10] = [30, 0]
    score = a.update(1, bbox, kp2, conf, 1.0)
    assert score > 0.0


def test_movement_analyzer_forgets_dead_tracks():
    a = MovementAnalyzer()
    a.update(1, (0, 0, 10, 10), None, None, 0.0)
    a.update(2, (0, 0, 10, 10), None, None, 0.0)
    a.keep_only([2])
    assert a.score(1) == 0.0


def test_alternating_posture_still_settles_on_the_majority():
    """A high-mounted camera genuinely alternates sitting/standing when hips
    and knees are occluded. A strict 'must hold continuously' debounce never
    commits in that case; majority-over-window must."""
    m = _mgr()
    t = 0.0
    # 3 sitting : 1 standing, repeating -- sitting never holds an unbroken
    # 2s run at 4 fps, but it is clearly the majority.
    for i in range(60):
        posture = "standing" if i % 4 == 3 else "sitting"
        m.update([Obs(17, posture=posture)], {17: 0.0}, t)
        t += 0.25
    assert m.states[17].primary.value == Activity.SITTING


def test_evenly_split_posture_does_not_flap():
    """A 50/50 split must not reach the 60% majority, so the committed state
    stays put rather than switching on every other frame."""
    m = _mgr()
    t = 0.0
    for _ in range(20):  # settle on sitting
        m.update([Obs(17, posture="sitting")], {17: 0.0}, t)
        t += 0.25
    switches = 0
    for i in range(60):
        posture = "standing" if i % 2 else "sitting"
        events = m.update([Obs(17, posture=posture)], {17: 0.0}, t)
        switches += len([e for e in events if e.activity_type in (Activity.SITTING, Activity.STANDING)])
        t += 0.25
    assert switches == 0
    assert m.states[17].primary.value == Activity.SITTING


def test_ghost_tracks_emit_no_events_but_still_total():
    """A track seen only a couple of times (detector firing briefly on a
    chair or reflection) must not write events."""
    m = _mgr(min_track_observations=8)
    t = 0.0
    # Seen 3 times only, then gone.
    for _ in range(3):
        m.update([Obs(99, posture="sitting")], {99: 0.0}, t)
        t += 0.25
    t += 11.0  # past track_grace_seconds
    events = m.update([], {}, t)
    assert events == []
    assert 99 not in m.states


def test_established_track_still_emits_events():
    m = _mgr(min_track_observations=8)
    t = 0.0
    for _ in range(40):  # 10s, 40 observations
        m.update([Obs(17, posture="sitting")], {17: 0.0}, t)
        t += 0.25
    t += 11.0
    events = m.update([], {}, t)
    assert {e.activity_type for e in events} >= {Activity.PRESENCE, Activity.SITTING}


# --- overlay view filters ----------------------------------------------


def _track(**kw):
    """Minimal object with the attributes _passes_filter reads."""
    from dataclasses import dataclass

    @dataclass
    class T:
        phone_in_use: bool = False
        monitor_facing: bool = False
        computer_in_use: bool = False

    return T(**kw)


def test_activity_filters_select_the_right_people():
    from app.ui.display import ACTIVITY_FILTERS, _passes_filter

    phone_user = _track(phone_in_use=True)
    monitor_user = _track(monitor_facing=True)
    pc_user = _track(computer_in_use=True)
    idle = _track()

    # "all" and "body" show everyone -- "body" only trims the LABEL down to
    # posture/movement, it does not hide people.
    for f in ("all", "body"):
        assert all(_passes_filter(t, f) for t in (phone_user, monitor_user, pc_user, idle))

    assert _passes_filter(phone_user, "phone") is True
    assert _passes_filter(idle, "phone") is False
    assert _passes_filter(monitor_user, "phone") is False

    assert _passes_filter(monitor_user, "monitor") is True
    assert _passes_filter(phone_user, "monitor") is False

    assert _passes_filter(pc_user, "computer") is True
    assert _passes_filter(idle, "computer") is False

    # Every filter the API accepts must be handled here.
    for f in ACTIVITY_FILTERS:
        assert isinstance(_passes_filter(idle, f), bool)


def test_unknown_filter_falls_back_to_showing_everyone():
    """A filter value the renderer doesn't know must not blank the overlay."""
    from app.ui.display import _passes_filter

    assert _passes_filter(_track(), "something-new") is True
