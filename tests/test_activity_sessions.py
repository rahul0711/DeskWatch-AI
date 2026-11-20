"""USER-level activity tracking: sessions, hysteresis, re-identification.

Pins down the rules the tracking architecture exists for:
  * TRACK_ID is never USER_ID -- a new track for a known user continues that
    user's state and totals.
  * Sessions need stable evidence to start, survive brief detection loss,
    and close only after the grace period.
  * Two sessions (25s + 20s) are stored separately, the total shows 45s.
  * Face embeddings reconnect a new track to its user; recognition runs on
    a schedule, not every frame.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from app.activity.identity_binder import Identity
from app.activity.identity_cache import IdentityCache
from app.activity.sessions import ActivitySession, SessionState
from app.activity.state_manager import Activity, ActivityStateManager
from app.processing.process_detector import _EmbeddingScheduler

WALL0 = dt.datetime(2026, 10, 5, 9, 0, tzinfo=dt.timezone.utc)
STEP = 0.25  # 4 passes per second


def wall(t: float) -> dt.datetime:
    return WALL0 + dt.timedelta(seconds=t)


@dataclass
class Box:
    bbox: tuple
    confidence: float = 0.9


@dataclass
class Obs:
    track_id: int
    posture: str = "sitting"
    phone_in_use: bool = False
    computer_in_use: bool = False
    monitor_facing: bool = False
    monitor_bbox: tuple | None = None
    face: Box = field(default_factory=lambda: Box((100, 100, 300, 600)))


@dataclass
class Binding:
    person_id: str
    person_name: str
    confidence: float = 1.0
    source: str = "explicit"


def _mgr(**kw) -> ActivityStateManager:
    defaults = dict(
        camera_id="camera-01", confirmation_seconds=2.0, track_grace_seconds=10.0,
        min_event_seconds=2.0, min_track_observations=4,
        activation_seconds=1.5, session_grace_seconds=2.0,
    )
    defaults.update(kw)
    return ActivityStateManager(**defaults)


# --- ActivitySession ----------------------------------------------------


def _feed(sess, pattern, t0=0.0):
    """pattern: list of (evidence, seconds). Returns (closed sessions, t)."""
    closed, t = [], t0
    for evidence, seconds in pattern:
        for _ in range(int(round(seconds / STEP))):
            c = sess.observe(evidence, t, wall(t))
            if c:
                closed.append(c)
            t += STEP
    return closed, t


def test_session_needs_stable_evidence_to_activate():
    s = ActivitySession("PHONE_INTERACTION", activation_seconds=1.5, grace_seconds=2.0)
    _feed(s, [(True, 1.0)])
    assert s.current_state == SessionState.PENDING and not s.is_on
    _feed(s, [(True, 1.0)], t0=1.0)
    assert s.current_state == SessionState.ACTIVE
    # Back-dated: the confirmation delay is not lost from the session.
    assert s.session_start == 0.0


def test_single_blip_never_activates():
    s = ActivitySession("PHONE_INTERACTION", activation_seconds=1.5, grace_seconds=2.0)
    closed, _ = _feed(s, [(False, 2), (True, 0.25), (False, 5)])
    assert closed == [] and s.accumulated_duration == 0.0 and not s.is_on


def test_brief_loss_inside_grace_keeps_one_session_and_stays_on():
    s = ActivitySession("PHONE_INTERACTION", activation_seconds=1.5, grace_seconds=2.0)
    closed, t = _feed(s, [(True, 10)])
    # 1.5s of missed detections: inside the 2s grace.
    for _ in range(6):
        closed += [c for c in [s.observe(False, t, wall(t))] if c]
        assert s.is_on, "UI must not flicker during the grace window"
        t += STEP
    closed += _feed(s, [(True, 10)], t0=t)[0]
    assert closed == []
    assert s.sessions_closed == 0
    assert s.total(t + 20) > 19.0


def test_two_sessions_stored_separately_total_is_their_sum():
    """User uses phone 25s, puts it away, uses it again 20s:
    two stored sessions (25s, 20s) and a 45s total."""
    s = ActivitySession("PHONE_INTERACTION", activation_seconds=1.5, grace_seconds=2.0)
    closed, _ = _feed(s, [(True, 25), (False, 10), (True, 20), (False, 5)])
    assert len(closed) == 2
    assert abs(closed[0].duration_seconds - 25) <= 0.5
    assert abs(closed[1].duration_seconds - 20) <= 0.5
    assert abs(s.accumulated_duration - 45) <= 1.0
    assert all(0 < c.confidence <= 1 for c in closed)
    # Ends at the last confirming pass -- grace is never billed.
    assert closed[0].end_wall <= wall(25)


def test_tick_without_observation_closes_after_grace():
    s = ActivitySession("PHONE_INTERACTION", activation_seconds=1.5, grace_seconds=2.0)
    _, t = _feed(s, [(True, 5)])
    assert s.tick(t + 1.0) is None and s.is_on
    closed = s.tick(t + 3.0)
    assert closed is not None and abs(closed.duration_seconds - 4.75) < 0.01


# --- TRACK_ID != USER_ID ---------------------------------------------------


def _run(m, track_id, seconds, t, identities=None, **obs_kw):
    events = []
    for _ in range(int(round(seconds / STEP))):
        events += m.update([Obs(track_id, **obs_kw)], {track_id: 0.0}, t, wall(t), identities=identities)
        t += STEP
    return events, t


def test_new_track_for_same_user_keeps_state_and_totals():
    """Track 15 -> User 1, occlusion, Track 42 -> User 1: one user, one total."""
    m = _mgr()
    user1 = Binding("1", "Rahul")
    ev1, t = _run(m, 15, 25, 0.0, {15: user1}, phone_in_use=True)
    # Occluded: nothing observed for 5s (past the session grace, inside track grace).
    for _ in range(20):
        ev1 += m.update([], {}, t, wall(t))
        t += STEP
    ev2, t = _run(m, 42, 20, t, {42: user1}, phone_in_use=True)
    ev3, t = _run(m, 42, 5, t, {42: user1}, phone_in_use=False)

    assert m.state_for_track(15) is m.state_for_track(42)
    users = [s for s in m.states.values() if s.is_user]
    assert len(users) == 1 and users[0].person_id == "1"
    assert users[0].track_ids >= {42}

    phone = [e for e in ev1 + ev2 + ev3 if e.activity_type == Activity.PHONE_INTERACTION]
    assert len(phone) == 2, "two separate sessions stored"
    assert {e.person_id for e in phone} == {"1"}, "sessions belong to the user, not the track"
    row = next(r for r in m.rows(t) if r["user_id"] == "1")
    assert 43.0 <= row["phone_seconds"] <= 46.0, row["phone_seconds"]


def test_anonymous_track_merges_into_existing_user_once_identified():
    m = _mgr()
    user1 = Binding("1", "Rahul")
    _, t = _run(m, 15, 10, 0.0, {15: user1})
    # New track appears before identity is known: anonymous for 3s...
    _, t = _run(m, 42, 3, t)
    assert m.state_for_track(42) is not m.state_for_track(15)
    # ...then re-identified -> folded into User 1, no second user.
    _, t = _run(m, 42, 3, t, {42: Binding("1", "Rahul", 0.6, "reid")})
    assert m.state_for_track(42) is m.state_for_track(15)
    assert len(m.states) == 1
    row = m.rows(t)[0]
    assert row["sitting_seconds"] > 14.0
    assert row["identity_source"] == "reid"


def test_user_totals_survive_leaving_and_coming_back():
    m = _mgr()
    u = Binding("7", "Asha")
    _, t = _run(m, 3, 12, 0.0, {3: u}, phone_in_use=True)
    t += 15.0  # gone past track grace -> user "left"
    m.update([], {}, t, wall(t))
    row = next(r for r in m.rows(t) if r["user_id"] == "7")
    assert row["present"] is False and row["phone_seconds"] > 11.0
    # Back with a brand new track id.
    _, t = _run(m, 99, 8, t, {99: u}, phone_in_use=True)
    row = next(r for r in m.rows(t) if r["user_id"] == "7")
    assert row["present"] is True
    assert row["phone_seconds"] > 18.0, "never reset by a new session/track"


def test_two_tracks_of_one_user_in_a_pass_count_once():
    m = _mgr()
    u = Binding("1", "Rahul")
    t = 0.0
    for _ in range(40):  # 10s, both tracks visible every pass
        m.update([Obs(15, phone_in_use=True), Obs(16, phone_in_use=True)], {}, t, wall(t),
                 identities={15: u, 16: u})
        t += STEP
    row = m.rows(t)[0]
    assert len(m.states) == 1
    assert row["phone_seconds"] <= 10.5


def test_phone_session_survives_a_longer_detector_gap():
    m = _mgr(phone_grace_seconds=5.0)
    u = Binding("1", "Rahul")
    _, t = _run(m, 1, 5, 0.0, {1: u}, phone_in_use=True)
    events, t = _run(m, 1, 3.5, t, {1: u}, phone_in_use=False)   # 3.5s miss
    assert not any(e.activity_type == Activity.PHONE_INTERACTION for e in events)
    _, t = _run(m, 1, 5, t, {1: u}, phone_in_use=True)
    assert m.state_for_track(1).sessions[Activity.PHONE_INTERACTION].sessions_closed == 0


def test_phone_shows_at_first_sighting_and_confirms_fast():
    m = _mgr(phone_activation_seconds=0.8)
    u = Binding("1", "Rahul")
    m.update([Obs(1, phone_in_use=True)], {1: 0.0}, 0.0, wall(0.0), identities={1: u})
    state = m.state_for_track(1)
    assert state.is_starting("phone") and not state.is_on("phone")
    _, t = _run(m, 1, 1.0, STEP, {1: u}, phone_in_use=True)
    assert state.is_on("phone"), "confirmed after ~0.8s, not 1.5s"


def test_baseline_from_database_is_added_to_live_totals():
    m = _mgr(baseline_loader=lambda pid: {Activity.PHONE_INTERACTION: 100.0})
    _, t = _run(m, 1, 5, 0.0, {1: Binding("1", "Rahul")}, phone_in_use=True)
    row = m.rows(t)[0]
    assert 104.0 <= row["phone_seconds"] <= 106.0


def test_only_selected_activities_are_tracked_and_stored():
    m = _mgr(enabled_activities=[Activity.PHONE_INTERACTION])
    events, t = _run(m, 1, 10, 0.0, {1: Binding("1", "Rahul")}, phone_in_use=True, monitor_facing=True)
    t += 15
    events += m.update([], {}, t, wall(t))
    kinds = {e.activity_type for e in events}
    assert Activity.PHONE_INTERACTION in kinds
    assert Activity.MONITOR_INTERACTION not in kinds and Activity.SITTING not in kinds
    row = m.rows(t)[0]
    assert row["monitor_seconds"] == 0.0 and row["sitting_seconds"] == 0.0


def test_deselecting_an_activity_closes_and_stores_its_open_session():
    m = _mgr()
    _, t = _run(m, 1, 10, 0.0, {1: Binding("1", "Rahul")}, phone_in_use=True)
    events = m.set_enabled_activities([Activity.SITTING], t, wall(t))
    kinds = [e.activity_type for e in events]
    assert Activity.PHONE_INTERACTION in kinds
    assert Activity.SITTING not in kinds, "still selected -> stays open"
    assert Activity.PHONE_INTERACTION not in m.state_for_track(1).sessions


# --- identity cache ----------------------------------------------------------


@dataclass
class FaceT:
    track_id: int
    face: Box
    embedding: object = None
    name: str = "Unknown"
    name_confidence: float = 0.0


@dataclass
class PersonT:
    track_id: int
    face: Box


def _unit(seed: int, noise: float = 0.0, base=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = base if base is not None else rng.normal(size=512)
    if noise:
        v = v + rng.normal(scale=noise, size=512)
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_embedding_reconnects_new_track_to_known_user():
    cache = IdentityCache(match_threshold=0.45)
    alice, bob = _unit(1), _unit(2)
    face_box, body_box = Box((150, 110, 210, 170)), Box((100, 100, 300, 600))

    # Track 15 identified by /Recognize; its embedding is learned.
    out = cache.update([FaceT(5, face_box, alice)], [PersonT(15, body_box)],
                       {5: Identity("1", "Alice", 1.0)}, now=0.0)
    assert out[15].person_id == "1" and out[15].source == "explicit"
    cache.update([FaceT(6, Box((450, 110, 510, 170)), bob)], [PersonT(20, Box((400, 100, 600, 600)))],
                 {6: Identity("2", "Bob", 1.0)}, now=0.1)

    # Occlusion -> brand new face track 9 / person track 42, no /Recognize yet.
    alice_again = _unit(11, noise=0.04, base=alice * np.sqrt(512))
    out = cache.update([FaceT(9, face_box, alice_again)], [PersonT(42, body_box)], {}, now=5.0)
    assert out[42].person_id == "1" and out[42].source == "reid"


def test_stranger_embedding_is_not_matched():
    cache = IdentityCache(match_threshold=0.45)
    face_box, body_box = Box((150, 110, 210, 170)), Box((100, 100, 300, 600))
    cache.update([FaceT(5, face_box, _unit(1))], [PersonT(15, body_box)], {5: Identity("1", "Alice", 1.0)}, 0.0)
    out = cache.update([FaceT(9, face_box, _unit(99))], [PersonT(42, body_box)], {}, now=5.0)
    assert 42 not in out


def test_binding_persists_while_face_is_hidden():
    """Head turned away: no face this pass, the person track keeps its user."""
    cache = IdentityCache()
    face_box, body_box = Box((150, 110, 210, 170)), Box((100, 100, 300, 600))
    cache.update([FaceT(5, face_box, _unit(1))], [PersonT(15, body_box)], {5: Identity("1", "Alice", 1.0)}, 0.0)
    out = cache.update([], [PersonT(15, body_box)], {}, now=3.0)
    assert out[15].person_id == "1"


def test_one_user_is_never_bound_to_several_people_at_once():
    """Four people on screen, only one is a known user: the others must not
    all become that user through re-id."""
    cache = IdentityCache(match_threshold=0.45)
    shivang = _unit(1)
    boxes = [(Box((150 + i * 300, 110, 210 + i * 300, 170)), Box((100 + i * 300, 100, 300 + i * 300, 600)))
             for i in range(4)]
    # Track 10 is Shivang (explicit), the other three look somewhat like him
    # (close enough to pass the re-id threshold on their own).
    look_alikes = [_unit(20 + i, noise=0.06, base=shivang * np.sqrt(512)) for i in range(3)]
    faces = [FaceT(1, boxes[0][0], shivang)] + [FaceT(2 + i, boxes[i + 1][0], e) for i, e in enumerate(look_alikes)]
    persons = [PersonT(10 + i, boxes[i][1]) for i in range(4)]
    out = {}
    for k in range(10):
        out = cache.update(faces, persons, {1: Identity("1", "Shivang", 1.0)}, now=k * 0.25)
    assert out[10].person_id == "1"
    assert [tid for tid, b in out.items() if b.person_id == "1"] == [10]


def test_reid_matches_do_not_teach_the_gallery():
    cache = IdentityCache(match_threshold=0.45)
    alice = _unit(1)
    face_box, body_box = Box((150, 110, 210, 170)), Box((100, 100, 300, 600))
    cache.update([FaceT(5, face_box, alice)], [PersonT(15, body_box)], {5: Identity("1", "Alice", 1.0)}, 0.0)
    # Alice leaves; a new track drifts away from her face over many passes.
    for k in range(40):
        drift = _unit(100 + k, noise=0.02 * k, base=alice * np.sqrt(512))
        cache.update([FaceT(9, face_box, drift)], [PersonT(42, body_box)], {}, now=40.0 + k)
    stranger = _unit(99)
    out = cache.update([FaceT(11, face_box, stranger)], [PersonT(77, body_box)], {}, now=100.0)
    assert 77 not in out


def test_detached_track_stops_sharing_the_users_activity():
    """Track 2 was wrongly merged into user 1; once identity moves to track 1
    only, track 2's phone must not count for user 1."""
    m = _mgr()
    u = Binding("1", "Shivang")
    t = 0.0
    for _ in range(8):  # wrong: both tracks bound to the same user
        m.update([Obs(1), Obs(2)], {}, t, wall(t), identities={1: u, 2: u})
        t += STEP
    for _ in range(40):  # fixed binding: only track 1 is the user, track 2 has the phone
        m.update([Obs(1), Obs(2, phone_in_use=True)], {}, t, wall(t), identities={1: u})
        t += STEP
    assert m.subject_for_track(2) != m.subject_for_track(1)
    user_row = next(r for r in m.rows(t) if r["user_id"] == "1")
    assert user_row["phone_seconds"] < 3.0


# --- recognition scheduling ----------------------------------------------------


@dataclass
class SchedT:
    track_id: int
    face: Box
    embedding: object = None
    registered_user_id: int | None = None


def test_recognition_runs_on_schedule_not_every_frame():
    sched = _EmbeddingScheduler(revalidate_seconds=10.0, retry_seconds=0.5)
    t1 = SchedT(1, Box((0, 0, 60, 60)), registered_user_id=5)  # already identified
    runs = 0
    now = 0.0
    for _ in range(100):  # 10s at 10 fps
        for t, _reason in sched.select([t1], now):
            sched.ran(t, now)
            t.embedding = np.ones(512, dtype=np.float32)
            runs += 1
        sched.seen([t1], now)
        now += 0.1
    assert runs == 1, "new track once, then cached until revalidation"
    assert sched.reason(t1, 10.5) == "revalidate"


def test_recognition_reruns_after_occlusion_and_box_jump():
    sched = _EmbeddingScheduler(revalidate_seconds=100.0)
    t1 = SchedT(1, Box((0, 0, 60, 60)), embedding=np.ones(512))
    sched.ran(t1, 0.0)
    sched.seen([t1], 0.0)
    assert sched.reason(t1, 0.1) is None
    assert sched.reason(t1, 2.0) == "after_occlusion"
    sched.seen([t1], 2.0)
    t1.face = Box((400, 400, 460, 460))
    assert sched.reason(t1, 2.1) == "uncertain"


def test_recognition_budget_per_pass_prefers_faces_without_embedding():
    sched = _EmbeddingScheduler(max_per_pass=2)
    tracks = [SchedT(i, Box((0, 0, 60, 60))) for i in range(5)]
    assert len(sched.select(tracks, 0.0)) == 2


def test_unidentified_face_is_retried_a_bounded_number_of_times():
    """A stranger is retried ~every second for a while, then only on
    revalidation -- never every frame, never forever."""
    sched = _EmbeddingScheduler(revalidate_seconds=100.0, unidentified_retry_seconds=1.0, unidentified_max_tries=8)
    t1 = SchedT(1, Box((0, 0, 60, 60)))
    runs, now = 0, 0.0
    for _ in range(300):  # 30s at 10 fps
        for t, _reason in sched.select([t1], now):
            sched.ran(t, now)
            t.embedding = np.ones(512, dtype=np.float32)
            runs += 1
        sched.seen([t1], now)
        now += 0.1
    assert runs == 8


def test_identified_face_stops_unidentified_retries():
    sched = _EmbeddingScheduler(revalidate_seconds=100.0)
    t1 = SchedT(1, Box((0, 0, 60, 60)), embedding=np.ones(512))
    sched.ran(t1, 0.0)
    sched.seen([t1], 0.0)
    sched.seen([t1], 1.4)  # continuously visible, no occlusion
    assert sched.reason(t1, 1.5) == "unidentified"
    t1.registered_user_id = 3
    assert sched.reason(t1, 1.5) is None
