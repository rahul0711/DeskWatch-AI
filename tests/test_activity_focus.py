"""Live-view focus: watching one camera's Phone/Body/... view pauses body
movement on the others and gives the watched camera the GPU."""
import threading
from types import SimpleNamespace

from app.web.server import _apply_focus, _recorded_activities

ADMIN = ("SITTING", "STANDING", "WALKING", "STATIONARY", "PHONE_INTERACTION", "MONITOR_INTERACTION", "COMPUTER_INTERACTION")


class FakePipeline:
    def __init__(self):
        self.calls = []

    def set_activity_focus(self, paused, focus, fps):
        self.calls.append((paused, focus))


class FakeAnalyzer:
    def __init__(self):
        self.enabled = list(ADMIN)

    def set_enabled_activities(self, acts):
        self.enabled = list(acts)


def cam(viewers=0, view="activity", flt="all", activity=True):
    return SimpleNamespace(
        _activity=FakeAnalyzer() if activity else None, _lock=threading.Lock(), _viewer_count=viewers,
        view_mode=view, activity_filter=flt, _focus_applied=(False, None),
        pipeline=FakePipeline(), camera=SimpleNamespace(name="cam"),
    )


def test_watched_phone_view_focuses_it_and_pauses_the_others():
    a, b, c = cam(viewers=1, flt="phone"), cam(), cam(viewers=1, flt="all")
    _apply_focus({"a": a, "b": b, "c": c}, 6.0)
    assert a._focus_applied == (False, "phone")
    assert b._focus_applied == (True, None) and c._focus_applied == (True, None)


def test_closing_the_page_resumes_everything():
    a, b = cam(viewers=1, flt="phone"), cam()
    _apply_focus({"a": a, "b": b}, 6.0)
    a._viewer_count = 0
    _apply_focus({"a": a, "b": b}, 6.0)
    assert a._focus_applied == (False, None) and b._focus_applied == (False, None)


def test_all_and_debug_views_never_pause_anything():
    a, b = cam(viewers=1, flt="all"), cam(viewers=1, flt="debug")
    _apply_focus({"a": a, "b": b}, 6.0)
    assert a.pipeline.calls == [] and b.pipeline.calls == []


def test_two_cameras_watched_in_focused_views_both_run():
    a, b, c = cam(viewers=1, flt="phone"), cam(viewers=1, flt="body"), cam()
    _apply_focus({"a": a, "b": b, "c": c}, 6.0)
    assert a._focus_applied == (False, "phone") and b._focus_applied == (False, "body")
    assert c._focus_applied == (True, None)


def test_settings_are_only_pushed_when_they_change():
    a, b = cam(viewers=1, flt="phone"), cam()
    _apply_focus({"a": a, "b": b}, 6.0)
    _apply_focus({"a": a, "b": b}, 6.0)
    assert len(a.pipeline.calls) == 1 and len(b.pipeline.calls) == 1


def test_phone_focus_records_only_phone_and_restores_after():
    a, b = cam(viewers=1, flt="phone"), cam()
    _apply_focus({"a": a, "b": b}, 6.0, ADMIN)
    assert a._activity.enabled == ["PHONE_INTERACTION"]
    assert b._activity.enabled == list(ADMIN), "paused camera keeps its selection"
    a._viewer_count = 0
    _apply_focus({"a": a, "b": b}, 6.0, ADMIN)
    assert a._activity.enabled == list(ADMIN)


def test_focus_never_records_what_the_admin_switched_off():
    assert _recorded_activities("phone", ["SITTING"]) == []
    assert _recorded_activities("body", ADMIN) == ["SITTING", "STANDING", "WALKING"]
    assert _recorded_activities("monitor", ADMIN) == ["MONITOR_INTERACTION"]
    assert _recorded_activities("computer", ADMIN) == ["COMPUTER_INTERACTION"]
    assert _recorded_activities(None, ADMIN) == list(ADMIN)
