"""A detector that can't initialize must fail loudly, not silently.

If detector construction raises inside the worker process, the child dies and
the parent's latest_result() keeps handing back the empty startup result --
so the UI shows "0 detections" forever and a broken backend is indistinguishable
from an empty scene. That's exactly how the Windows/oneDNN OCR failure hid
(see OCRDetector.__init__), so the worker entry point catches and logs it.

_worker_main only needs .get()/.put_nowait()/.is_set() from its queues and
stop event, so a plain queue.Queue and threading.Event exercise it directly
without paying for a real spawn.
"""
import logging
import queue
import threading

from app.processing.process_detector import _worker_main


class ExplodingDetector:
    def __init__(self, **kwargs):
        raise RuntimeError("backend unavailable on this platform")

    def detect(self, frame):  # pragma: no cover - never reached
        return []

    def close(self):  # pragma: no cover - never reached
        pass


def test_worker_survives_detector_init_failure(caplog):
    with caplog.at_level(logging.ERROR):
        # Must return rather than propagate -- an uncaught raise here is what
        # killed the child silently.
        _worker_main(
            ExplodingDetector, {}, False,
            queue.Queue(), queue.Queue(), threading.Event(),
        )

    assert "ExplodingDetector" in caplog.text
    assert "failed to initialize" in caplog.text
    # The underlying cause has to reach the log, or there's nothing to act on.
    assert "backend unavailable on this platform" in caplog.text
