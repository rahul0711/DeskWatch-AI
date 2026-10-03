"""Minimal OpenCV window renderer: HUD text + face/head boxes.

HEAD_COUNT_SOURCE (set in .env) controls what is detected and shown:
  face   -- only green face boxes + "Faces detected: N"
  head   -- only cyan body/head boxes + "Heads detected: N"  (people
            looking down at desks are captured that face-only misses)
  both   -- green face boxes AND cyan body/head boxes, HUD shows both counts

Deliberately not a web UI for Phase 1 — cv2.imshow is the fastest way to
visually prove the pipeline works. This module only reads a PipelineResult
and draws; a Phase 2 FastAPI/WebSocket MJPEG endpoint can reuse
`draw_overlay()` and just skip the imshow/waitKey part.
"""
from __future__ import annotations

import cv2
import numpy as np

from app.camera.rtsp_stream import ConnectionState
from app.processing.pipeline import PipelineResult

WINDOW_NAME = "CCTV Face Detection - Phase 1"

_STATE_COLORS = {
    ConnectionState.CONNECTED: (0, 200, 0),
    ConnectionState.CONNECTING: (0, 200, 200),
    ConnectionState.RECONNECTING: (0, 140, 255),
    ConnectionState.STOPPED: (0, 0, 200),
}

# Colours for detection types
_FACE_COLOR = (0, 255, 0)        # bright green (YOLO Face)
_HEAD_COLOR = (255, 220, 0)      # cyan / teal (Person Body)
_ATTENDANCE_COLOR = (255, 120, 0)  # orange / purple (InsightFace Face Attendance)

# Activity mode: box color keyed by posture (BGR).
_POSTURE_COLORS = {
    "sitting": (0, 165, 255),   # orange
    "standing": (0, 255, 0),    # green
    "unknown": (150, 150, 150),  # gray
}

# OCR mode: magenta for a confirmed (multi-frame-voted) reading, dim gray
# for a region that's been seen but hasn't voted-in a stable string yet.
_OCR_CONFIRMED_COLOR = (255, 0, 255)
_OCR_PENDING_COLOR = (120, 120, 120)
_OCR_MIN_VOTES_TO_CONFIRM = 2


# COCO-pose skeleton edges, for the activity debug overlay.
_SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
)
_MONITOR_COLOR = (200, 200, 60)
_SKELETON_COLOR = (0, 255, 255)
_ACTIVITY_LABEL_COLORS = {
    "PHONE_INTERACTION": (0, 80, 255),
    "MONITOR_INTERACTION": (200, 200, 60),
    "COMPUTER_INTERACTION": (255, 180, 0),
    "WALKING": (0, 220, 255),
    "SITTING": (0, 165, 255),
    "STANDING": (0, 255, 0),
    "STATIONARY": (150, 150, 150),
    "UNKNOWN": (150, 150, 150),
}


def _fmt_dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# Body-movement sub-views. "all" shows everything; the others narrow the
# overlay to one kind of evidence so an operator watching for phone use
# isn't reading five labels per person.
ACTIVITY_FILTERS = ("all", "phone", "body", "monitor", "computer", "debug")


def _passes_filter(t, activity_filter: str) -> bool:
    if activity_filter == "phone":
        return bool(t.phone_in_use)
    if activity_filter == "monitor":
        return bool(t.monitor_facing)
    if activity_filter == "computer":
        return bool(t.computer_in_use)
    # "all" and "body" show every tracked person; "body" only trims the
    # label down to posture/movement (see below).
    return True


_PHONE_BOX_COLOR = (0, 80, 255)   # orange-red
_DEBUG_TEXT_COLOR = (255, 255, 255)
_DEBUG_BG = (30, 30, 30)


def _draw_skeleton(frame, t):
    if t.keypoints is None:
        return
    kxy, kconf = t.keypoints, t.keypoints_conf
    for a, b in _SKELETON:
        if kconf is None or a >= len(kconf) or b >= len(kconf) or kconf[a] < 0.25 or kconf[b] < 0.25:
            continue
        cv2.line(frame, (int(kxy[a][0]), int(kxy[a][1])),
                 (int(kxy[b][0]), int(kxy[b][1])), _SKELETON_COLOR, 2, cv2.LINE_AA)


def _draw_label_block(frame, x1, y1, lines, color):
    """Draw a multi-line label above a box."""
    box_h = 4 + 17 * len(lines)
    widest = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for l in lines)
    ly = max(box_h, y1)
    cv2.rectangle(frame, (x1, ly - box_h), (x1 + widest + 8, ly), color, -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x1 + 4, ly - box_h + 14 + 17 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_debug_track(frame: np.ndarray, t) -> None:
    """Raw measurements overlay — no interaction labels, just evidence."""
    x1, y1, x2, y2 = t.face.bbox

    _draw_skeleton(frame, t)

    # Person box colored by posture
    color = _POSTURE_COLORS.get(t.posture, (150, 150, 150))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Phone box (if detected for this person)
    phone_bbox = getattr(t, "phone_bbox", None)
    if phone_bbox is not None:
        px1, py1, px2, py2 = phone_bbox
        cv2.rectangle(frame, (px1, py1), (px2, py2), _PHONE_BOX_COLOR, 2)
        p_conf = getattr(t, "phone_confidence", 0.0)
        cv2.putText(frame, f"phone {p_conf:.2f}", (px1, py1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _PHONE_BOX_COLOR, 1, cv2.LINE_AA)
        # Draw line from phone to nearest wrist
        ph_dist = getattr(t, "phone_hand_distance", -1.0)
        if ph_dist >= 0 and t.keypoints is not None:
            kxy, kconf = t.keypoints, t.keypoints_conf
            for wi in (9, 10):  # wrist L/R
                if kconf is not None and wi < len(kconf) and kconf[wi] >= 0.25:
                    wx, wy = int(kxy[wi][0]), int(kxy[wi][1])
                    pc = ((px1 + px2) // 2, (py1 + py2) // 2)
                    cv2.line(frame, pc, (wx, wy), _PHONE_BOX_COLOR, 1, cv2.LINE_AA)

    # Monitor box (nearest, regardless of interaction)
    m_bbox = getattr(t, "monitor_bbox", None)
    m_dist = getattr(t, "monitor_distance", -1.0)
    m_conf = getattr(t, "monitor_confidence", 0.0)
    # In debug, always show the nearest monitor even if not "facing"
    # We need the raw nearest_monitor from the detector — monitor_bbox
    # is only set when raw_monitor=True. For debug we show whatever is nearest.
    if m_bbox is not None:
        mx1, my1, mx2, my2 = m_bbox
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), _MONITOR_COLOR, 2)
        cv2.putText(frame, f"mon {m_conf:.2f}", (mx1, my1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _MONITOR_COLOR, 1, cv2.LINE_AA)
        cv2.line(frame, ((mx1 + mx2) // 2, (my1 + my2) // 2),
                 ((x1 + x2) // 2, y1), _MONITOR_COLOR, 1)

    # Raw measurement labels
    name = t.name if t.name and t.name != "Unknown" else f"Track {t.track_id}"
    head = getattr(t, "head_orientation", "UNKNOWN")
    yaw = getattr(t, "head_yaw", 0.0)
    pitch = getattr(t, "head_pitch", 0.0)
    move = getattr(t, "movement_score", 0.0)
    ph_dist = getattr(t, "phone_hand_distance", -1.0)

    lines = [
        f"{name}  #{t.track_id}",
        f"Posture: {t.posture.upper()}",
        f"Head: {head} yaw={yaw:.2f} pitch={pitch:.2f}",
        f"Move: {move:.3f}",
    ]
    if phone_bbox is not None:
        p_conf = getattr(t, "phone_confidence", 0.0)
        lines.append(f"Phone: conf={p_conf:.2f} hand_d={ph_dist:.0f}px")
    else:
        lines.append("Phone: none detected")
    if m_dist >= 0:
        lines.append(f"Monitor: conf={m_conf:.2f} dist={m_dist:.0f}px facing={'Y' if t.monitor_facing else 'N'}")
    else:
        lines.append("Monitor: none nearby")

    _draw_label_block(frame, x1, y1, lines, _DEBUG_BG)


def _draw_activity_tracks(
    frame: np.ndarray,
    tracked: list,
    show_skeleton: bool = False,
    activity_filter: str = "all",
) -> None:
    """Person boxes labelled with identity, current activity and live timers.

    Drawn from the activity analyzer's annotations (name/activity/
    activity_seconds/movement_score), which only exist once the analyzer has
    seen this track -- an un-annotated track still renders, just with its
    raw posture.
    """
    activity_filter = (activity_filter or "all").lower().strip()

    if activity_filter == "debug":
        for t in tracked:
            _draw_debug_track(frame, t)
        return

    for t in tracked:
        if not _passes_filter(t, activity_filter):
            continue
        x1, y1, x2, y2 = t.face.bbox
        activity = getattr(t, "activity", "UNKNOWN") or "UNKNOWN"
        if activity_filter == "body":
            primary = getattr(t, "primary_activity", None)
            activity = primary or (t.posture.upper() if t.posture != "unknown" else "UNKNOWN")
        color = _ACTIVITY_LABEL_COLORS.get(activity, _POSTURE_COLORS.get(t.posture, (150, 150, 150)))

        if t.monitor_bbox is not None:
            mx1, my1, mx2, my2 = t.monitor_bbox
            cv2.rectangle(frame, (mx1, my1), (mx2, my2), _MONITOR_COLOR, 1)
            cv2.line(frame, ((mx1 + mx2) // 2, (my1 + my2) // 2),
                     ((x1 + x2) // 2, y1), _MONITOR_COLOR, 1)

        # Phone box on non-debug views too
        phone_bbox = getattr(t, "phone_bbox", None)
        if phone_bbox is not None and activity_filter in ("all", "phone") and t.phone_in_use:
            px1, py1, px2, py2 = phone_bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), _PHONE_BOX_COLOR, 2)

        if show_skeleton:
            _draw_skeleton(frame, t)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        name = t.name if t.name and t.name != "Unknown" else f"Track {t.track_id}"
        lines = [
            f"{name}  #{t.track_id}",
            f"{activity}  {_fmt_dur(getattr(t, 'activity_seconds', 0.0))}",
        ]
        head = getattr(t, "head_orientation", "UNKNOWN")
        if head and head != "UNKNOWN":
            lines.append(f"Head: {head}")

        extras = []
        if activity_filter in ("all", "phone") and t.phone_in_use:
            extras.append("PHONE")
        if activity_filter in ("all", "computer") and t.computer_in_use:
            extras.append("PC")
        if activity_filter in ("all", "monitor") and t.monitor_facing:
            extras.append("MONITOR")
        stationary = getattr(t, "stationary_seconds", 0.0)
        if stationary > 0 and activity_filter in ("all", "body"):
            extras.append(f"still {_fmt_dur(stationary)}")
        if activity_filter in ("all", "body"):
            extras.append(f"move {getattr(t, 'movement_score', 0.0):.2f}")
        if extras:
            lines.append("  ".join(extras))

        _draw_label_block(frame, x1, y1, lines, color)


def draw_overlay(
    result: PipelineResult,
    camera_name: str,
    head_count_source: str = "face",
    show_hud: bool = True,
    draw_activity: bool = False,
    activity_skeleton: bool = False,
    activity_filter: str = "all",
    # False hides the head_count_source boxes (face/person/ocr/...) so the
    # "Body movement" view isn't also covered in face boxes. Detection and
    # recognition are unaffected -- this is purely what gets drawn.
    draw_primary: bool = True,
) -> np.ndarray:
    """Render bounding boxes and HUD onto a copy of result.frame.

    head_count_source controls which boxes are drawn and what the HUD says:
      'face'           -- green face boxes only (YOLO)
      'head'/'person'  -- cyan body/head boxes only (YOLO Person)
      'attendance'     -- orange/violet face boxes (InsightFace SCRFD + ArcFace)
      'activity'       -- boxes colored by posture, labeled with phone/PC use
      'ocr'            -- magenta once a text region's reading is voted-stable,
                          gray while still accumulating votes; label shows the
                          recognized text + confidence (+ vote count once confirmed)
    """
    frame = result.frame.copy()
    source = head_count_source.lower().strip()
    if not draw_primary:
        # No branch below matches this, so the primary boxes are skipped
        # while the HUD and the activity overlay still render.
        source = "__hidden__"

    # --- Draw face boxes (green - YOLO Face) ---
    if source == "face":
        for t in result.tracked_faces:
            x1, y1, x2, y2 = t.face.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), _FACE_COLOR, 2)
            if t.name != "Unknown":
                label = f"{t.name} (#{t.track_id})"
            else:
                label = f"Face #{t.track_id} {t.face.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), _FACE_COLOR, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # --- Draw attendance face boxes (InsightFace SCRFD + ArcFace) ---
    elif source == "attendance":
        for t in result.tracked_faces:
            x1, y1, x2, y2 = t.face.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), _ATTENDANCE_COLOR, 2)
            if t.name != "Unknown":
                label = f"✓ {t.name} (#{t.track_id})"
            else:
                label = f"Att Face #{t.track_id} {t.face.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), _ATTENDANCE_COLOR, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # --- Draw head/body boxes (cyan) ---
    elif source in ("head", "person"):
        for t in result.tracked_heads:
            x1, y1, x2, y2 = t.face.bbox  # bbox field reused for body bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), _HEAD_COLOR, 2)
            label = f"Head #{t.track_id}  {t.face.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), _HEAD_COLOR, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # --- Draw activity boxes, labelled with identity + activity + timers ---
    elif source == "activity":
        _draw_activity_tracks(
            frame, result.tracked_activity,
            show_skeleton=activity_skeleton, activity_filter=activity_filter,
        )

    # --- Draw OCR text regions (magenta once voted-stable, gray while pending) ---
    elif source == "ocr":
        for t in result.tracked_ocr:
            x1, y1, x2, y2 = t.face.bbox
            confirmed = t.text_votes >= _OCR_MIN_VOTES_TO_CONFIRM and bool(t.text)
            color = _OCR_CONFIRMED_COLOR if confirmed else _OCR_PENDING_COLOR
            if t.text_quad:
                pts = np.array(t.text_quad, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], True, color, 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if confirmed:
                label = f"\"{t.text}\" {t.text_confidence:.2f} (x{t.text_votes})"
            elif t.text:
                label = f"...{t.text}? {t.text_confidence:.2f}"
            else:
                label = "..."
            if t.text_moving:
                label = "[moving] " + label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(th + 8, y1)
            cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 4, label_y), color, -1)
            text_color = (0, 0, 0) if confirmed else (255, 255, 255)
            cv2.putText(frame, label, (x1 + 2, label_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Activity analysis can run concurrently with any other mode (see
    # FacePipeline's activity_always_on), so its annotations are drawn in
    # addition to the selected mode's boxes rather than instead of them --
    # otherwise running attendance for identity would hide every activity
    # label.
    if draw_activity and source != "activity" and result.tracked_activity:
        _draw_activity_tracks(
            frame, result.tracked_activity,
            show_skeleton=activity_skeleton, activity_filter=activity_filter,
        )

    if show_hud:
        _draw_hud(frame, result, camera_name, source)
    return frame


def to_square(frame: np.ndarray, size: int = 1080, pad_color: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Letterbox `frame` into a size x size square for display, preserving
    aspect ratio (pads with `pad_color` instead of cropping or stretching --
    a stretched or cropped feed reads as unpolished in front of a client).

    Independent of the camera's native capture resolution -- call this last,
    after draw_overlay(), so boxes/HUD scale with the image for free instead
    of needing their own coordinate math.
    """
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

    canvas = np.full((size, size, 3), pad_color, dtype=np.uint8)
    y_off = (size - new_h) // 2
    x_off = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def _draw_hud(
    frame: np.ndarray,
    result: PipelineResult,
    camera_name: str,
    source: str,
) -> None:
    color = _STATE_COLORS.get(result.connection_state, (255, 255, 255))

    lines = [
        (f"Camera: {camera_name}", (255, 255, 255)),
        (f"Connection: {result.connection_state.upper()}", color),
        (f"Resolution: {result.resolution[0]}x{result.resolution[1]}", (255, 255, 255)),
        (f"Capture FPS: {result.capture_fps:.1f}  Process FPS: {result.process_fps:.1f}", (255, 255, 255)),
        (f"Latency: {result.latency_ms:.0f} ms", (255, 255, 255)),
    ]

    if source == "face":
        lines.append((f"Faces detected: {len(result.tracked_faces)}", _FACE_COLOR))
    elif source == "attendance":
        lines.append((f"Attendance faces: {len(result.tracked_faces)}", _ATTENDANCE_COLOR))
    elif source in ("head", "person"):
        lines.append((f"Heads/bodies: {len(result.tracked_heads)}", _HEAD_COLOR))
    elif source == "activity":
        lines.append((f"Activity: {len(result.tracked_activity)} people", _POSTURE_COLORS["sitting"]))
    elif source == "ocr":
        confirmed = sum(1 for t in result.tracked_ocr if t.text_votes >= _OCR_MIN_VOTES_TO_CONFIRM and t.text)
        lines.append((f"OCR: {confirmed}/{len(result.tracked_ocr)} regions confirmed", _OCR_CONFIRMED_COLOR))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (370, 18 + 22 * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 22
    for text, col in lines:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        y += 22


_BAR_H = 56
_BTN_W = 220
_BTN_GAP = 10
_BTN_ACTIVE = (0, 140, 255)
_BTN_IDLE = (70, 70, 70)
_BAR_BG = (30, 30, 30)


def _fit(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """Letterbox `frame` into a w x h black canvas, preserving aspect ratio."""
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    canvas[y:y + nh, x:x + nw] = cv2.resize(frame, (nw, nh), interpolation=interp)
    return canvas


class MultiCameraDisplay:
    """One window for every camera, with a button bar to pick which to watch.

    Buttons: one per camera ("1  Camera 01", "2  Inside", ...) plus "Both"
    (or "All" for 3+ cameras). Click them, or press 1..9 for a single camera
    and 0 / a / b for all of them. q or Esc quits.
    """

    def __init__(
        self,
        camera_names: list[str],
        head_count_source: str = "face",
        window_name: str = "CCTV Cameras",
        size: tuple[int, int] = (1600, 900),
    ) -> None:
        self._names = list(camera_names)
        self._source = head_count_source.lower().strip()
        self._window_name = window_name
        self._w, self._h = size
        self.selected: int | None = None  # None = all cameras
        self._view_changed = True

        all_label = "Both" if len(self._names) == 2 else "All"
        labels = [(f"{i + 1}  {n}", i) for i, n in enumerate(self._names)]
        if len(self._names) > 1:
            labels.append((all_label, None))
        self._buttons: list[tuple[int, int, str, int | None]] = []
        x = _BTN_GAP
        for text, value in labels:
            self._buttons.append((x, x + _BTN_W, text, value))
            x += _BTN_W + _BTN_GAP

        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, self._w, self._h + _BAR_H)
        cv2.setMouseCallback(self._window_name, self._on_mouse)

    def _select(self, value: int | None) -> None:
        if value != self.selected:
            self.selected = value
            self._view_changed = True

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or y >= _BAR_H:
            return
        for x1, x2, _, value in self._buttons:
            if x1 <= x < x2:
                self._select(value)
                return

    def is_visible(self, index: int) -> bool:
        return self.selected is None or self.selected == index

    def consume_view_changed(self) -> bool:
        changed, self._view_changed = self._view_changed, False
        return changed

    def _tile(self, result: PipelineResult | None, name: str, w: int, h: int) -> np.ndarray:
        if result is None:
            tile = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(tile, f"Waiting for {name}...", (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
            return tile
        # Boxes are drawn at native resolution; the HUD is drawn after
        # scaling so its text stays readable in the smaller grid tiles.
        frame = draw_overlay(result, camera_name=name,
                             head_count_source=self._source, show_hud=False)
        tile = _fit(frame, w, h)
        _draw_hud(tile, result, name, self._source)
        return tile

    def render(self, results: list[PipelineResult | None]) -> None:
        canvas = np.zeros((self._h + _BAR_H, self._w, 3), dtype=np.uint8)
        canvas[:_BAR_H] = _BAR_BG
        for x1, x2, text, value in self._buttons:
            color = _BTN_ACTIVE if value == self.selected else _BTN_IDLE
            cv2.rectangle(canvas, (x1, 8), (x2, _BAR_H - 8), color, -1)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(canvas, text, (x1 + (_BTN_W - tw) // 2, (_BAR_H + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if self.selected is not None:
            i = self.selected
            canvas[_BAR_H:] = self._tile(results[i], self._names[i], self._w, self._h)
        else:
            n = len(self._names)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            tw, th = self._w // cols, self._h // rows
            for i in range(n):
                r, c = divmod(i, cols)
                y, x = _BAR_H + r * th, c * tw
                canvas[y:y + th, x:x + tw] = self._tile(results[i], self._names[i], tw, th)
                if c:
                    cv2.line(canvas, (x, y), (x, y + th), (90, 90, 90), 2)

        cv2.imshow(self._window_name, canvas)

    def poll_quit(self) -> bool:
        """Handles view-switch keys; True if the user pressed q/Esc or closed the window."""
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return True
        if ord("1") <= key <= ord("9") and key - ord("1") < len(self._names):
            self._select(key - ord("1"))
        elif key in (ord("0"), ord("a"), ord("b")):
            self._select(None)
        try:
            if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
                return True
        except cv2.error:
            return True
        return False

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._window_name)
        except cv2.error:
            pass


class Display:
    def __init__(self, window_name: str = WINDOW_NAME) -> None:
        self._window_name = window_name
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)

    def show(self, frame: np.ndarray) -> None:
        cv2.imshow(self._window_name, frame)

    def poll_quit(self) -> bool:
        """Returns True if the user pressed 'q' or Esc, or closed the window."""
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return True
        try:
            if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
                return True
        except cv2.error:
            return True
        return False

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._window_name)
        except cv2.error:
            # Window already gone (user closed it, or Qt tore down the GUI
            # when another camera's window was destroyed first).
            pass
