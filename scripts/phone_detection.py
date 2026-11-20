# Stable CCTV mobile-phone detection and tracking
# Replacement for the original phone_detection.py
#
# Main improvements:
# 1. Person IDs remain stable with an IoU tracker.
# 2. Phones have their OWN temporal tracker.
# 3. Phone boxes are EMA-smoothed to remove detector jitter.
# 4. A phone survives temporary YOLO misses instead of disappearing.
# 5. Once a phone is assigned to a person, ownership is locked temporarily.
# 6. Full-frame + upper-body crop detections are de-duplicated.
# 7. The original CLI style is preserved.

from __future__ import annotations

import argparse
import itertools
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ---------------------------- COCO classes ---------------------------------

CLASS_PERSON = 0
CLASS_PHONE = 67

# COCO pose keypoints
KP_WRIST_L, KP_WRIST_R = 9, 10
KP_MIN_CONF = 0.20


# ---------------------------- Configuration --------------------------------

DEFAULT_PHONE_CONF = 0.10
DEFAULT_PERSON_CONF = 0.35
DEFAULT_IMGSZ = 960

POSE_IOU_MATCH = 0.20
PHONE_OWNER_MIN_SCORE = 0.32

# Person phone-status smoothing
SMOOTH_WINDOW = 7
PHONE_MIN_VOTES = 2

# Phone temporal tracking
PHONE_MAX_MISSED = 10
PHONE_MIN_HITS = 2
PHONE_MATCH_MAX_DIST = 0.22       # centre distance / person height
PHONE_MATCH_MIN_IOU = 0.02

# Bounding-box smoothing
PHONE_EMA_ALPHA = 0.45
PHONE_VELOCITY_ALPHA = 0.35


# ---------------------------- Geometry --------------------------------------

def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    if inter <= 0:
        return 0.0

    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter

    return inter / denom if denom > 0 else 0.0


def containment(inner, outer) -> float:
    area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])

    if area <= 0:
        return 0.0

    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])

    return (
        max(0, ix2 - ix1) * max(0, iy2 - iy1) / area
    )


def expand_box(box, mx, my, w, h):
    x1, y1, x2, y2 = box

    dx = int((x2 - x1) * mx)
    dy = int((y2 - y1) * my)

    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(w, x2 + dx),
        min(h, y2 + dy),
    )


def clip_box(box, w, h):
    x1, y1, x2, y2 = box

    return (
        max(0, min(w - 1, int(x1))),
        max(0, min(h - 1, int(y1))),
        max(0, min(w, int(x2))),
        max(0, min(h, int(y2))),
    )


def box_center(box):
    return (
        (box[0] + box[2]) * 0.5,
        (box[1] + box[3]) * 0.5,
    )


def center_distance(a, b):
    ax, ay = box_center(a)
    bx, by = box_center(b)

    return float(np.hypot(ax - bx, ay - by))


def nms_phone_boxes(detections, iou_threshold=0.30):
    """
    The same phone can be detected twice:
      - once by the full-frame pass
      - once by the enlarged crop pass

    Remove duplicate boxes before tracking.
    """
    if not detections:
        return []

    detections = sorted(
        detections,
        key=lambda x: x[4],
        reverse=True,
    )

    kept = []

    for det in detections:
        if all(
            iou(det[:4], other[:4]) < iou_threshold
            for other in kept
        ):
            kept.append(det)

    return kept


# ---------------------------- Person tracker --------------------------------

class PersonTracker:
    """
    Lightweight IoU tracker for stable person IDs.

    This is intentionally kept separate from the phone tracker.
    """

    def __init__(
        self,
        iou_threshold=0.25,
        max_missed=20,
    ):
        self.tracks = {}
        self.ids = itertools.count(1)

        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def update(self, boxes):
        assigned = []
        used_ids = set()
        assigned_boxes = set()

        candidates = []

        for tid, data in self.tracks.items():
            for index, box in enumerate(boxes):
                candidates.append(
                    (
                        iou(data["box"], box),
                        tid,
                        index,
                    )
                )

        candidates.sort(reverse=True)

        for score, tid, index in candidates:
            if score < self.iou_threshold:
                break

            if tid in used_ids:
                continue

            if index in assigned_boxes:
                continue

            used_ids.add(tid)
            assigned_boxes.add(index)

            self.tracks[tid] = {
                "box": boxes[index],
                "missed": 0,
            }

            assigned.append(
                (
                    tid,
                    boxes[index],
                )
            )

        for index, box in enumerate(boxes):
            if index in assigned_boxes:
                continue

            tid = next(self.ids)

            self.tracks[tid] = {
                "box": box,
                "missed": 0,
            }

            assigned.append(
                (
                    tid,
                    box,
                )
            )

        # Age tracks that were not detected.
        for tid in list(self.tracks.keys()):
            if tid in used_ids:
                continue

            # A newly-created track was already returned above.
            if any(item[0] == tid for item in assigned):
                continue

            self.tracks[tid]["missed"] += 1

            if self.tracks[tid]["missed"] > self.max_missed:
                del self.tracks[tid]

        return assigned


# ---------------------------- Phone tracker ---------------------------------

@dataclass
class PhoneTrack:
    track_id: int
    bbox: tuple
    score: float

    owner_id: Optional[int] = None

    hits: int = 1
    missed: int = 0
    age: int = 1

    vx: float = 0.0
    vy: float = 0.0

    confirmed: bool = False

    def predict(self):
        """
        Predict the phone location during detector misses.

        The longer the miss, the more the velocity is applied.
        """
        if self.missed <= 0:
            return self.bbox

        x1, y1, x2, y2 = self.bbox

        dx = self.vx * self.missed
        dy = self.vy * self.missed

        return (
            x1 + dx,
            y1 + dy,
            x2 + dx,
            y2 + dy,
        )

    def update(self, new_box, score):
        old_cx, old_cy = box_center(self.bbox)
        new_cx, new_cy = box_center(new_box)

        raw_vx = new_cx - old_cx
        raw_vy = new_cy - old_cy

        self.vx = (
            (1 - PHONE_VELOCITY_ALPHA) * self.vx
            + PHONE_VELOCITY_ALPHA * raw_vx
        )

        self.vy = (
            (1 - PHONE_VELOCITY_ALPHA) * self.vy
            + PHONE_VELOCITY_ALPHA * raw_vy
        )

        # EMA smoothing prevents YOLO's box from jumping around.
        self.bbox = tuple(
            int(
                (1 - PHONE_EMA_ALPHA) * old
                + PHONE_EMA_ALPHA * new
            )
            for old, new in zip(self.bbox, new_box)
        )

        self.score = max(
            float(score),
            self.score * 0.85,
        )

        self.hits += 1
        self.missed = 0
        self.age += 1

        if self.hits >= PHONE_MIN_HITS:
            self.confirmed = True

    def mark_missed(self):
        self.missed += 1
        self.age += 1


class PhoneTracker:
    """
    Actual temporal tracker for phones.

    This is the important difference from the original script.

    Original:
        YOLO sees phone -> draw box
        YOLO misses phone -> remove box

    New:
        YOLO sees phone -> update phone track
        YOLO misses phone -> predict and keep track alive
    """

    def __init__(
        self,
        max_missed=PHONE_MAX_MISSED,
        max_dist=PHONE_MATCH_MAX_DIST,
        min_iou=PHONE_MATCH_MIN_IOU,
    ):
        self.tracks = {}
        self.ids = itertools.count(1)

        self.max_missed = max_missed
        self.max_dist = max_dist
        self.min_iou = min_iou

    def _match_cost(
        self,
        track,
        detection,
        persons_by_id,
    ):
        predicted = track.predict()
        detection_box = detection[:4]

        # Use the owner's body height as the scale for distance.
        person_height = 500.0

        if track.owner_id in persons_by_id:
            person = persons_by_id[track.owner_id]
            person_height = max(
                1.0,
                person[3] - person[1],
            )

        elif persons_by_id:
            # If ownership isn't known yet, use the smallest nearby person
            # height as a reasonable scale.
            for person in persons_by_id.values():
                person_height = min(
                    person_height,
                    max(1.0, person[3] - person[1]),
                )

        distance = (
            center_distance(predicted, detection_box)
            / person_height
        )

        overlap = iou(
            predicted,
            detection_box,
        )

        # Existing overlap is the strongest match.
        if overlap >= self.min_iou:
            return distance - 0.35 * overlap

        # Also allow a phone to move between frames.
        if distance <= self.max_dist:
            return distance + 0.15

        return 999.0

    def update(
        self,
        detections,
        persons_by_id,
    ):
        matched_tracks = set()
        matched_detections = set()

        candidates = []

        for tid, track in self.tracks.items():
            for index, detection in enumerate(detections):
                cost = self._match_cost(
                    track,
                    detection,
                    persons_by_id,
                )

                if cost < 999:
                    candidates.append(
                        (
                            cost,
                            tid,
                            index,
                        )
                    )

        candidates.sort(
            key=lambda item: item[0]
        )

        # Greedy best-match assignment.
        for cost, tid, index in candidates:
            if tid in matched_tracks:
                continue

            if index in matched_detections:
                continue

            self.tracks[tid].update(
                detections[index][:4],
                detections[index][4],
            )

            matched_tracks.add(tid)
            matched_detections.add(index)

        # Create new phone tracks.
        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue

            tid = next(self.ids)

            self.tracks[tid] = PhoneTrack(
                track_id=tid,
                bbox=tuple(
                    map(int, detection[:4])
                ),
                score=float(detection[4]),
            )

        # Keep unmatched phone tracks alive.
        for tid in list(self.tracks.keys()):
            if tid in matched_tracks:
                continue

            track = self.tracks[tid]

            # A newly created track should not immediately become "missed".
            if track.age <= 1:
                continue

            track.mark_missed()

            if track.missed > self.max_missed:
                del self.tracks[tid]

        return list(self.tracks.values())


# ---------------------------- Ownership -------------------------------------

def phone_person_score(
    phone,
    person,
    pose,
    img_w,
    img_h,
):
    """
    Score how likely the phone belongs to this person.

    Wrist is strongest.
    Body containment and vertical position are supporting evidence.
    """

    px1, py1, px2, py2 = person

    person_height = py2 - py1

    if px2 - px1 <= 0 or person_height <= 0:
        return 0.0

    cx, cy = box_center(phone)

    wrist_score = 0.0

    if pose is not None:
        keypoints, confidences = pose

        for index in (
            KP_WRIST_L,
            KP_WRIST_R,
        ):
            if index >= len(confidences):
                continue

            if confidences[index] < KP_MIN_CONF:
                continue

            distance = (
                np.hypot(
                    cx - keypoints[index][0],
                    cy - keypoints[index][1],
                )
                / max(person_height, 1)
            )

            if distance <= 0.08:
                score = 1.0

            elif distance <= 0.15:
                score = 0.75

            elif distance <= 0.25:
                score = 0.35

            else:
                score = 0.0

            wrist_score = max(
                wrist_score,
                score,
            )

    body_score = (
        min(
            containment(
                phone,
                expand_box(
                    person,
                    0.08,
                    0.08,
                    img_w,
                    img_h,
                ),
            ),
            1.0,
        )
        * 0.20
    )

    relative_y = (
        (cy - py1)
        / max(person_height, 1)
    )

    zone_score = (
        0.15
        if 0.25 <= relative_y <= 1.05
        else 0.0
    )

    return (
        wrist_score * 0.65
        + body_score
        + zone_score
    )


# ---------------------------- Detector --------------------------------------

class PhoneDetector:
    def __init__(
        self,
        obj_model,
        pose_model,
        conf=DEFAULT_PHONE_CONF,
        person_conf=DEFAULT_PERSON_CONF,
        imgsz=DEFAULT_IMGSZ,
        crop_pass=True,
        crop_upscale=2.5,
        crop_imgsz=768,
        device=None,
        phone_max_missed=PHONE_MAX_MISSED,
    ):
        self.obj = YOLO(obj_model)
        self.pose = YOLO(pose_model)

        self.conf = conf
        self.person_conf = person_conf

        self.imgsz = imgsz
        self.crop_pass = crop_pass
        self.crop_upscale = crop_upscale
        self.crop_imgsz = crop_imgsz

        self.device = device

        self.person_tracker = PersonTracker()

        self.phone_tracker = PhoneTracker(
            max_missed=phone_max_missed
        )

        self.history = {}

        self.frame_index = 0

    # ------------------------------------------------------------------

    def _crop_pass_phones(
        self,
        frame,
        persons,
    ):
        """
        Enlarged upper-body pass.

        This helps when the phone is too small for the full-frame detector.
        """

        h, w = frame.shape[:2]

        crops = []
        origins = []

        for _tid, (
            x1,
            y1,
            x2,
            y2,
        ) in persons:

            person_w = x2 - x1
            person_h = y2 - y1

            cx1 = max(
                0,
                int(x1 - 0.15 * person_w),
            )

            cx2 = min(
                w,
                int(x2 + 0.15 * person_w),
            )

            cy1 = max(
                0,
                int(y1),
            )

            # Include most of the body. Phones can be held lower
            # than the chest for seated people.
            cy2 = min(
                h,
                int(y1 + 0.80 * person_h),
            )

            if (
                cx2 - cx1 < 32
                or cy2 - cy1 < 32
            ):
                continue

            crop = frame[
                cy1:cy2,
                cx1:cx2,
            ]

            crop = cv2.resize(
                crop,
                None,
                fx=self.crop_upscale,
                fy=self.crop_upscale,
                interpolation=cv2.INTER_CUBIC,
            )

            crops.append(crop)
            origins.append(
                (
                    cx1,
                    cy1,
                )
            )

        if not crops:
            return []

        results = self.obj.predict(
            crops,
            imgsz=self.crop_imgsz,
            conf=self.conf,
            classes=[CLASS_PHONE],
            device=self.device,
            verbose=False,
        )

        output = []

        for result, (
            origin_x,
            origin_y,
        ) in zip(
            results,
            origins,
        ):
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, score in zip(
                boxes,
                scores,
            ):
                x1, y1, x2, y2 = box

                scale = self.crop_upscale

                phone = (
                    max(
                        0,
                        int(
                            origin_x
                            + x1 / scale
                        ),
                    ),
                    max(
                        0,
                        int(
                            origin_y
                            + y1 / scale
                        ),
                    ),
                    min(
                        w,
                        int(
                            origin_x
                            + x2 / scale
                        ),
                    ),
                    min(
                        h,
                        int(
                            origin_y
                            + y2 / scale
                        ),
                    ),
                    float(score),
                )

                if (
                    phone[2] > phone[0]
                    and phone[3] > phone[1]
                ):
                    output.append(phone)

        return output

    # ------------------------------------------------------------------

    def _detect_objects(self, frame):
        h, w = frame.shape[:2]

        result = self.obj.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            classes=[
                CLASS_PERSON,
                CLASS_PHONE,
            ],
            device=self.device,
            verbose=False,
        )[0]

        persons_raw = []
        phones = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            for box, score, cls in zip(
                boxes,
                scores,
                classes,
            ):
                clipped = clip_box(
                    box,
                    w,
                    h,
                )

                if cls == CLASS_PERSON:
                    if score >= self.person_conf:
                        persons_raw.append(
                            clipped
                        )

                elif cls == CLASS_PHONE:
                    if score >= self.conf:
                        phones.append(
                            (
                                *clipped,
                                float(score),
                            )
                        )

        # Track people first.
        persons = self.person_tracker.update(
            persons_raw
        )

        # Enlarged second pass for small phones.
        if self.crop_pass:
            phones.extend(
                self._crop_pass_phones(
                    frame,
                    persons,
                )
            )

        # Remove duplicate full-frame/crop detections.
        phones = nms_phone_boxes(
            phones,
            iou_threshold=0.30,
        )

        return persons, phones

    # ------------------------------------------------------------------

    def _detect_pose(
        self,
        frame,
        persons,
    ):
        poses = {}

        result = self.pose.predict(
            frame,
            imgsz=self.imgsz,
            conf=0.30,
            device=self.device,
            verbose=False,
        )[0]

        if (
            result.keypoints is None
            or result.boxes is None
            or len(result.boxes) == 0
        ):
            return poses

        pose_boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        keypoints = (
            result.keypoints.xy
            .cpu()
            .numpy()
        )

        confidences = result.keypoints.conf

        if confidences is not None:
            confidences = (
                confidences
                .cpu()
                .numpy()
            )
        else:
            confidences = np.ones(
                keypoints.shape[:2],
                dtype=np.float32,
            )

        for tid, person_box in persons:

            best_pose = None
            best_overlap = POSE_IOU_MATCH

            for (
                pose_box,
                pose_keypoints,
                pose_confidence,
            ) in zip(
                pose_boxes,
                keypoints,
                confidences,
            ):

                overlap = iou(
                    person_box,
                    tuple(pose_box),
                )

                if overlap > best_overlap:
                    best_overlap = overlap

                    best_pose = (
                        pose_keypoints,
                        pose_confidence,
                    )

            poses[tid] = best_pose

        return poses

    # ------------------------------------------------------------------

    def _assign_phone_owners(
        self,
        phone_tracks,
        persons,
        poses,
        img_w,
        img_h,
    ):
        persons_by_id = {
            tid: box
            for tid, box in persons
        }

        # --------------------------------------------------------------
        # First preserve an existing owner.
        #
        # This is important.
        #
        # Once Phone #1 is associated with Person #56, a temporary pose
        # failure must NOT instantly make the phone owner become None.
        # --------------------------------------------------------------

        for track in phone_tracks:

            if not track.confirmed:
                continue

            if track.owner_id not in persons_by_id:
                track.owner_id = None
                continue

            owner_box = persons_by_id[
                track.owner_id
            ]

            score = phone_person_score(
                track.bbox,
                owner_box,
                poses.get(track.owner_id),
                img_w,
                img_h,
            )

            inside_owner = (
                containment(
                    track.bbox,
                    expand_box(
                        owner_box,
                        0.20,
                        0.20,
                        img_w,
                        img_h,
                    ),
                )
                > 0.05
            )

            # Keep the lock while the phone remains near its owner.
            if (
                track.missed <= PHONE_MAX_MISSED
                and (
                    score >= 0.18
                    or inside_owner
                )
            ):
                continue

            track.owner_id = None

        # --------------------------------------------------------------
        # Assign unowned phones.
        # --------------------------------------------------------------

        for track in phone_tracks:

            if track.owner_id is not None:
                continue

            best_owner = None
            best_score = PHONE_OWNER_MIN_SCORE

            for tid, person_box in persons:

                score = phone_person_score(
                    track.bbox,
                    person_box,
                    poses.get(tid),
                    img_w,
                    img_h,
                )

                if score > best_score:
                    best_score = score
                    best_owner = tid

            if best_owner is not None:
                track.owner_id = best_owner

        return persons_by_id

    # ------------------------------------------------------------------

    def detect(self, frame):
        self.frame_index += 1

        h, w = frame.shape[:2]

        # 1. Person + phone detection.
        persons, phone_detections = (
            self._detect_objects(frame)
        )

        persons_by_id = {
            tid: box
            for tid, box in persons
        }

        # 2. PHONE TRACKING.
        #
        # This is the key anti-flicker stage.
        phone_tracks = self.phone_tracker.update(
            phone_detections,
            persons_by_id,
        )

        # 3. Pose/wrist detection.
        poses = self._detect_pose(
            frame,
            persons,
        )

        # 4. Associate phone tracks to people.
        self._assign_phone_owners(
            phone_tracks,
            persons,
            poses,
            w,
            h,
        )

        # Best active phone for each person.
        owned = {}

        for track in phone_tracks:

            if track.owner_id is None:
                continue

            if (
                track.owner_id not in owned
                or track.score
                > owned[
                    track.owner_id
                ].score
            ):
                owned[
                    track.owner_id
                ] = track

        output_people = []
        live_people = set()

        for tid, person_box in persons:

            live_people.add(tid)

            history = self.history.setdefault(
                tid,
                deque(
                    maxlen=SMOOTH_WINDOW
                ),
            )

            phone = owned.get(tid)

            # Fresh detector evidence.
            detected_now = (
                phone is not None
                and phone.missed == 0
            )

            # Temporal tracker evidence.
            tracked_phone_alive = (
                phone is not None
                and phone.confirmed
                and phone.missed
                <= PHONE_MAX_MISSED
            )

            history.append(
                bool(
                    detected_now
                    or tracked_phone_alive
                )
            )

            # The phone track itself is now strong enough to keep
            # the status stable. The history protects against startup noise.
            in_use = (
                sum(history)
                >= PHONE_MIN_VOTES
                or tracked_phone_alive
            )

            if phone is not None:

                # During a detector miss this is a predicted position.
                phone_box = clip_box(
                    phone.predict(),
                    w,
                    h,
                )

                phone_track_id = (
                    phone.track_id
                )

                phone_missed = (
                    phone.missed
                )

                phone_score = (
                    phone.score
                )

            else:
                phone_box = None
                phone_track_id = None
                phone_missed = 0
                phone_score = 0.0

            output_people.append(
                {
                    "track_id": tid,
                    "bbox": person_box,
                    "phone_in_use": in_use,
                    "phone_bbox": (
                        phone_box
                        if in_use
                        else None
                    ),
                    "phone_track_id": (
                        phone_track_id
                        if in_use
                        else None
                    ),
                    "phone_missed": (
                        phone_missed
                        if in_use
                        else 0
                    ),
                    "score": round(
                        float(phone_score),
                        2,
                    ),
                }
            )

        # Remove old person histories.
        self.history = {
            tid: hist
            for tid, hist
            in self.history.items()
            if tid in live_people
        }

        # Active tracked phones.
        tracked_phones = []

        for track in phone_tracks:

            if (
                not track.confirmed
                or track.missed
                > PHONE_MAX_MISSED
            ):
                continue

            box = clip_box(
                track.predict(),
                w,
                h,
            )

            tracked_phones.append(
                (
                    *box,
                    float(track.score),
                    track.track_id,
                    track.missed,
                    track.owner_id,
                )
            )

        return {
            "persons": output_people,
            "phones": phone_detections,
            "tracked_phones": tracked_phones,
        }


# ---------------------------- Drawing ---------------------------------------

def draw(frame, result):

    # Raw YOLO phone detections are thin orange boxes.
    for (
        x1,
        y1,
        x2,
        y2,
        confidence,
    ) in result["phones"]:

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 165, 255),
            1,
        )

    # Stable person + phone boxes.
    for person in result["persons"]:

        x1, y1, x2, y2 = (
            person["bbox"]
        )

        is_phone = (
            person["phone_in_use"]
        )

        color = (
            (0, 80, 255)
            if is_phone
            else (180, 180, 180)
        )

        # Person box.
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        status = (
            "PHONE"
            if is_phone
            else "no phone"
        )

        label = (
            f"#{person['track_id']} | "
            f"{status}"
        )

        if (
            is_phone
            and person["phone_track_id"]
            is not None
        ):
            label += (
                f" | P"
                f"{person['phone_track_id']}"
            )

        # Show when the phone box is being held by the tracker.
        if (
            is_phone
            and person["phone_missed"] > 0
        ):
            label += (
                f" | hold "
                f"{person['phone_missed']}"
            )

        (
            text_width,
            text_height,
        ), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        cv2.rectangle(
            frame,
            (
                x1,
                max(
                    0,
                    y1 - text_height - 8,
                ),
            ),
            (
                x1 + text_width + 6,
                y1,
            ),
            color,
            -1,
        )

        cv2.putText(
            frame,
            label,
            (
                x1 + 3,
                y1 - 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        # Stable/smoothed phone box.
        if (
            person["phone_bbox"]
            is not None
        ):

            px1, py1, px2, py2 = (
                person["phone_bbox"]
            )

            cv2.rectangle(
                frame,
                (px1, py1),
                (px2, py2),
                (0, 80, 255),
                3,
            )

    return frame


# ---------------------------- Main -------------------------------------------

def main():

    root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    parser = argparse.ArgumentParser(
        description=(
            "Stable CCTV mobile-phone "
            "detection and tracking"
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "image, video, RTSP URL, "
            "or webcam index"
        ),
    )

    parser.add_argument(
        "--obj-model",
        default=str(
            root
            / "models"
            / "yolov8x.pt"
        ),
    )

    parser.add_argument(
        "--pose-model",
        default=str(
            root
            / "models"
            / "yolov8x-pose.pt"
        ),
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_PHONE_CONF,
        help=(
            "phone confidence. "
            "0.10-0.20 is suitable "
            "for small CCTV phones."
        ),
    )

    parser.add_argument(
        "--person-conf",
        type=float,
        default=DEFAULT_PERSON_CONF,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
    )

    parser.add_argument(
        "--crop-pass",
        action="store_true",
        default=True,
        help=(
            "use enlarged upper-body "
            "phone detection"
        ),
    )

    parser.add_argument(
        "--no-crop-pass",
        action="store_false",
        dest="crop_pass",
    )

    parser.add_argument(
        "--crop-upscale",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--crop-imgsz",
        type=int,
        default=768,
    )

    parser.add_argument(
        "--phone-max-missed",
        type=int,
        default=PHONE_MAX_MISSED,
        help=(
            "frames a phone stays alive "
            "after detector misses"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="cuda:0 | cpu | mps",
    )

    parser.add_argument(
        "--show",
        action="store_true",
    )

    parser.add_argument(
        "--save",
        default=None,
        help="annotated output path",
    )

    args = parser.parse_args()

    detector = PhoneDetector(
        obj_model=args.obj_model,
        pose_model=args.pose_model,
        conf=args.conf,
        person_conf=args.person_conf,
        imgsz=args.imgsz,
        crop_pass=args.crop_pass,
        crop_upscale=args.crop_upscale,
        crop_imgsz=args.crop_imgsz,
        device=args.device,
        phone_max_missed=(
            args.phone_max_missed
        ),
    )

    source = args.source

    # ---------------- Image ----------------

    if Path(source).suffix.lower() in (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    ):

        frame = cv2.imread(source)

        if frame is None:
            raise SystemExit(
                f"Cannot read image: {source}"
            )

        result = detector.detect(
            frame
        )

        for person in result["persons"]:

            print(
                f"person #{person['track_id']}: "
                f"{'PHONE' if person['phone_in_use'] else 'no phone'} "
                f"(score {person['score']})"
            )

        output = draw(
            frame,
            result,
        )

        if args.save:
            cv2.imwrite(
                args.save,
                output,
            )

        if args.show:
            cv2.imshow(
                "Stable phone detection",
                output,
            )

            cv2.waitKey(0)

        return

    # ---------------- Video / RTSP / webcam ----------------

    capture = cv2.VideoCapture(
        int(source)
        if source.isdigit()
        else source
    )

    if not capture.isOpened():
        raise SystemExit(
            f"Cannot open source: {source}"
        )

    # Ask OpenCV for the smallest possible live buffer.
    # Some RTSP backends ignore this; that's okay.
    try:
        capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )
    except Exception:
        pass

    writer = None

    try:

        while True:

            ok, frame = (
                capture.read()
            )

            if not ok:
                break

            result = detector.detect(
                frame
            )

            output = draw(
                frame,
                result,
            )

            on_phone = [
                person["track_id"]
                for person
                in result["persons"]
                if person["phone_in_use"]
            ]

            print(
                f"people="
                f"{len(result['persons'])} "
                f"on_phone="
                f"{on_phone}"
            )

            if args.save:

                if writer is None:

                    fps = (
                        capture.get(
                            cv2.CAP_PROP_FPS
                        )
                    )

                    if not fps or fps <= 1:
                        fps = 10

                    writer = cv2.VideoWriter(
                        args.save,
                        cv2.VideoWriter_fourcc(
                            *"mp4v"
                        ),
                        fps,
                        (
                            output.shape[1],
                            output.shape[0],
                        ),
                    )

                writer.write(output)

            if args.show:

                cv2.imshow(
                    "Stable phone detection",
                    output,
                )

                if (
                    cv2.waitKey(1) & 0xFF
                    == ord("q")
                ):
                    break

    finally:

        capture.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

