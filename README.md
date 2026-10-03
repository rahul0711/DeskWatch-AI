# CCTV Face AI — Phase 1: Live Capture + Face Detection

Prove that we can pull live video **directly from the Prama PT-NC123D3-WNM(D2) IP
camera** (no screen capture, no VMS-in-the-middle) and detect/extract faces
from it in real time. No recognition, no attendance, no database yet.

```
Prama IP Camera → RTSP (direct network stream) → OpenCV/FFmpeg decode →
frames → SCRFD face detector (InsightFace/ONNX Runtime) → boxes + IDs →
optional face crop → HUD window
```

The Prama VMS can keep running independently for monitoring — this app talks
to the camera itself, over the network, in parallel.

---

## 1. Technology decision

| Concern | Choice | Rejected | Why |
|---|---|---|---|
| Language | **Python 3.10+** | C++ | Same real-time performance ceiling for this workload (the hot path is OpenCV/ONNX Runtime C++ internals either way); Python is 3-5x faster to iterate on and has first-class bindings for every library below. |
| Stream ingestion | **OpenCV `VideoCapture` (FFmpeg backend)** | GStreamer, DeepStream | Ships with OpenCV, zero extra install on Windows, handles RTSP/H.264/H.265 out of the box. GStreamer/DeepStream give lower latency + hardware decode but need a much heavier, Linux-centric, NVIDIA-specific install — overkill for proving a single-camera pipeline. Revisit for Phase 2+ multi-camera GPU deployment. |
| Face detection | **InsightFace SCRFD** (via `insightface` + ONNX Runtime) | MediaPipe, raw YOLO-face, OpenCV Haar/DNN | SCRFD is fast, accurate, and — critically — ships in the **same library** (`insightface`) as ArcFace face **recognition**. Phase 3 (embeddings) becomes `allowed_modules=["detection", "recognition"]` — a one-line change, same runtime, same bounding boxes. MediaPipe/YOLO would mean gluing a second, unrelated recognition library on top later. |
| Inference runtime | **ONNX Runtime**, `CPUExecutionProvider` by default | Raw PyTorch/TensorRT | Runs anywhere with no GPU required for Phase 1's single camera; swapping to `onnxruntime-gpu` + `CUDAExecutionProvider` later is a config change, not a rewrite. TensorRT/DeepStream stays an option for a future multi-camera, GPU-server deployment. |
| Tracking (temporary IDs) | **Custom minimal IoU tracker** (~50 lines) | ByteTrack/DeepSORT | Phase 1 only needs a face to keep the same "#3" label while it's in frame between detection passes. A full tracker is Phase 2 scope; the module boundary (`app/tracking/`) is already there for it. |
| Display | **`cv2.imshow` window with HUD overlay** | FastAPI + WebSocket + browser UI | Fastest way to visually prove the pipeline; zero web server, zero extra process. `draw_overlay()` is a pure function of a frame + results, so a Phase 2 web/MJPEG endpoint can reuse it without touching capture/detection code. |
| Camera discovery | **ONVIF (best-effort) + brute-force RTSP probing** | Assuming a URL | Prama is a Hikvision-partnered/OEM brand in India, so Hikvision-style `/Streaming/Channels/10x` paths are the leading hypothesis — but `scripts/discover_camera.py` verifies this against the actual camera instead of assuming it, and the app never hard-codes it (fully `.env`-driven). |

**Expected performance (single camera, modern quad-core CPU, no GPU):**
sub-stream at 640×360–480, SCRFD at `det_size=320`, detection every 3rd
frame → roughly 10-15 processed FPS with detection, <150 ms glass-to-box
latency. A GPU (`onnxruntime-gpu`) removes detection as the bottleneck
entirely, which matters once this scales to multiple cameras.

**Hardware requirements (Phase 1):** any Windows/Linux machine with a
4-core CPU, 4 GB RAM, and network access to the camera. No GPU required.

**Future scalability:** the capture layer (`RTSPStream`), detector
(`FaceDetector`), and pipeline (`FacePipeline`) all take their dependencies
as constructor arguments and hold no global state — running N of them (one
process/task per camera) is the Phase 2 multi-camera story, not a redesign.
Swapping SCRFD-only for SCRFD+ArcFace for Phase 3 recognition, or the
IoU tracker for ByteTrack for Phase 2 tracking, are both isolated,
single-module changes.

---

## 2. Investigating the Prama camera

Prama India has historically been Hikvision's joint-venture/OEM partner for
the Indian market, so this camera's ONVIF/RTSP stack is expected to follow
Hikvision conventions:

- RTSP port **554** (standard, changeable in camera network settings)
- ONVIF supported (per the product listing: TLS 1.3, HTTPS, ONVIF, ISAPI)
- Likely stream paths: `/Streaming/Channels/101` (channel 1, main) and
  `/Streaming/Channels/102` (channel 1, sub)
- Whether the VMS is talking to the camera directly or through an NVR
  determines the IP/channel number you'll use — if there's an NVR in the
  path, the RTSP endpoint is usually the **NVR's** IP with a channel-specific
  path, not the camera's IP directly. Check your Prama VMS's device list.

**This is a documented hypothesis, not an assumption baked into the code.**
Run the investigation script against your actual camera before touching
`.env`:

```bash
python scripts/discover_camera.py --ip 192.168.1.64 --user admin --password yourpassword
```

It will:
1. Check TCP reachability on ports 554 (RTSP), 80 (ONVIF/HTTP), 8000.
2. If `onvif-zeep` is installed, ask the camera's ONVIF Media service for
   its **actual** stream URIs (authoritative — not a guess).
3. Brute-force-try a list of known vendor URL conventions
   (Hikvision-style first, Dahua-style as a fallback) by actually opening
   each one and reading a frame, and report which ones work with what
   resolution/FPS.

Take whichever URL it confirms works and put it in `.env`.

---

## 3. Project structure

```
cctv-face-ai/
├── app/
│   ├── config.py          # all settings, loaded from .env — no hard-coded values
│   ├── camera/
│   │   └── rtsp_stream.py # threaded RTSP reader, always-latest-frame, auto-reconnect
│   ├── detection/
│   │   └── face_detector.py  # InsightFace SCRFD wrapper + crop_face()
│   ├── tracking/
│   │   └── simple_tracker.py # minimal IoU tracker for stable "Face #N" IDs
│   ├── processing/
│   │   └── pipeline.py    # orchestrates capture -> detection interval -> tracking
│   ├── ui/
│   │   └── display.py     # HUD overlay + OpenCV window
│   └── utils.py            # shared RollingFps counter
├── scripts/
│   ├── discover_camera.py # Step 1: find the camera's real RTSP/ONVIF endpoints
│   ├── test_connection.py # Step 2/3: smallest possible program — raw frames only
│   └── benchmark.py        # Step 6: FPS/CPU benchmarking against the live camera
├── tests/                  # pure-logic unit tests (no camera/model required)
├── models/                  # InsightFace model cache (downloaded on first run)
├── data/samples/            # optional saved face crops (gitignored)
├── main.py                  # Phase 1 full app: capture + detect + HUD
├── .env.example
└── requirements.txt
```

---

## 4. Setup (clean machine)

### Prerequisites
- Python 3.10 or later
- A network path from this machine to the camera (same LAN/VLAN, or routed
  with the relevant ports open)
- (Optional, for GPU) NVIDIA GPU + CUDA/cuDNN + `onnxruntime-gpu`

### Install

```bash
git clone <this-repo>   # or just use the folder as-is
cd cctv-face-ai
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

> `insightface` builds a couple of small Cython extensions on first install.
> On Windows, if this fails, install "Build Tools for Visual Studio" (C++
> build tools workload) and re-run `pip install -r requirements.txt`.

### Configure

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/Mac
```

Edit `.env`:

```env
CAMERA_NAME=Camera 01
CAMERA_IP=192.168.1.64
CAMERA_RTSP_PORT=554
CAMERA_USERNAME=admin
CAMERA_PASSWORD=your-real-password
CAMERA_RTSP_PATH=/Streaming/Channels/102
```

Never commit `.env` (it's gitignored). `.env.example` has no real
credentials and is safe to commit.

### Run

```bash
# Step 1 — confirm the RTSP endpoint (skip if you already know it works)
python scripts/discover_camera.py --ip 192.168.1.64 --user admin --password yourpassword

# Step 2/3 — smallest possible program: raw frames, no AI
python scripts/test_connection.py

# Step 4/5 — full pipeline: detection + extraction + HUD
python main.py
```

You should see a window titled **"CCTV Face Detection - Phase 1"** with the
live feed, green boxes around detected faces labeled `Face #N  0.9x`, and a
HUD showing camera name, connection state, resolution, capture/process FPS,
latency, and face count. Press **q** or **Esc** to quit.

---

## 5. Testing

### Automated (no camera needed)

```bash
pytest
```

Covers RTSP URL construction from `.env` parts vs. explicit override,
credential masking, and the IoU tracker's ID-stability logic.

### Manual, against the real camera

Run in this order — each step isolates one layer, so a failure tells you
exactly where to look:

```text
[ ] python scripts/discover_camera.py ...   → confirms camera reachable + RTSP works
[ ] python scripts/test_connection.py       → confirms live frames decode (no AI)
[ ] python main.py                          → confirms face detector loads + detects
[ ] set SAVE_FACE_CROPS=true in .env, re-run → confirms face crop extraction to data/samples/
```

Expected checklist by the end:
```
[✓] Camera reachable            (discover_camera.py port checks)
[✓] RTSP connection successful  (test_connection.py opens the stream)
[✓] Live frames received        (test_connection.py prints frame count/FPS)
[✓] Video decoded                (frame.shape printed, window shows video)
[✓] Face detector loaded        (main.py "Face detector ready" log line)
[✓] Face detected                (green box + confidence in the window)
[✓] Face bounding box displayed (HUD "Faces detected: N" > 0)
[✓] Face crop successfully generated (files appear in data/samples/)
```

Also verify recovery behavior:
- **Camera disconnected**: unplug/reboot the camera — HUD should show
  `RECONNECTING` (orange) and recover automatically once it's back, no
  restart needed.
- **Wrong credentials**: set a bad password — `test_connection.py` reports
  "FAILED to open stream" immediately.
- **Invalid RTSP URL**: same failure mode, check the printed masked URL.
- **No face / multiple faces**: point the camera at an empty room, then
  have several people walk in — face count and boxes should update live.

---

## 6. Performance tuning

Defaults in `.env.example` are a reasonable starting point, not measured
truth for your exact hardware — **run `scripts/benchmark.py` and adjust**:

```bash
python scripts/benchmark.py --seconds 30
```

Levers, in order of impact:

- **`CAMERA_RTSP_PATH` → sub-stream vs main stream.** Detection cost scales
  with pixel count; the sub-stream (`102`, typically 640×360 or similar) is
  usually 4x+ cheaper to run SCRFD on than the main stream (`101`,
  1920×1080+), with negligible accuracy loss for face detection at
  classroom/CCTV distances. Display can still use the main stream later if
  desired — Phase 1 uses one stream for both to keep things simple.
- **`DETECTION_SIZE`** — SCRFD's internal input resolution. 320 is a solid
  CPU default; drop to 224 for more speed if faces are large/close, raise
  toward 640 if you need to catch small/far faces and have CPU/GPU headroom.
- **`DETECTION_INTERVAL`** — run the detector every Nth frame; frames in
  between reuse the last boxes via the IoU tracker. Raise this if CPU-bound,
  lower it (toward 1) if faces move fast enough that stale boxes lag
  visibly.
- **`MAX_PROCESS_FPS`** — hard cap on how often the pipeline processes a
  frame at all, independent of how fast the camera delivers them. Keeps CPU
  usage predictable even on a 25-30 FPS camera stream.
- **GPU**: `pip install onnxruntime-gpu` (matching your CUDA version) and
  set `ONNX_PROVIDER=CUDAExecutionProvider` in `.env` — no code changes.

The capture thread always keeps only the *latest* frame (never queues a
backlog), so raising these knobs trades CPU for freshness/accuracy, not for
latency creep.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `discover_camera.py` finds nothing / all ports closed | Wrong IP, camera on a different VLAN, firewall | Confirm the camera's IP from the Prama VMS device list; ping it; check Windows Firewall isn't blocking outbound RTSP. |
| RTSP connection failure but ports are open | Wrong path/channel, or an NVR sits between you and the camera | Re-check whether Prama VMS connects to the camera directly or through an NVR — if an NVR, use the NVR's IP and its channel-specific path, not the camera's IP. |
| Authentication failure | Wrong username/password, or RTSP auth disabled/misconfigured on camera | Verify credentials work in the Prama VMS itself first; check camera's Configuration > Network > Advanced for RTSP auth settings. |
| Black screen, window opens but no video | Codec mismatch, or stream opened but never delivers frames | Try the main stream instead of sub-stream (`CAMERA_RTSP_PATH=/Streaming/Channels/101`); confirm with VLC (`Media > Open Network Stream` → same RTSP URL) as an independent sanity check. |
| H.264/H.265 decode errors in console | Camera set to H.265 but local FFmpeg build lacks HEVC support | Set the camera's video codec to H.264 in its web UI (Configuration > Video), or ensure your OpenCV/FFmpeg build includes an HEVC decoder. |
| High CPU usage | Main stream resolution too high for detection, or `DETECTION_INTERVAL=1` | Switch to sub-stream, raise `DETECTION_SIZE` down, raise `DETECTION_INTERVAL` up — see Section 6. |
| Low FPS / laggy boxes | CPU-bound detection | Lower `DETECTION_SIZE`, raise `DETECTION_INTERVAL`, or switch to GPU (`onnxruntime-gpu`). |
| Camera disconnects repeatedly | Wi-Fi camera with poor signal, or `CAMERA_RTSP_TRANSPORT=udp` on a lossy network | Use TCP transport (`CAMERA_RTSP_TRANSPORT=tcp`, the default) — UDP packet loss shows up as corrupt/green frames or drops. |
| `pip install insightface` fails to build | Missing C++ build tools on Windows | Install "Build Tools for Visual Studio" (Desktop development with C++ workload), then retry. |
| First run hangs for a while at "Loading face detector..." | InsightFace is downloading the `buffalo_l` model pack (~300 MB) into `models/` (actually `~/.insightface`) | Normal on first run only; subsequent runs load from the local cache instantly. |

---

## 8. What's deliberately NOT in Phase 1

No student database, no attendance logic, no face identification/matching,
no MySQL, no Node.js/React Native, no auth, no admin dashboard. The only
question Phase 1 answers is: **can we reliably get live Prama footage and
detect/extract faces from it?** Once that's solid, Phase 2 (real tracking)
and Phase 3 (ArcFace embeddings + face database) build directly on this
same `RTSPStream` → `FaceDetector` → `FacePipeline` foundation.

---

## 9. Scene-text OCR (walls, signs, boards, carried items)

A fifth detection mode alongside face/person/attendance/activity:
`HEAD_COUNT_SOURCE=ocr`, or the "🔤 OCR" button per camera in the web UI.
Reads text anywhere in the 1920x1080 frame -- static (walls, signs, boards)
and moving (papers/cards/phones/clothes carried by walking people) -- with
multi-frame voting so a moving reading doesn't flicker between misreads.

**Pipeline** (`app/detection/ocr_detector.py`): a full-frame pass runs
PaddleOCR's PP-OCR text DETECTION model (a DBNet-family detector) at native
1080p to catch static text anywhere in frame; a second pass tracks
person/book/cell-phone/laptop boxes via the same YOLO11m + BoT-SORT
tracker `ActivityDetector` already uses, then re-runs detection on each
tracked box's padded, upscaled crop to catch small moving text that's
below what the full-frame pass alone can resolve. Every found region goes
through: `cv2.warpPerspective` off the detector's own quadrilateral
(rotation/perspective correction, not just an axis-aligned crop),
upscaling if short, a CLAHE contrast boost, then PP-OCR text RECOGNITION
for the string + confidence. `app/tracking/text_tracker.py` (`TextTracker`)
then IoU-matches regions across frames and votes on the most consistent
reading -- a region needs 2+ agreeing votes before the UI marks it
"confirmed" (solid magenta on the video overlay; bold in the results
table) instead of "pending" (gray; still accumulating votes).

**Why PaddleOCR, not MMOCR:** the ask was DBNet++ via MMOCR, but MMOCR's
detection backbones depend on `mmcv`, which has no working build for macOS
arm64 (no CUDA, no prebuilt wheel) -- attempting it on this Apple Silicon
dev machine would mean sinking real time into a broken install. PP-OCR's
detector is also a DBNet-family model (architecturally the same lineage as
DBNet++), ships official CPU/arm64 wheels, and is actively maintained, so
this gets the same detector family without the unbuildable dependency. On
a Linux/CUDA box, MMOCR + DBNet++ would be a drop-in swap for
`app/detection/ocr_detector.py`'s detection call without touching anything
else in the pipeline.

**Performance** (measured on this CPU-only Apple Silicon Mac -- no GPU
acceleration for `paddlepaddle` here, unlike the PyTorch-based detectors
elsewhere in this app which get MPS): the default "mobile" det/rec models
run a full-frame pass in ~1.3-1.8s; PaddleOCR's own default "medium"
models (`OCR_DET_MODEL=PP-OCRv6_medium_det` /
`OCR_REC_MODEL=PP-OCRv6_medium_rec`) are ~5-7x slower but more accurate --
worth it on a GPU box or if accuracy matters more than update latency.
Either way, this never blocks the video: the OCR worker runs in its own
process on the same "always the latest frame, never a backlog" pattern as
every other detector here (`app/processing/process_detector.py`), so the
live stream stays smooth regardless of how long a text detection pass
takes -- the OCR overlay/table just updates less often.

**Windows note (oneDNN):** `paddlepaddle`'s oneDNN/MKL-DNN CPU kernels can't
run these models on Windows -- every `predict()` raises `(Unimplemented)
ConvertPirAttribute2RuntimeAttribute not support` from
`onednn_instruction.cc`. Because the per-region calls in `ocr_detector.py`
catch-and-log so one bad crop can't kill a live stream, that failure is
silent at the pipeline level: OCR mode reports **zero text on every frame,
forever**. `OCRDetector` therefore defaults oneDNN **off on Windows** (on
everywhere else) and runs a one-off self-test at construction so a broken
backend fails loudly at startup instead. Override with `OCR_ENABLE_MKLDNN`.
Cost of running without it, measured on this Windows box: ~15%.

**Performance on Windows CPU** (Ryzen-class, no GPU for paddle): a full-frame
mobile-model pass at `OCR_DET_LIMIT_SIDE_LEN=1920` takes ~4s, or ~7s with the
carrier pass on. Dropping to 1280 roughly halves that (~2.1s) and still reads
wall-sign and card-sized text in testing -- a good trade if you want the
table to refresh more often.

**Config:** `OcrConfig` in `app/config.py`, all `OCR_*` env vars (commented
with defaults in `.env`) -- detector thresholds, model choice, upscale
limits, carrier-pass tuning, region caps.

**API:** `GET /api/cameras/{camera_id}/ocr` -- voted reading per tracked
text region: `{track_id, text, confidence, votes, confirmed, moving, bbox,
first_seen_str, last_seen_str}`.

## 10. AdaFace/SCRFD/ByteTrack/FAISS attendance system

A separate, newer system living in `app/attendance/` and
`app/web/attendance_server.py` -- **not** the pipeline described in sections
1-9 above, and not compatible with it. Built for the group-entrance
attendance use case: detect every face in frame, track each person as one
identity while they walk through, pick their clearest frame, recognize them
once, mark attendance.

Stack: insightface SCRFD (buffalo_l pack, detection only) for detection,
`sv.ByteTrack` (supervision package) for tracking, AdaFace IR-101/WebFace12M
(exported to ONNX, `scripts/export_adaface_onnx.py`) for recognition, FAISS
`IndexFlatIP` for matching, FastAPI + SQLite for the backend, React for the
frontend.

### Run

```bash
source .venv/bin/activate   # also sets LD_LIBRARY_PATH for onnxruntime-gpu, see below
python -m uvicorn app.web.attendance_server:app --host 0.0.0.0 --port 8001
```

```bash
cd frontend && npm run dev   # dev server proxies /api, /data, /ws to :8001
```

The recognition worker starts automatically at server startup, reading the
camera at `ATTENDANCE_CAMERA_INDEX` (default 2) from the `.env` `CAMERA_*`
config described in section 4. Port 8001, not 8000, because the legacy
`app/web/server.py` (sections 1-9) spins up a detector subprocess per camera
on its own startup and the two shouldn't fight over GPU memory during
development -- run only one at a time, or move one to a different machine
for production.

### API

```
POST   /api/users                    register with images (multipart: name, employee_id, images[])
GET    /api/users                    list enrolled users + stored face thumbnails
DELETE /api/users/{id}                remove a user and all their embeddings
POST   /api/users/{id}/faces          add another angle
DELETE /api/faces/{id}                remove a single stored angle
GET    /api/attendance?date=          attendance log (YYYY-MM-DD, omit for all)
WS     /ws/events?token=...           live events: {type: "frame"|"attendance"|"unknown", ...}
```

Same demo bearer token as the rest of the app (`app/web/auth.py`).

### Scripts

- `scripts/export_adaface_onnx.py` -- one-time checkpoint -> ONNX conversion,
  gated by a (32, 3, 112, 112) -> (32, 512) shape + CUDA-provider check.
- `scripts/calibrate_threshold.py` -- feed it known/unknown face crops,
  it prints both cosine-similarity distributions and recommends
  `MATCH_THRESHOLD`. Run this before trusting any threshold in production.
- `scripts/check_scrfd_detection.py`, `scripts/live_tracking_check.py` --
  the step-by-step gate scripts used to validate detection/tracking against
  the real camera before any app code was written.

### Every place a wrong assumption here would silently break accuracy

No exceptions are thrown for any of these -- the system keeps running,
keeps producing scores and matches, they're just wrong. Check this list
first if recognition quality degrades without an obvious error in the logs.

1. **Alignment mismatch between enrollment and live recognition.**
   `app/attendance/alignment.py` is the *only* place the 5-landmark
   similarity transform and BGR normalization are implemented, and both
   `enrollment.py` and `worker.py` import it from there. If anyone ever adds
   a second implementation (e.g. a "quick fix" inline warp somewhere), even
   a slightly different reference-point set or normalization formula will
   silently shift every embedding into a different region of AdaFace's
   embedding space. Same person, same camera -- cosine similarity quietly
   drops and nobody gets matched. No error, no crash, just a system that
   stops recognizing people. Verified the reference landmarks here against
   AdaFace's own training-time alignment code (see the module docstring)
   before writing anything against them.

2. **A threshold copied from ArcFace/SFace/another project.** AdaFace's
   cosine-similarity distribution is not the same as ArcFace's -- a
   threshold tuned for one is not a safe assumption for the other, even
   though both are "face embeddings normalized to [-1, 1]." `MATCH_THRESHOLD`
   is an env var, not a constant, specifically so nobody is tempted to
   hardcode a number that "feels about right." Always run
   `scripts/calibrate_threshold.py` against crops from the real camera.

3. **Stale FAISS index after an enrollment change.** The index is an
   in-process singleton (`app/attendance/pipeline.py`) rebuilt from SQLite
   on every add/delete (`FaceIndex.rebuild()`, called at the end of every
   `enrollment.py` mutation). If a future change adds a way to write to
   `face_embeddings` without going through `enrollment.py`'s functions --
   a direct DB script, a bulk-import tool, a migration -- and forgets to
   call `rebuild()` afterward, the index keeps serving searches against the
   old embedding set. New enrollments won't be recognized; deleted users
   still will be. No error -- it will just keep returning confident-looking
   wrong answers.

4. **Face pixel width too small.** The CP Plus camera (4MP, fixed 3.6mm
   lens) makes faces beyond ~6m under 40px and effectively unrecognizable
   -- this is optics, not a bug fixable in software. Every recognition
   attempt logs the detected face's pixel width
   (`app/attendance/worker.py: _finalize_track`); anything under
   `LOW_CONFIDENCE_FACE_PX` (default 60) is flagged `low_confidence: true`
   in both the DB row and the WebSocket event, specifically so a string of
   low-confidence flags is diagnosable as a placement/distance problem
   instead of looking like "the model doesn't work." Registration enforces
   a stricter floor (`MIN_REGISTER_FACE_PX`, default 100px) since a bad
   enrollment photo poisons every future match against that person.
   Confirmed empirically during setup: camera 1 (a different, non-CP-Plus
   camera on this network) tops out around 27-48px even for someone sitting
   close, and is not usable for this system at its current mounting;
   camera 2 (the actual CP Plus unit, near the entrance) measured 92-152px
   for someone walking through -- see `ATTENDANCE_CAMERA_INDEX`.

5. **Old ArcFace/facenet enrollment data treated as reusable.** AdaFace
   embeddings and ArcFace/facenet embeddings are different vector spaces --
   there is no meaningful cosine similarity between them. `app/attendance/db.py`
   is a fresh schema, not a migration of anything in the legacy system's
   storage. If any pre-existing enrollment data is ever found, it must be
   treated as unusable and everyone re-enrolled, never imported directly.

6. **Recognizing the same track twice.** `worker.py` maintains
   `finalized_ids` and checks it before ever finalizing a track -- but this
   set lives only in the worker's process memory. A server restart mid-day
   resets it, so anyone whose track ID (from before the restart) is still
   technically "active" in a resumed tracker state could theoretically be
   re-processed. In practice a restart also restarts the tracker from
   scratch, so this is a non-issue for tracks entirely before/after a
   restart -- flagged here only because it's the kind of assumption ("track
   IDs are globally unique forever") that's obviously false the moment you
   say it out loud, and the failure mode (a duplicate attendance row) is
   silent, not a crash.

7. **`sv.ByteTrack` is deprecated** (removed in supervision 0.31.0; this
   project pins 0.30.3). The spec calls for `sv.ByteTrack` by name, so
   that's what's used, but it will need migrating to the `trackers` package's
   `ByteTrackTracker` (`update()` instead of `update_with_detections()`)
   before upgrading supervision past 0.30.x.
# Face_Testing_Website_With_InsightFace
#   F a c e _ T e s t i n g _ W e b s i t e _ W i t h _ I n s i g h t F a c e  
 # DeskWatch-AI
