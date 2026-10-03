# =============================================================================
# CCTV Face AI - Camera & Pipeline Configuration
# Copy this file to `.env` and fill in your real values. Never commit `.env`.
# =============================================================================

# --- Camera identity (cosmetic, shown in the HUD) ---------------------------
CAMERA_NAME=Camera 01

# --- Connection ---------------------------------------------------------------
# Legacy single-camera vars -- unused now that CAMERA_1_*/CAMERA_2_* below are
# set (numbered blocks take priority, see app/config.py:_load_camera_configs).
CAMERA_IP=192.168.1.64
CAMERA_RTSP_PORT=554
CAMERA_USERNAME=admin
CAMERA_PASSWORD=Khanishk
# 101 = channel 1, main stream (high res, high CPU cost for detection)
# 102 = channel 1, sub stream  (low res, recommended for face detection)
CAMERA_RTSP_PATH=/Streaming/Channels/102

# Option B: if `scripts/discover_camera.py` (or ONVIF) reveals a different
# path/scheme, or you just want full control, set the complete URL here.
# When set, this OVERRIDES all the CAMERA_* parts above.
# CAMERA_RTSP_URL=rtsp://admin:changeme@192.168.1.64:554/Streaming/Channels/102

# rtsp transport: tcp is more reliable over Wi-Fi/WAN, udp is lower latency on
# a clean local network.
CAMERA_RTSP_TRANSPORT=tcp

# --- ONVIF (used only by scripts/discover_camera.py) -------------------------
ONVIF_PORT=80

# --- Multiple cameras ---------------------------------------------------------
# Both cameras point at the same device/API (172.29.7.7, admin) for now --
# swap CAMERA_2_IP (and its RTSP_PATH/channel) once a second physical camera
# is available.
CAMERA_1_NAME=Camera 01
CAMERA_1_IP=172.29.7.7
CAMERA_1_RTSP_PORT=554
CAMERA_1_USERNAME=admin
CAMERA_1_PASSWORD=Kanishk@2216
CAMERA_1_RTSP_PATH=/Streaming/Channels/101
CAMERA_1_RTSP_TRANSPORT=tcp

CAMERA_2_NAME=Inside
CAMERA_2_IP=172.29.7.8
CAMERA_2_RTSP_PORT=554
CAMERA_2_USERNAME=admin
CAMERA_2_PASSWORD=India@4321
# Tried subtype=1 (sub-stream, 1280x720) to fight the ~700ms/keyframe
# stall: it halves how often the stall happens (every ~2s instead of every
# ~1s), but tested it with a real walk-through and face width at the door
# dropped to ~60px best-case (was 92-152px on subtype=0/1080p) -- right at
# the low-confidence cutoff for the *closest* position, worse everywhere
# else. Not worth it: the dashboard's broadcast-buffer smoothing (see
# app/attendance/worker.py) already fully absorbs subtype=0's stall
# pattern, so switching streams traded away real recognition accuracy for
# a problem that was already solved. Stay on subtype=0 (main/high-res
# stream, channel 1) unless the camera's own keyframe interval setting
# changes (a camera-admin-page fix, not a software one).
CAMERA_2_RTSP_PATH=/video/live?channel=1&subtype=1
CAMERA_2_RTSP_TRANSPORT=tcp
#
# (Each numbered block also accepts its own CAMERA_N_RTSP_URL as a full
# override, same as the single-camera CAMERA_RTSP_URL above.)

# --- Face detection ------------------------------------------------------------
# yolo (default, fast, no landmarks) | insightface/scrfd (has 5-point landmarks,
# needed for the frontal-face filter in app/web/server.py to reject side-profile
# captures) | mtcnn.
DETECTOR_BACKEND=insightface
# Which per-camera detector worker subprocesses to start: any of
# face,person,attendance,activity,ocr (comma-separated). Unset/empty =
# all 5 (previous default). Only "attendance" is actually used -- it's
# the only mode that calls /Recognize, and it already records head-count
# too (see server.py:296-297), so the standalone "face" worker was pure
# overhead. Switching a camera to face/person/activity/ocr via the UI
# still won't error, it'll just report nothing (no worker to read).
ENABLED_DETECTORS=attendance
# ONNX Runtime execution provider. Use CPUExecutionProvider unless you have
# installed onnxruntime-gpu + CUDA/cuDNN, in which case CUDAExecutionProvider.
ONNX_PROVIDER=CUDAExecutionProvider
# SCRFD detector input size (square). Smaller = faster, less accurate on small/
# far faces. Raised to 1280 (from 320) to feed the detector far more actual
# resolution off the 1920x1080/2688x1520 source frames -- audit found the
# full frame was already being passed in uncropped, just downscaled hard
# internally by insightface before detection ran.
DETECTION_SIZE=1280
DETECTION_CONF_THRESHOLD=0.5
# Run the detector once every N frames; frames in between reuse the last
# known boxes via lightweight IoU tracking. 1 = detect every frame.
DETECTION_INTERVAL=3

# Web app "head count per minute" table / detection mode:
# face | person | attendance | activity | ocr
# Face-only misses anyone whose face isn't visible to the camera (looking
# down, turned away) -- person detection finds the body regardless of head
# angle. Switchable at runtime per-camera from the web UI's mode buttons;
# this is just the mode a camera starts in.
HEAD_COUNT_SOURCE=attendance
PERSON_CONF_THRESHOLD=0.4

# --- Face crop saving (optional, for testing only) ---------------------------
SAVE_FACE_CROPS=false
SAVE_CROPS_DIR=data/samples
SAVE_CROPS_MAX=20

# --- OCR mode (HEAD_COUNT_SOURCE=ocr) ----------------------------------------
# See app/detection/ocr_detector.py's module docstring for the full pipeline.
# Defaults below match app/config.py's OcrConfig -- only uncomment to override.
# OCR_DET_LIMIT_SIDE_LEN=1920      # cap on the side PP-OCR's detector resizes to (native 1080p = no downscale)
# OCR_DET_THRESH=0.3               # DBNet pixel-level text/no-text threshold
# OCR_DET_BOX_THRESH=0.5           # per-box confidence floor to keep a detected region
# OCR_DET_UNCLIP_RATIO=1.6         # how far to expand DBNet's shrunk box back out
# OCR_DET_MODEL=PP-OCRv5_mobile_det   # default: ~1.3s/frame @1080p on this CPU-only Mac. Set to PP-OCRv6_medium_det
#                                      # (or *_server_det on a GPU box) for higher accuracy at ~7-9s/frame instead.
# OCR_REC_MODEL=PP-OCRv5_mobile_rec   # pair with OCR_DET_MODEL's tier -- PP-OCRv6_medium_rec / *_server_rec for the accurate option.
# OCR_REC_MIN_CONFIDENCE=0.35      # recognizer confidence floor to keep a reading at all
# OCR_MIN_CROP_HEIGHT_PX=10        # below this, a region is too small to ever OCR reliably -- skipped
# OCR_TARGET_REC_HEIGHT=48         # upscale a corrected crop up to this tall before recognition
# OCR_MAX_UPSCALE=6.0              # cap on how much a tiny crop gets blown up
# OCR_MAX_REGIONS_PER_FRAME=40     # hard cap on OCR regions processed per frame (perf safety valve)
# OCR_CARRIER_CONF_THRESHOLD=0.35  # YOLO+BoT-SORT confidence floor for a moving-text "carrier" (person/phone/book/laptop)
# OCR_MAX_CARRIERS_PER_FRAME=6     # cap on how many tracked carriers get a dedicated re-detect pass
# OCR_CARRIER_CROP_MAX_SIDE=800    # cap a carrier crop's longest side before upscaling (perf safety valve)
# OCR_CARRIER_UPSCALE=2.0          # zoom factor applied to each carrier crop before re-running text detection
# OCR_ENABLE_CARRIER_PASS=true     # false = static full-frame OCR only, no moving-text tracking
# OCR_ENABLE_MKLDNN=               # unset = auto (off on Windows, on elsewhere). PaddlePaddle's oneDNN
#                                  # kernels crash these models on Windows ("ConvertPirAttribute2Runtime-
#                                  # Attribute not support"), which makes OCR silently read nothing --
#                                  # only set this to true on Windows if a future paddle release fixes it.

# --- Performance / display -----------------------------------------------------
# Phase 2.1: display and detection are now independently rate-limited --
# previously MAX_PROCESS_FPS throttled both, capping the displayed stream
# down to the AI sampling rate instead of the camera's native decode rate.
# MAX_DISPLAY_FPS caps how often a frame is returned for rendering;
# MAX_DETECTION_FPS caps how often a frame is submitted to the detector
# (this also subsumes DETECTION_INTERVAL's old role, now vestigial).
# MAX_PROCESS_FPS is kept as a legacy fallback: if it's set and the two
# below are NOT, it's used for both (old .env files keep working as before).
MAX_DISPLAY_FPS=25
MAX_DETECTION_FPS=10
# MAX_PROCESS_FPS=15

# --- Reconnect behaviour ---------------------------------------------------
RECONNECT_INITIAL_DELAY=1.0
RECONNECT_MAX_DELAY=30.0

# =============================================================================
# New AdaFace/SCRFD/ByteTrack/FAISS attendance system (app/attendance/,
# app/web/attendance_server.py). All optional -- sane defaults in
# app/attendance/config.py if unset.
# =============================================================================

# Which configured camera (1-indexed, matches CAMERA_<n>_* above) the live
# recognition worker reads from. Confirmed via live testing that camera 2
# ("Inside", the CP Plus unit) gives usable face widths (92-152px) near the
# door; camera 1 does not (27-48px, below the low-confidence cutoff).
ATTENDANCE_CAMERA_INDEX=2

# Cosine similarity cutoff for a FAISS match to count as a recognition.
# DO NOT hand-tune this by guessing -- run scripts/calibrate_threshold.py
# against real known/unknown face crops from this camera and use its
# recommendation. This default is a starting point, not a validated value.
MATCH_THRESHOLD=0.35

# Registration: reject an enrollment photo if the detected face is smaller
# than this many pixels wide.
MIN_REGISTER_FACE_PX=100
# Live recognition: attempt still happens below this width, but the result
# is flagged low_confidence=true so you can diagnose camera placement.
LOW_CONFIDENCE_FACE_PX=60

# SCRFD detector input size (square). 640 balances small-face recall against
# throughput on the RTX 3060; raise if you need to detect smaller/farther
# faces and have GPU headroom to spare.
SCRFD_DET_SIZE=640

# How often the recognition worker runs detection (throttles CPU/GPU load
# independent of the camera's native frame rate).
ATTENDANCE_DETECTION_FPS=10
# How often an annotated frame is pushed to the dashboard over /ws/events.
# Cheap to run higher than you'd think -- each frame is a small (~960px)
# JPEG -- and a higher rate reads as visibly smoother. Capped below the
# camera's native ~25fps on purpose: the backlog that smooths over its
# ~0.7-0.9s stall only builds because drain is slower than the camera's
# burst-fill rate right after each stall -- push this too close to native
# fps and that margin disappears, bringing the freeze back.
FRAME_BROADCAST_FPS=20
# Max age of a frame the live view will still show, and how much backlog
# it banks to smooth over the camera's own ~0.7-0.9s keyframe stall (see
# README). Sized just above that stall duration.
BROADCAST_BUFFER_SECONDS=1.2
# Force-finalize (embed + match + write attendance) a track that has
# lingered this long without being lost, so someone who stands still near
# the camera still eventually gets marked present.
MAX_TRACK_AGE_S=6.0

# Cooldown window in seconds after a person is recognized and punched.
# During this window (default 600s = 10 minutes), subsequent sightings of this
# person will NOT save new crop images to disk or insert duplicate DB rows.
# The dashboard announces/displays: "You're done punching for like X minutes".
PUNCH_COOLDOWN_SECONDS=600

# Retention window in seconds for storing unknown face crops on disk and in UI.
# Unrecognized face crops are removed after 30 seconds to prevent disk buildup.
UNKNOWN_FACE_TTL_SECONDS=30

