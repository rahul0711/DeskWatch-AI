"""Step 1 investigation tool: figure out how THIS camera actually exposes RTSP.

Usage:
    python scripts/discover_camera.py --ip 192.168.1.64 --user admin --password changeme

    # NVR / multi-channel device: same IP + login, different channel number
    python scripts/discover_camera.py --ip 192.168.1.64 --user admin --password changeme --channel 2

What it does, in order:
  1. TCP reachability check on the RTSP port (554) and common ONVIF/HTTP ports.
  2. If `onvif-zeep` is installed: connects via ONVIF and asks the camera's
     Media service for its REAL stream URIs (the authoritative source of
     truth — no guessing). This also confirms ONVIF support/port. For an
     NVR this typically returns ALL of its channels' URIs in one call, so
     it's the fastest way to find channel 2+ even without --channel.
  3. Regardless of ONVIF success, brute-force probes a list of known-vendor
     RTSP path conventions for the requested --channel (Hikvision-style,
     since Prama is a Hikvision-partnered/OEM brand in India; Dahua-style
     as a secondary fallback) by actually opening each URL with OpenCV and
     reading a frame. Reports resolution/fps for every URL that works.

Nothing here is written back into the app automatically — copy whichever
working URL/path this prints into your `.env` (CAMERA_RTSP_URL or
CAMERA_RTSP_PATH / CAMERA_N_RTSP_PATH).
"""
from __future__ import annotations

import argparse
import re
import socket
import sys
import time
from urllib.parse import quote

import cv2


def candidate_paths(channel: int) -> list[str]:
    """RTSP path conventions for the given 1-indexed channel number.

    Hikvision-style numbering is <channel><stream>, e.g. channel 1 main =
    101, channel 1 sub = 102, channel 2 main = 201, channel 2 sub = 202 --
    this is how a single NVR/admin account exposes multiple camera channels
    under one IP.
    """
    return [
        "/stream1",                           # TP-Link Tapo main stream (HD)
        "/stream2",                           # TP-Link Tapo sub stream (SD)
        f"/Streaming/Channels/{channel}01",   # Hikvision-style main stream
        f"/Streaming/Channels/{channel}02",   # Hikvision-style sub stream
        f"/Streaming/Channels/{channel}",
        f"/h264/ch{channel}/main/av_stream",  # some Hikvision OEM firmwares
        f"/h264/ch{channel}/sub/av_stream",
        f"/cam/realmonitor?channel={channel}&subtype=0",  # Dahua-style main
        f"/cam/realmonitor?channel={channel}&subtype=1",  # Dahua-style sub
        "/live" if channel == 1 else f"/live{channel}",
        f"/live/ch{channel - 1}",
        f"/onvif{channel}",
        f"/media/video{channel}",
    ]


def check_port(ip: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def try_onvif(ip: str, port: int, user: str, password: str) -> list[str]:
    """Ask the camera's ONVIF Media service for real stream URIs. Best effort."""
    found: list[str] = []
    try:
        from onvif import ONVIFCamera  # provided by onvif-zeep
    except ImportError:
        print("[ONVIF] onvif-zeep not installed, skipping ONVIF query "
              "(pip install onvif-zeep to enable). Falling back to brute force only.")
        return found

    try:
        cam = ONVIFCamera(ip, port, user, password)
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        for profile in profiles:
            req = media.create_type("GetStreamUri")
            req.ProfileToken = profile.token
            req.StreamSetup = {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            }
            uri_info = media.GetStreamUri(req)
            print(f"[ONVIF] Profile '{profile.Name}' -> {uri_info.Uri}")
            found.append(uri_info.Uri)
    except Exception as exc:  # noqa: BLE001 - report and continue, don't crash the tool
        print(f"[ONVIF] Query failed: {exc}")
    return found


def probe_rtsp_url(url: str, timeout_frames: int = 30) -> dict | None:
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return {"width": w, "height": h, "fps": fps}


def mask(url: str) -> str:
    """Hide the password in a printed RTSP URL, regardless of quoting."""
    return re.sub(r"(rtsp://[^:@/]+):[^@/]+@", r"\1:***@", url)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", required=True, help="Camera IP address")
    parser.add_argument("--rtsp-port", type=int, default=554)
    parser.add_argument("--onvif-port", type=int, default=80)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument(
        "--channel", type=int, default=1,
        help="1-indexed camera channel to probe. Use 2, 3, ... for a second "
             "camera on the same NVR/admin account (same --ip, same "
             "--user/--password, different channel). Default: 1.",
    )
    args = parser.parse_args()

    print(f"=== Step 1: TCP reachability ===")
    for label, port in [("RTSP", args.rtsp_port), ("ONVIF/HTTP", args.onvif_port), ("HTTP-alt", 8000)]:
        reachable = check_port(args.ip, port)
        print(f"  {label} port {port}: {'OPEN' if reachable else 'closed/unreachable'}")

    print(f"\n=== Step 2: ONVIF probe (authoritative, if available) ===")
    print("(For an NVR/multi-channel device this usually lists every channel's")
    print(" URI at once -- check whether channel 2 is already in this output")
    print(" before relying on the --channel brute force in Step 3.)")
    onvif_uris = try_onvif(args.ip, args.onvif_port, args.user, args.password)

    print(f"\n=== Step 3: Brute-force RTSP path probing (channel {args.channel}) ===")
    working: list[tuple[str, dict]] = []

    # URL-quote credentials: an '@' or ':' inside the password would
    # otherwise be misread as the userinfo/host separator by the RTSP parser.
    user_q = quote(args.user, safe="")
    pwd_q = quote(args.password, safe="")

    urls_to_try: list[str] = list(onvif_uris)
    for path in candidate_paths(args.channel):
        urls_to_try.append(f"rtsp://{user_q}:{pwd_q}@{args.ip}:{args.rtsp_port}{path}")

    for url in urls_to_try:
        masked = mask(url)
        print(f"  Trying {masked} ...", end=" ", flush=True)
        start = time.time()
        info = probe_rtsp_url(url)
        elapsed = time.time() - start
        if info:
            print(f"OK  ({info['width']}x{info['height']} @ {info['fps']:.1f}fps, {elapsed:.1f}s)")
            working.append((url, info))
        else:
            print(f"failed ({elapsed:.1f}s)")

    print("\n=== Result ===")
    if not working:
        print("No candidate RTSP URL worked. Check credentials, that RTSP is enabled")
        print("on the camera's Network settings, and that this machine can reach the")
        print("camera (same subnet / VLAN / firewall rules). See README troubleshooting.")
        return 1

    best_url, best_info = working[0]
    print(f"Found {len(working)} working URL(s). Recommended (first that worked):")
    print(f"  {mask(best_url)}")
    print(f"  -> {best_info['width']}x{best_info['height']} @ {best_info['fps']:.1f} fps")
    env_var = "CAMERA_RTSP_PATH" if args.channel == 1 else f"CAMERA_{args.channel}_RTSP_PATH"
    print(f"\nCopy the path portion of this into your .env as {env_var},")
    print("or set the matching *_RTSP_URL to the full URL directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
