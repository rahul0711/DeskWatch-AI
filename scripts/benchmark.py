"""Step 6: benchmark detection throughput and CPU usage against the live camera.

Runs the real pipeline (capture + SCRFD detection, no display) for a fixed
duration and reports process/capture FPS and CPU usage, so DETECTION_SIZE
and DETECTION_INTERVAL can be tuned with real numbers instead of guesses.

Usage:
    python scripts/benchmark.py --seconds 30
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from app.camera.rtsp_stream import RTSPStream
from app.config import load_config
from app.processing.pipeline import FacePipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    config = load_config()
    stream = RTSPStream(
        rtsp_url=config.camera.rtsp_url,
        transport=config.camera.transport,
        rtsp_url_masked=config.camera.rtsp_url_masked,
    ).start()

    pipeline = FacePipeline(
        stream=stream,
        detector_kwargs=dict(
            backend=config.detection.backend,
            onnx_provider=config.detection.onnx_provider,
            det_size=config.detection.det_size,
            conf_threshold=config.detection.conf_threshold,
        ),
        detection_interval=config.detection.interval,
        max_process_fps=config.max_process_fps,
    )

    proc = psutil.Process() if _HAS_PSUTIL else None
    if proc:
        proc.cpu_percent()  # prime the measurement

    print(f"Benchmarking for {args.seconds:.0f}s "
          f"(det_size={config.detection.det_size}, interval={config.detection.interval})...")
    detections = 0
    frames = 0
    start = time.time()
    while time.time() - start < args.seconds:
        result = pipeline.poll()
        if result is None:
            time.sleep(0.005)
            continue
        frames += 1
        if result.ran_detection:
            detections += 1

    elapsed = time.time() - start
    stream.stop()
    pipeline.close()

    print("\n=== Results ===")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Frames processed: {frames} ({frames / elapsed:.1f} fps)")
    print(f"Detection passes: {detections} ({detections / elapsed:.1f} fps)")
    print(f"Final capture FPS (camera side): {result.capture_fps:.1f}" if result else "")
    if proc:
        print(f"CPU usage (this process, avg over run): {proc.cpu_percent():.1f}%")
        print(f"Memory (RSS): {proc.memory_info().rss / 1e6:.0f} MB")
    else:
        print("(install psutil for CPU/memory stats: pip install psutil)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
