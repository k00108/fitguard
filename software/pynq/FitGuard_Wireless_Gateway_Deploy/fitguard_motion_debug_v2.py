"""Visual-only acceptance test for the FitGuard motion tracker V2."""

import json
import gc
import threading
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np


def _run_motion_debug_v2_impl(
    bitstream="fitguard_unified.bit",
    seconds=60,
    preview_fps=8,
    preview_width=1400,
    output_dir="motion_debug_v2_results",
):
    from IPython.display import Image, display
    from fitguard_camera_runtime import FitGuardCamera
    from motion_tracker_v2 import MotionPersonTracker

    try:
        import ipywidgets as widgets
        widget = widgets.Image(
            format="jpeg",
            layout=widgets.Layout(width=f"{int(preview_width)}px", max_width="100%"),
        )
        display(widget)
    except Exception:
        widget = None
    display_handle = None
    tracker = MotionPersonTracker()
    rows = []
    next_preview = 0.0
    last_preview = None
    started = time.monotonic()

    with FitGuardCamera(bitstream=bitstream) as camera:
        deadline = started + float(seconds)
        while time.monotonic() < deadline:
            timestamp, raw = camera.capture()
            pl_mask, dma_ms = camera.run_pl(raw)
            feature, motion_mask, annotated = tracker.update(raw, pl_mask, timestamp)
            row = asdict(feature)
            row["dma_ms"] = float(dma_ms)
            rows.append(row)

            now = time.monotonic()
            if now >= next_preview:
                raw_bgr = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
                pl_bgr = cv2.cvtColor(pl_mask, cv2.COLOR_GRAY2BGR)
                motion_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(raw_bgr, "Camera", (6, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                cv2.putText(pl_bgr, "PL mask (kept)", (6, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                motion_overlay = cv2.addWeighted(annotated, 0.78, motion_bgr, 0.22, 0)
                cv2.putText(motion_overlay, "Motion-aware target V2", (6, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
                last_preview = np.hstack((raw_bgr, pl_bgr, motion_overlay))
                ok, jpeg = cv2.imencode(".jpg", last_preview, [cv2.IMWRITE_JPEG_QUALITY, 84])
                if ok:
                    if widget is not None:
                        widget.value = jpeg.tobytes()
                    else:
                        image = Image(data=jpeg.tobytes(), format="jpeg")
                        if display_handle is None:
                            display_handle = display(image, display_id=True)
                        else:
                            display_handle.update(image)
                next_preview = now + 1.0 / max(float(preview_fps), 1.0)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if last_preview is not None:
        cv2.imwrite(str(output / "latest_motion_debug.jpg"), last_preview)
    locked = sum(row["valid"] for row in rows)
    summary = {
        "frames": len(rows),
        "elapsed_s": time.monotonic() - started,
        "lock_rate": locked / len(rows) if rows else 0.0,
        "mean_quality": float(np.mean([row["quality"] for row in rows])) if rows else 0.0,
        "sources": {name: sum(row["source"] == name for row in rows) for name in sorted({row["source"] for row in rows})},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


_RUN_LOCK = threading.Lock()
_LAST_FINISHED = -1e9


def run_motion_debug_v2(*args, **kwargs):
    """Notebook-safe wrapper that ignores accidental queued double-runs."""
    global _LAST_FINISHED
    if not _RUN_LOCK.acquire(blocking=False):
        print("A motion debug run is already active; duplicate request ignored.")
        return {"duplicate_ignored": True, "reason": "already_running"}
    try:
        if time.monotonic() - _LAST_FINISHED < 4.0:
            print("Duplicate queued run ignored. Wait four seconds before a deliberate rerun.")
            return {"duplicate_ignored": True, "reason": "queued_rerun"}
        return _run_motion_debug_v2_impl(*args, **kwargs)
    finally:
        _LAST_FINISHED = time.monotonic()
        _RUN_LOCK.release()
        gc.collect()
