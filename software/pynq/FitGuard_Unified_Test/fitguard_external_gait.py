"""External full-body camera tracking on the final PL binary mask.

The PL remains responsible for thresholding and morphology.  This module only
selects and tracks the most plausible human silhouette, preserves the existing
64x7 visual feature contract, and computes explainable display indicators.
"""

from collections import deque
from dataclasses import asdict, dataclass
import csv
import json
from math import exp, log
from pathlib import Path
import time

import cv2
import numpy as np

from opencv_backend import Target, extract_targets


WIDTH = 320
HEIGHT = 240


@dataclass(frozen=True)
class ExternalVisionRecord:
    timestamp: float
    valid: int
    cx: float
    cy: float
    area: float
    width: float
    height: float
    vx: float
    vy: float
    tracking_quality: float


class ExternalGaitTracker:
    """Track one full-body silhouette without thresholding or morphology in PS."""

    def __init__(self, min_area=800.0, history_seconds=8.0):
        self.min_area = float(min_area)
        self.history_seconds = float(history_seconds)
        self.previous = None
        self.history = deque(maxlen=512)

    def _score(self, target):
        x, y, width, height = target.bbox
        cx, cy = target.center
        height_ratio = height / HEIGHT
        area_ratio = target.area / (WIDTH * HEIGHT)
        aspect = height / max(float(width), 1.0)

        if height_ratio < 0.18 or area_ratio > 0.65 or width / WIDTH > 0.85:
            return -1.0

        height_score = min(height_ratio / 0.65, 1.0)
        area_score = min(area_ratio / 0.20, 1.0)
        aspect_score = exp(-abs(log(max(aspect, 1e-3) / 2.0)) / 0.8)
        continuity_score = 0.5
        if self.previous is not None and self.previous.valid:
            dx = cx / WIDTH - self.previous.cx
            dy = cy / HEIGHT - self.previous.cy
            continuity_score = exp(-np.hypot(dx, dy) / 0.20)

        border_contacts = sum((x <= 0, y <= 0, x + width >= WIDTH, y + height >= HEIGHT))
        border_factor = 0.82 ** border_contacts
        return border_factor * (
            0.36 * height_score
            + 0.28 * area_score
            + 0.16 * aspect_score
            + 0.20 * continuity_score
        )

    def select(self, targets):
        scored = [(self._score(target), target) for target in targets]
        scored = [item for item in scored if item[0] >= 0.0]
        return max(scored, key=lambda item: item[0]) if scored else (0.0, None)

    def update(self, binary, timestamp=None):
        if binary.shape != (HEIGHT, WIDTH) or binary.dtype != np.uint8:
            raise ValueError("binary must be a 240x320 uint8 PL mask")
        values = np.unique(binary)
        if not np.all(np.isin(values, (0, 255))):
            raise ValueError("external tracker accepts the binary PL output only")

        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        targets = extract_targets(binary, min_area=self.min_area, max_area_ratio=0.65)
        quality, selected = self.select(targets)

        if selected is None:
            record = ExternalVisionRecord(timestamp, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            cx, cy = selected.center
            _, _, width, height = selected.bbox
            nx, ny = cx / WIDTH, cy / HEIGHT
            vx = vy = 0.0
            if self.previous is not None and self.previous.valid:
                dt = timestamp - self.previous.timestamp
                if dt > 0:
                    vx = (nx - self.previous.cx) / dt
                    vy = (ny - self.previous.cy) / dt
            record = ExternalVisionRecord(
                timestamp=timestamp,
                valid=1,
                cx=nx,
                cy=ny,
                area=selected.area / (WIDTH * HEIGHT),
                width=width / WIDTH,
                height=height / HEIGHT,
                vx=vx,
                vy=vy,
                tracking_quality=float(quality),
            )

        self.previous = record
        self.history.append(record)
        while self.history and timestamp - self.history[0].timestamp > self.history_seconds:
            self.history.popleft()
        return record, selected, targets, self.indicators()

    def indicators(self):
        rows = [row for row in self.history if row.valid]
        if len(rows) < 8:
            return {
                "tracked_seconds": 0.0,
                "forward_speed_body_widths_s": None,
                "vertical_bob_percent_height": None,
                "silhouette_width_cv_percent": None,
                "estimated_cadence_steps_min": None,
            }

        t = np.asarray([row.timestamp for row in rows], dtype=np.float64)
        t -= t[0]
        cx = np.asarray([row.cx for row in rows])
        cy = np.asarray([row.cy for row in rows])
        width = np.asarray([row.width for row in rows])
        height = np.asarray([row.height for row in rows])
        median_width = max(float(np.median(width)), 1e-4)
        median_height = max(float(np.median(height)), 1e-4)
        slope_x = float(np.polyfit(t, cx, 1)[0]) if t[-1] > 0 else 0.0
        cy_residual = cy - np.polyval(np.polyfit(t, cy, 1), t)
        width_cv = float(np.std(width) / max(float(np.mean(width)), 1e-4) * 100.0)

        cadence = None
        if t[-1] >= 3.0:
            rate = 20.0
            sample_t = np.arange(0.0, t[-1], 1.0 / rate)
            signal = np.interp(sample_t, t, cy_residual)
            signal += 0.35 * (
                np.interp(sample_t, t, width - np.mean(width))
                / max(float(np.std(width)), 1e-4)
            ) * max(float(np.std(signal)), 1e-4)
            signal -= np.mean(signal)
            frequencies = np.fft.rfftfreq(len(signal), 1.0 / rate)
            power = np.abs(np.fft.rfft(signal)) ** 2
            band = (frequencies >= 0.6) & (frequencies <= 3.0)
            if np.any(band) and float(power[band].max()) > 1e-10:
                cadence = float(frequencies[band][np.argmax(power[band])] * 60.0)

        return {
            "tracked_seconds": float(t[-1]),
            "forward_speed_body_widths_s": abs(slope_x) / median_width,
            "vertical_bob_percent_height": float(np.std(cy_residual) / median_height * 100.0),
            "silhouette_width_cv_percent": width_cv,
            "estimated_cadence_steps_min": cadence,
        }


def annotate_external(image_bgr, selected, candidates, record, indicators):
    result = image_bgr.copy()
    for target in candidates:
        color = (0, 255, 0) if target is selected else (90, 90, 90)
        cv2.drawContours(result, [target.contour], -1, color, 1)
    if selected is not None:
        x, y, width, height = selected.bbox
        cv2.rectangle(result, (x, y), (x + width - 1, y + height - 1), (255, 0, 0), 2)
        cv2.drawMarker(result, selected.center, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)

    lines = [
        f"person {'LOCKED' if record.valid else 'NOT FOUND'}  quality {record.tracking_quality:.2f}",
        "speed --" if indicators["forward_speed_body_widths_s"] is None else f"speed {indicators['forward_speed_body_widths_s']:.2f} body-width/s",
        "vertical bob --" if indicators["vertical_bob_percent_height"] is None else f"vertical bob {indicators['vertical_bob_percent_height']:.1f}% height",
        "cadence --" if indicators["estimated_cadence_steps_min"] is None else f"cadence {indicators['estimated_cadence_steps_min']:.0f} step/min",
    ]
    cv2.rectangle(result, (0, 0), (WIDTH, 78), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(result, line, (6, 17 + 19 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    return result


def feature_dict(record):
    """Return the existing 64x7-compatible visual feature row."""
    values = asdict(record)
    return {key: values[key] for key in ("timestamp", "valid", "cx", "cy", "area", "width", "height", "vx", "vy")}


def run_external_demo(
    bitstream="fitguard_unified.bit",
    seconds=30,
    min_area=800,
    output_dir="external_fullbody_results",
    preview_fps=8,
):
    """Run a separate, non-classifying external full-body validation demo.

    This deliberately does not feed the waist-camera model: the geometry is
    different and must be retrained with external-view data before accuracy is
    claimed.  The PL still performs every threshold/morphology operation.
    """
    from fitguard_camera_runtime import FitGuardCamera

    preview_widget = None
    if preview_fps > 0:
        try:
            import ipywidgets as widgets
            from IPython.display import display

            preview_widget = widgets.Image(
                format="jpeg", layout=widgets.Layout(width=f"{WIDTH * 2}px")
            )
            display(preview_widget)
        except Exception:
            preview_widget = None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tracker = ExternalGaitTracker(min_area=min_area)
    rows = []
    started = time.monotonic()
    deadline = started + float(seconds)
    next_preview = 0.0
    preview_interval = 1.0 / max(float(preview_fps), 1.0)
    last_mask = last_annotated = None

    with FitGuardCamera(bitstream=bitstream) as camera:
        while time.monotonic() < deadline:
            frame_started = time.monotonic()
            timestamp, raw = camera.capture()
            mask, dma_ms = camera.run_pl(raw)
            record, selected, candidates, indicators = tracker.update(mask, timestamp)
            bgr = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
            annotated = annotate_external(bgr, selected, candidates, record, indicators)
            row = asdict(record)
            row.update(indicators)
            row["dma_ms"] = dma_ms
            row["total_ms"] = (time.monotonic() - frame_started) * 1000.0
            rows.append(row)
            last_mask, last_annotated = mask, annotated

            now = time.monotonic()
            if preview_widget is not None and now >= next_preview:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                header = np.zeros((28, WIDTH * 2, 3), dtype=np.uint8)
                cv2.putText(header, "PL binary mask", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(header, "external full-body tracking", (WIDTH + 6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                preview = np.vstack((header, np.hstack((mask_bgr, annotated))))
                ok, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    preview_widget.value = jpeg.tobytes()
                next_preview = now + preview_interval

        capture_fps = camera.capture_fps()

    elapsed = time.monotonic() - started
    fields = list(rows[0].keys()) if rows else []
    with (output / "external_tracking.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if last_mask is not None:
        cv2.imwrite(str(output / "last_pl_mask.png"), last_mask)
        cv2.imwrite(str(output / "last_external_tracking.png"), last_annotated)

    valid = sum(int(row["valid"]) for row in rows)
    summary = {
        "mode": "external_fixed_camera_full_body",
        "status": "passed" if rows else "no_frames",
        "elapsed_seconds": elapsed,
        "processed_frames": len(rows),
        "processing_fps": len(rows) / elapsed if elapsed else 0.0,
        "camera_capture_fps": capture_fps,
        "valid_target_ratio": valid / len(rows) if rows else 0.0,
        "pl_operations": "grayscale + threshold + 3x3 open + 3x3 close",
        "ps_operations": "contour selection + tracking + explainable gait indicators",
        "classifier_used": False,
        "classifier_note": "Collect external-view labels and retrain before claiming classification accuracy.",
        "final_indicators": tracker.indicators(),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results saved in {output.resolve()}")
    return summary
