"""Final PS runtime. OpenCV does contours/features only; no threshold/morphology."""

import csv
import json
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from pynq import Overlay, allocate

from opencv_backend import annotate, extract_targets


WIDTH = 320
HEIGHT = 240
WORDS_PER_FRAME = WIDTH * HEIGHT // 2
RAW_BYTES = WIDTH * HEIGHT * 2
FEATURE_WINDOW_SAMPLES = 64
FEATURE_WINDOW_HZ = 25.0


@dataclass
class VisionFeature:
    timestamp: float
    valid: int
    cx: float
    cy: float
    area: float
    width: float
    height: float
    vx: float
    vy: float
    dma_ms: float
    feature_ms: float
    total_ms: float


class FitGuardCamera:
    def __init__(self, bitstream="fitguard_camera.bit", camera_index=0):
        bitstream = Path(bitstream)
        hwh = bitstream.with_suffix(".hwh")
        if not bitstream.exists() or not hwh.exists():
            raise FileNotFoundError(
                f"Final PL files are missing: {bitstream} and {hwh}"
            )

        self.overlay = Overlay(str(bitstream), download=True)
        self.dma = self.overlay.axi_dma_0
        self.input_buffer = allocate(shape=(WORDS_PER_FRAME,), dtype=np.uint32)
        self.output_buffer = allocate(shape=(WORDS_PER_FRAME,), dtype=np.uint32)
        self.binary_flat = np.empty(WIDTH * HEIGHT, dtype=np.uint8)
        self.window = deque(maxlen=256)
        self.previous: Optional[VisionFeature] = None

        self.camera = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        self.camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc("Y", "U", "Y", "V"),
        )
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if not self.camera.isOpened():
            self.close()
            raise RuntimeError("Cannot open the V4L2 camera")

        # Capture runs independently so the 30 FPS USB transfer overlaps PL
        # DMA and PS contour extraction.  Only the newest frame is retained;
        # stale camera frames are intentionally dropped for low latency.
        self._capture_condition = threading.Condition()
        self._capture_running = True
        self._capture_error = None
        self._latest_capture = None
        self._capture_sequence = 0
        self._consumed_sequence = 0
        self._capture_started = time.monotonic()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="fitguard-camera-capture",
            daemon=True,
        )
        self._capture_thread.start()

    def _capture_loop(self):
        while self._capture_running:
            ok, raw = self.camera.read()
            if not ok:
                if not self._capture_running:
                    break
                with self._capture_condition:
                    self._capture_error = RuntimeError("Camera read failed")
                    self._capture_condition.notify_all()
                break
            if raw.dtype != np.uint8 or raw.nbytes != RAW_BYTES:
                with self._capture_condition:
                    self._capture_error = RuntimeError(
                        f"Expected {RAW_BYTES} YUYV bytes, got shape={raw.shape}, "
                        f"dtype={raw.dtype}, bytes={raw.nbytes}"
                    )
                    self._capture_condition.notify_all()
                break
            captured = (time.monotonic(), np.ascontiguousarray(raw).copy())
            with self._capture_condition:
                self._latest_capture = captured
                self._capture_sequence += 1
                self._capture_condition.notify_all()

    def capture(self, timeout_seconds=2.0):
        deadline = time.monotonic() + timeout_seconds
        with self._capture_condition:
            while (
                self._capture_sequence == self._consumed_sequence
                and self._capture_error is None
                and self._capture_running
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for a new camera frame")
                self._capture_condition.wait(timeout=remaining)

            if self._capture_error is not None:
                raise self._capture_error
            if self._latest_capture is None:
                raise RuntimeError("Camera capture stopped before producing a frame")

            self._consumed_sequence = self._capture_sequence
            return self._latest_capture

    def run_pl(self, raw, timeout_seconds=3.0):
        self.input_buffer.view(np.uint8)[:] = raw.reshape(-1)
        self.output_buffer[:] = 0
        started = time.monotonic()
        self.dma.recvchannel.transfer(self.output_buffer)
        self.dma.sendchannel.transfer(self.input_buffer)

        deadline = started + timeout_seconds
        while not (self.dma.sendchannel.idle and self.dma.recvchannel.idle):
            if time.monotonic() >= deadline:
                mm2s = int(self.dma.mmio.read(0x04))
                s2mm = int(self.dma.mmio.read(0x34))
                raise TimeoutError(
                    f"DMA timeout: MM2S_DMASR=0x{mm2s:08X}, "
                    f"S2MM_DMASR=0x{s2mm:08X}"
                )
            time.sleep(0.001)

        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        self.output_buffer.invalidate()
        dma_ms = (time.monotonic() - started) * 1000.0

        words = np.asarray(self.output_buffer)
        self.binary_flat[0::2] = (words & 0xFF).astype(np.uint8)
        self.binary_flat[1::2] = ((words >> 8) & 0xFF).astype(np.uint8)
        mask = self.binary_flat.reshape(HEIGHT, WIDTH).copy()
        values = np.unique(mask)
        if not np.all(np.isin(values, [0, 255])):
            raise RuntimeError(f"PL output is not binary: {values.tolist()}")
        return mask, dma_ms

    def process_frame(self, min_area=200.0):
        total_started = time.monotonic()
        timestamp, raw = self.capture()
        mask, dma_ms = self.run_pl(raw)

        feature_started = time.monotonic()
        # Final path: contours only. No threshold/erode/dilate/morphologyEx here.
        targets = extract_targets(mask, min_area=min_area)
        bgr = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
        annotated = annotate(bgr, targets)

        if targets:
            target = targets[0]
            cx, cy = target.center
            _, _, width, height = target.bbox
            valid = 1
            area = target.area
        else:
            cx = cy = width = height = 0
            valid = 0
            area = 0.0

        vx = vy = 0.0
        if valid and self.previous is not None and self.previous.valid:
            dt = timestamp - self.previous.timestamp
            if dt > 0:
                vx = (cx / WIDTH - self.previous.cx) / dt
                vy = (cy / HEIGHT - self.previous.cy) / dt

        feature_ms = (time.monotonic() - feature_started) * 1000.0
        record = VisionFeature(
            timestamp=timestamp,
            valid=valid,
            cx=cx / WIDTH,
            cy=cy / HEIGHT,
            area=area / (WIDTH * HEIGHT),
            width=width / WIDTH,
            height=height / HEIGHT,
            vx=vx,
            vy=vy,
            dma_ms=dma_ms,
            feature_ms=feature_ms,
            total_ms=(time.monotonic() - total_started) * 1000.0,
        )
        self.previous = record
        self.window.append(record)
        return raw, mask, targets, annotated, record

    def feature_window(self):
        if len(self.window) < 2:
            return None
        fields = ("cx", "cy", "area", "width", "height", "vx", "vy")
        items = list(self.window)
        timestamps = np.asarray([item.timestamp for item in items], dtype=np.float64)
        end_time = timestamps[-1]
        sample_times = end_time - (
            np.arange(FEATURE_WINDOW_SAMPLES - 1, -1, -1, dtype=np.float64)
            / FEATURE_WINDOW_HZ
        )
        if timestamps[0] > sample_times[0]:
            return None

        source = np.asarray(
            [[getattr(item, field) for field in fields] for item in items],
            dtype=np.float64,
        )
        resampled = np.column_stack(
            [np.interp(sample_times, timestamps, source[:, index]) for index in range(len(fields))]
        )
        return resampled.astype(np.float32)

    def capture_fps(self):
        elapsed = time.monotonic() - self._capture_started
        return self._capture_sequence / elapsed if elapsed > 0 else 0.0

    def close(self):
        self._capture_running = False
        if getattr(self, "camera", None) is not None:
            self.camera.release()
            self.camera = None
        condition = getattr(self, "_capture_condition", None)
        if condition is not None:
            with condition:
                condition.notify_all()
        capture_thread = getattr(self, "_capture_thread", None)
        if capture_thread is not None and capture_thread.is_alive():
            capture_thread.join(timeout=1.0)
        self._capture_thread = None
        for name in ("input_buffer", "output_buffer"):
            buffer = getattr(self, name, None)
            if buffer is not None:
                if hasattr(buffer, "close"):
                    buffer.close()
                else:
                    buffer.freebuffer()
                setattr(self, name, None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def run_and_record(
    bitstream="fitguard_camera.bit",
    seconds=30,
    min_area=200,
    output_dir="camera_results",
    preview_fps=10,
):
    from IPython.display import Image, display

    try:
        import ipywidgets as widgets

        preview_widget = widgets.Image(
            format="jpeg",
            layout=widgets.Layout(width=f"{WIDTH * 2}px"),
        )
        display(preview_widget)
    except Exception:
        preview_widget = None

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    run_started = time.monotonic()
    deadline = run_started + seconds
    last_mask = last_annotated = None
    preview_handle = None
    next_preview = 0.0
    preview_interval = 1.0 / max(float(preview_fps), 1.0)

    with FitGuardCamera(bitstream=bitstream) as camera:
        while time.monotonic() < deadline:
            _, mask, targets, annotated, record = camera.process_frame(min_area)
            records.append(asdict(record))
            last_mask = mask
            last_annotated = annotated

            # Reuse one JPEG display object instead of rebuilding a matplotlib
            # figure.  This keeps the browser preview smooth without slowing
            # down the frame-processing and 64-frame feature window.
            now = time.monotonic()
            if now >= next_preview:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                header = np.zeros((28, WIDTH * 2, 3), dtype=np.uint8)
                cv2.putText(
                    header,
                    "PL threshold + open + close",
                    (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    header,
                    f"OpenCV contours  targets={len(targets)}",
                    (WIDTH + 6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                preview = np.vstack((header, np.hstack((mask_bgr, annotated))))
                ok, jpeg = cv2.imencode(
                    ".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 82]
                )
                if ok:
                    jpeg_bytes = jpeg.tobytes()
                    if preview_widget is not None:
                        preview_widget.value = jpeg_bytes
                    else:
                        frame = Image(data=jpeg_bytes, format="jpeg")
                        if preview_handle is None:
                            preview_handle = display(frame, display_id=True)
                        else:
                            preview_handle.update(frame)
                next_preview = now + preview_interval

        window = camera.feature_window()
        source_capture_fps = camera.capture_fps()

    actual_elapsed = time.monotonic() - run_started

    if last_mask is not None:
        cv2.imwrite(str(output / "latest_pl_mask.png"), last_mask)
        cv2.imwrite(str(output / "latest_annotated.png"), last_annotated)
    if records:
        with (output / "camera_metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        summary = {
            "frames": len(records),
            "seconds_requested": seconds,
            "seconds_actual": actual_elapsed,
            "measured_fps": len(records) / actual_elapsed,
            "camera_capture_fps": source_capture_fps,
            "dma_ms_mean": float(np.mean([r["dma_ms"] for r in records])),
            "total_ms_mean": float(np.mean([r["total_ms"] for r in records])),
            "feature_window_ready": window is not None,
            "feature_window_shape": list(window.shape) if window is not None else None,
            "feature_window_rate_hz": FEATURE_WINDOW_HZ,
            "feature_window_seconds": FEATURE_WINDOW_SAMPLES / FEATURE_WINDOW_HZ,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(summary)
    if window is not None:
        np.save(output / "visual_window_64x7.npy", window)
    print("Results saved in", output.resolve())
