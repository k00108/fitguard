"""Synchronized FitGuard camera/MPU6050 dataset collector.

The PL performs grayscale, threshold, 3x3 opening, and 3x3 closing.  The PS
uses OpenCV only for contours and converts both modalities to aligned windows.
"""

import csv
import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from fitguard_camera_runtime import FitGuardCamera
from fitguard_mpu6050 import MPU6050


LABELS = ("Stable", "Mild Abnormality", "Marked Abnormality")
IMU_FIELDS = ("ax", "ay", "az", "gx", "gy", "gz")
VISION_FIELDS = ("cx", "cy", "area", "width", "height", "vx", "vy")
WINDOW_SECONDS = 2.56
STRIDE_SECONDS = 1.28
IMU_SAMPLES = 128
VISION_SAMPLES = 64


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _resample(rows, fields, end_time, samples, duration):
    if len(rows) < 2:
        return None
    timestamp = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
    start_time = end_time - duration
    if timestamp[0] > start_time or timestamp[-1] < end_time:
        return None
    target = np.linspace(start_time, end_time, samples, endpoint=False)
    values = np.asarray(
        [[row[field] for field in fields] for row in rows], dtype=np.float64
    )
    return np.column_stack(
        [np.interp(target, timestamp, values[:, index]) for index in range(len(fields))]
    ).astype(np.float32)


def collect_trial(
    subject,
    label,
    seconds=30,
    trial="01",
    bitstream="fitguard_unified.bit",
    output_root="dataset",
    min_area=200,
    calibrate_seconds=2.0,
    preview_callback=None,
    preview_interval=0.25,
):
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    if seconds < WINDOW_SECONDS:
        raise ValueError(f"seconds must be at least {WINDOW_SECONDS}")

    safe_subject = "".join(c for c in str(subject) if c.isalnum() or c in "-_")
    safe_trial = "".join(c for c in str(trial) if c.isalnum() or c in "-_")
    if not safe_subject or not safe_trial:
        raise ValueError("subject and trial must contain letters, numbers, '-' or '_'")
    label_slug = label.lower().replace(" ", "_")
    output = Path(output_root) / safe_subject / label_slug / f"trial_{safe_trial}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing trial: {output}")
    output.mkdir(parents=True, exist_ok=True)

    imu_rows = []
    vision_rows = []
    stop_event = threading.Event()
    imu_error = []
    last_mask = last_annotated = None
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    last_preview_time = 0.0

    with FitGuardCamera(bitstream=bitstream) as camera:
        imu = MPU6050(camera.overlay.axi_iic_0)
        identity = imu.initialize()
        calibration_samples = max(1, int(round(calibrate_seconds * 50)))
        print(f"Keep MPU6050 still for {calibrate_seconds:.1f}s calibration...")
        gyro_bias = imu.calibrate_gyro(calibration_samples, 0.02)

        def sample_imu():
            next_sample = time.monotonic()
            try:
                while not stop_event.is_set():
                    imu_rows.append(asdict(imu.read_sample()))
                    next_sample += 0.02
                    stop_event.wait(max(0.0, next_sample - time.monotonic()))
            except Exception as exc:
                imu_error.append(exc)
                stop_event.set()

        imu_thread = threading.Thread(target=sample_imu, daemon=True)
        imu_thread.start()
        capture_start = time.monotonic()
        deadline = capture_start + float(seconds)
        try:
            while time.monotonic() < deadline and not stop_event.is_set():
                _, mask, _, annotated, record = camera.process_frame(min_area)
                vision_rows.append(asdict(record))
                last_mask, last_annotated = mask, annotated
                now = time.monotonic()
                if (
                    preview_callback is not None
                    and now - last_preview_time >= float(preview_interval)
                ):
                    preview_callback(annotated, mask, record)
                    last_preview_time = now
        finally:
            stop_event.set()
            imu_thread.join(timeout=2.0)
        camera_fps = camera.capture_fps()

    if imu_error:
        raise RuntimeError("MPU6050 sampling failed") from imu_error[0]
    if not imu_rows or not vision_rows:
        raise RuntimeError("No synchronized sensor data was captured")

    first_common = max(imu_rows[0]["timestamp"], vision_rows[0]["timestamp"])
    last_common = min(imu_rows[-1]["timestamp"], vision_rows[-1]["timestamp"])
    window_ends = np.arange(
        first_common + WINDOW_SECONDS, last_common + 1e-9, STRIDE_SECONDS
    )
    imu_windows, vision_windows, kept_ends = [], [], []
    for end_time in window_ends:
        imu_window = _resample(
            imu_rows, IMU_FIELDS, end_time, IMU_SAMPLES, WINDOW_SECONDS
        )
        vision_window = _resample(
            vision_rows, VISION_FIELDS, end_time, VISION_SAMPLES, WINDOW_SECONDS
        )
        if imu_window is not None and vision_window is not None:
            imu_windows.append(imu_window)
            vision_windows.append(vision_window)
            kept_ends.append(end_time)

    if not imu_windows:
        raise RuntimeError("Capture completed but produced no full aligned window")

    elapsed = last_common - first_common
    _write_csv(output / "imu.csv", imu_rows)
    _write_csv(output / "vision.csv", vision_rows)
    np.savez_compressed(
        output / "windows.npz",
        imu=np.stack(imu_windows),
        vision=np.stack(vision_windows),
        window_end=np.asarray(kept_ends),
        label=np.asarray([label] * len(kept_ends)),
        subject=np.asarray([safe_subject] * len(kept_ends)),
    )
    if last_mask is not None:
        cv2.imwrite(str(output / "latest_pl_mask.png"), last_mask)
        cv2.imwrite(str(output / "latest_annotated.png"), last_annotated)

    summary = {
        "subject": safe_subject,
        "label": label,
        "trial": safe_trial,
        "started": started_wall,
        "bitstream": str(Path(bitstream).resolve()),
        "who_am_i": f"0x{identity:02X}",
        "gyro_bias_deg_s": gyro_bias.tolist(),
        "imu_rows": len(imu_rows),
        "vision_rows": len(vision_rows),
        "aligned_windows": len(imu_windows),
        "imu_window_shape": [IMU_SAMPLES, len(IMU_FIELDS)],
        "vision_window_shape": [VISION_SAMPLES, len(VISION_FIELDS)],
        "window_seconds": WINDOW_SECONDS,
        "stride_seconds": STRIDE_SECONDS,
        "measured_imu_hz": len(imu_rows) / elapsed if elapsed > 0 else None,
        "measured_vision_hz": len(vision_rows) / elapsed if elapsed > 0 else None,
        "camera_capture_fps": camera_fps,
        "pl_processing": "grayscale + threshold + 3x3 opening + 3x3 closing",
        "ps_processing": "OpenCV contours and temporal features only",
    }
    (output / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Trial saved in", output.resolve())
    return summary


def check_dual_hardware(bitstream="fitguard_unified.bit", min_area=200):
    """Open both devices once and return a camera frame plus health details."""
    with FitGuardCamera(bitstream=bitstream) as camera:
        imu = MPU6050(camera.overlay.axi_iic_0)
        identity = imu.initialize()
        sample = imu.read_sample()
        _, mask, targets, annotated, record = camera.process_frame(min_area)
        camera_fps = camera.capture_fps()

    return {
        "camera_opened": True,
        "camera_capture_fps": float(camera_fps),
        "mpu6050_who_am_i": f"0x{identity:02X}",
        "imu_sample": asdict(sample),
        "targets": len(targets),
        "vision_valid": int(record.valid),
        "annotated": annotated,
        "mask": mask,
    }
