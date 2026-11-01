"""Isolated dual-modal collector for the ESP32 BLE-to-USB gateway."""

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from fitguard_camera_runtime import FitGuardCamera
from fitguard_dual_collector import (
    IMU_FIELDS,
    IMU_SAMPLES,
    LABELS,
    STRIDE_SECONDS,
    VISION_FIELDS,
    VISION_SAMPLES,
    WINDOW_SECONDS,
    _resample,
    _write_csv,
)
from fitguard_gateway_imu import GatewayMPU6050
from motion_tracker_v2 import MotionPersonTracker


VISION_AUX_FIELDS = ("valid", "quality", "motion_score")


def collect_trial_gateway(
    subject,
    label,
    seconds=30,
    trial="01",
    bitstream="fitguard_unified.bit",
    output_root="dataset_motion_v2",
    min_area=200,
    calibrate_seconds=2.0,
    preview_callback=None,
    preview_interval=0.5,
    port="/dev/ttyUSB0",
    training_candidate=True,
    min_visual_hz=10.0,
):
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    if seconds < WINDOW_SECONDS:
        raise ValueError(f"seconds must be at least {WINDOW_SECONDS}")
    if Path(output_root).resolve().name in {"dataset", "dataset_ble", "dataset_gateway"}:
        raise ValueError("Gateway collector refuses to write into an existing dataset folder")

    safe_subject = "".join(c for c in str(subject) if c.isalnum() or c in "-_")
    safe_trial = "".join(c for c in str(trial) if c.isalnum() or c in "-_")
    if not safe_subject or not safe_trial:
        raise ValueError("subject and trial must contain letters, numbers, '-' or '_'")
    label_slug = label.lower().replace(" ", "_")
    output_parent = Path(output_root) / safe_subject / label_slug
    final_output = output_parent / f"trial_{safe_trial}"
    staging_output = output_parent / f".inprogress_trial_{safe_trial}"
    output_parent.mkdir(parents=True, exist_ok=True)
    if final_output.exists() and any(final_output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing trial: {final_output}")
    for stale in (final_output, staging_output):
        if stale.exists():
            failed = output_parent / f".failed_{stale.name}_{time.time_ns()}"
            stale.rename(failed)
    staging_output.mkdir()
    output = staging_output

    imu_rows, vision_rows = [], []
    stop_event, imu_error = threading.Event(), []
    last_pl_mask = last_motion_mask = last_annotated = None
    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    last_preview_time = 0.0
    imu = GatewayMPU6050(port=port)
    tracker = MotionPersonTracker()

    with FitGuardCamera(bitstream=bitstream) as camera:
        try:
            identity = imu.initialize()
            calibration_samples = max(1, int(round(calibrate_seconds * 50)))
            print(f"Gateway connected. Keep the waist unit still for {calibrate_seconds:.1f}s...")
            gyro_bias = imu.calibrate_gyro(calibration_samples)

            def sample_imu():
                try:
                    while not stop_event.is_set():
                        imu_rows.append(asdict(imu.read_sample()))
                except Exception as exc:
                    imu_error.append(exc)
                    stop_event.set()

            imu_thread = threading.Thread(target=sample_imu, daemon=True)
            imu_thread.start()
            deadline = time.monotonic() + float(seconds)
            try:
                while time.monotonic() < deadline and not stop_event.is_set():
                    timestamp, raw = camera.capture()
                    pl_mask, _ = camera.run_pl(raw)
                    record, motion_mask, annotated = tracker.update(raw, pl_mask, timestamp)
                    vision_rows.append(asdict(record))
                    last_pl_mask = pl_mask
                    last_motion_mask = motion_mask
                    last_annotated = annotated
                    now = time.monotonic()
                    if preview_callback is not None and now - last_preview_time >= float(preview_interval):
                        preview_callback(annotated, motion_mask, record)
                        last_preview_time = now
            finally:
                stop_event.set()
                imu_thread.join(timeout=2.0)
            camera_fps = camera.capture_fps()
        finally:
            gateway_stats = imu.diagnostics()
            imu.close()

    if imu_error:
        raise RuntimeError("Gateway MPU6050 sampling failed") from imu_error[0]
    if not imu_rows or not vision_rows:
        raise RuntimeError("No synchronized gateway/camera data was captured")

    first_common = max(imu_rows[0]["timestamp"], vision_rows[0]["timestamp"])
    last_common = min(imu_rows[-1]["timestamp"], vision_rows[-1]["timestamp"])
    window_ends = np.arange(first_common + WINDOW_SECONDS, last_common + 1e-9, STRIDE_SECONDS)
    imu_windows, vision_windows, vision_aux_windows, kept_ends = [], [], [], []
    for end_time in window_ends:
        imu_window = _resample(imu_rows, IMU_FIELDS, end_time, IMU_SAMPLES, WINDOW_SECONDS)
        vision_window = _resample(vision_rows, VISION_FIELDS, end_time, VISION_SAMPLES, WINDOW_SECONDS)
        vision_aux_window = _resample(
            vision_rows, VISION_AUX_FIELDS, end_time, VISION_SAMPLES, WINDOW_SECONDS
        )
        if imu_window is not None and vision_window is not None and vision_aux_window is not None:
            imu_windows.append(imu_window)
            vision_windows.append(vision_window)
            vision_aux_windows.append(vision_aux_window)
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
        vision_aux=np.stack(vision_aux_windows),
        vision_fields=np.asarray(VISION_FIELDS),
        vision_aux_fields=np.asarray(VISION_AUX_FIELDS),
        window_end=np.asarray(kept_ends),
        label=np.asarray([label] * len(kept_ends)),
        subject=np.asarray([safe_subject] * len(kept_ends)),
    )
    if last_pl_mask is not None:
        cv2.imwrite(str(output / "latest_pl_mask.png"), last_pl_mask)
        cv2.imwrite(str(output / "latest_motion_mask.png"), last_motion_mask)
        cv2.imwrite(str(output / "latest_annotated.png"), last_annotated)

    measured_imu_hz = len(imu_rows) / elapsed if elapsed > 0 else 0.0
    measured_vision_hz = len(vision_rows) / elapsed if elapsed > 0 else 0.0
    vision_valid_ratio = sum(int(row["valid"]) for row in vision_rows) / len(vision_rows)
    mean_tracking_quality = float(np.mean([
        row["quality"] for row in vision_rows if row["valid"]
    ])) if any(row["valid"] for row in vision_rows) else 0.0
    quality_checks = {
        "clock_year_at_least_2026": time.localtime().tm_year >= 2026,
        "imu_rate_45_to_55_hz": 45.0 <= measured_imu_hz <= 55.0,
        "vision_rate_at_least_minimum": measured_vision_hz >= float(min_visual_hz),
        "gateway_receive_ratio_at_least_0_995": gateway_stats.get("receive_ratio", 0.0) >= 0.995,
        "gateway_bad_crc_zero": gateway_stats.get("bad_crc", 1) == 0,
        "gateway_bad_length_zero": gateway_stats.get("bad_length", 1) == 0,
        "gateway_lost_samples_zero": gateway_stats.get("lost_samples", 1) == 0,
        "gateway_out_of_order_zero": gateway_stats.get("out_of_order_samples", 1) == 0,
        "gateway_queue_drops_zero": gateway_stats.get("queue_drops", 1) == 0,
        "vision_valid_ratio_at_least_0_75": vision_valid_ratio >= 0.75,
        "mean_tracking_quality_at_least_0_35": mean_tracking_quality >= 0.35,
        "aligned_windows_present": len(imu_windows) > 0,
    }
    summary = {
        "dataset_format_version": 2,
        "subject": safe_subject,
        "label": label,
        "trial": safe_trial,
        "started": started_wall,
        "sensor_transport": "BLE via ESP32 USB gateway",
        "port": port,
        "bitstream": str(Path(bitstream).resolve()),
        "who_am_i": f"0x{identity:02X}",
        "gyro_bias_deg_s": gyro_bias.tolist(),
        "imu_rows": len(imu_rows),
        "vision_rows": len(vision_rows),
        "aligned_windows": len(imu_windows),
        "measured_imu_hz": measured_imu_hz,
        "measured_vision_hz": measured_vision_hz,
        "camera_capture_fps": camera_fps,
        "vision_tracker": "motion_tracker_v2",
        "vision_window_shape": [VISION_SAMPLES, len(VISION_FIELDS)],
        "vision_aux_window_shape": [VISION_SAMPLES, len(VISION_AUX_FIELDS)],
        "vision_valid_ratio": vision_valid_ratio,
        "mean_tracking_quality": mean_tracking_quality,
        "gateway": gateway_stats,
        "training_candidate": bool(training_candidate),
        "quality_checks": quality_checks,
        "accepted_for_training": bool(training_candidate and all(quality_checks.values())),
    }
    accepted_for_training = bool(training_candidate and all(quality_checks.values()))
    summary["accepted_for_training"] = accepted_for_training
    # A failed formal-quality gate must never look like an ordinary trial to
    # downstream training scripts.  Keep it for diagnosis, but outside the
    # conventional ``trial_*`` location so it cannot be consumed by accident.
    if training_candidate and not accepted_for_training:
        rejected_root = output_parent / "rejected_trials"
        rejected_root.mkdir(exist_ok=True)
        destination = rejected_root / f"trial_{safe_trial}_{time.time_ns()}"
    else:
        destination = final_output
    summary["saved_path"] = str(destination.resolve())
    (output / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    output.rename(destination)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if accepted_for_training:
        print("Gateway trial ACCEPTED for training:", destination.resolve())
    elif training_candidate:
        print("Gateway trial REJECTED by quality gate; kept for diagnosis:", destination.resolve())
    else:
        print("Gateway test trial saved (not a training candidate):", destination.resolve())
    return summary


def check_gateway_environment():
    """Safe pre-hardware diagnostic; does not load the FPGA overlay."""
    import glob
    import os
    import termios
    return {
        "serial_ports": sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")),
        "baud_921600_supported": hasattr(termios, "B921600"),
        "usb_devices": os.listdir("/sys/bus/usb/devices") if os.path.isdir("/sys/bus/usb/devices") else [],
    }
