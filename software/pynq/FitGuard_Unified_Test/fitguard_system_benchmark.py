"""Long-duration acceptance benchmark for the deployed FitGuard system.

The script exercises the final unified overlay, USB camera, PL preprocessing,
OpenCV contour features, MPU6050 sampling, and the deployed fusion model.  It
does not change the bitstream, model, or dataset.
"""

import csv
import json
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fitguard_camera_runtime import FitGuardCamera
from fitguard_live_classifier import (
    FusionNumpyCNN,
    IMU_FIELDS,
    IMU_SAMPLES,
    VISION_FIELDS,
    VISION_SAMPLES,
    WINDOW_SECONDS,
    _resample,
)
from fitguard_mpu6050 import MPU6050


def _percentiles(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    bitstream="fitguard_unified.bit",
    model_dir="trained_models_waist_v2",
    seconds=600,
    output_dir="validation_results/stability_10min",
    min_area=200,
    prediction_stride=0.32,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    imu_rows = deque(maxlen=512)
    vision_rows = deque(maxlen=256)
    frame_rows = []
    prediction_rows = []
    resource_rows = []
    imu_error = []
    stop_event = threading.Event()
    model = FusionNumpyCNN(model_dir)

    try:
        import psutil

        process = psutil.Process()
        process.cpu_percent(None)
    except Exception:
        process = None

    started_wall = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    started = time.monotonic()
    status = "failed"
    failure = None

    try:
        with FitGuardCamera(bitstream=bitstream) as camera:
            imu = MPU6050(camera.overlay.axi_iic_0)
            identity = imu.initialize()
            print("Keep MPU6050 still for 2 seconds...")
            gyro_bias = imu.calibrate_gyro(100, 0.02)

            def sample_imu():
                next_sample = time.monotonic()
                try:
                    while not stop_event.is_set():
                        imu_rows.append(asdict(imu.read_sample()))
                        next_sample += 0.02
                        stop_event.wait(max(0.0, next_sample - time.monotonic()))
                except Exception as exc:
                    imu_error.append(repr(exc))
                    stop_event.set()

            imu_thread = threading.Thread(target=sample_imu, daemon=True)
            imu_thread.start()
            run_started = time.monotonic()
            deadline = run_started + float(seconds)
            capture_sequence_start = camera._capture_sequence
            last_inference = 0.0
            last_resource = 0.0
            progress_interval = 10.0
            next_progress = run_started + progress_interval

            try:
                while time.monotonic() < deadline and not stop_event.is_set():
                    _, _, targets, _, record = camera.process_frame(min_area)
                    record_dict = asdict(record)
                    vision_rows.append(record_dict)
                    frame_rows.append(record_dict)

                    end_time = min(
                        imu_rows[-1]["timestamp"] if imu_rows else 0.0,
                        vision_rows[-1]["timestamp"],
                    )
                    if end_time - last_inference >= prediction_stride:
                        imu_window = _resample(
                            list(imu_rows), IMU_FIELDS, end_time,
                            IMU_SAMPLES, WINDOW_SECONDS,
                        )
                        vision_window = _resample(
                            list(vision_rows), VISION_FIELDS, end_time,
                            VISION_SAMPLES, WINDOW_SECONDS,
                        )
                        if imu_window is not None and vision_window is not None:
                            inference_started = time.monotonic()
                            probability = model.predict(imu_window, vision_window)
                            inference_ms = (time.monotonic() - inference_started) * 1000.0
                            prediction_rows.append({
                                "timestamp": float(end_time),
                                "class_index": int(probability.argmax()),
                                "confidence": float(probability.max()),
                                "stable": float(probability[0]),
                                "mild": float(probability[1]),
                                "marked": float(probability[2]),
                                "inference_ms": float(inference_ms),
                            })
                            last_inference = end_time

                    now = time.monotonic()
                    if process is not None and now - last_resource >= 1.0:
                        resource_rows.append({
                            "elapsed_s": float(now - run_started),
                            "cpu_percent": float(process.cpu_percent(None)),
                            "rss_mb": float(process.memory_info().rss / (1024.0 * 1024.0)),
                        })
                        last_resource = now
                    if now >= next_progress:
                        print(
                            "progress",
                            round(now - run_started, 1),
                            "frames", len(frame_rows),
                            "predictions", len(prediction_rows),
                        )
                        next_progress += progress_interval
            finally:
                stop_event.set()
                imu_thread.join(timeout=2.0)

            if imu_error:
                raise RuntimeError("MPU6050 sampling failed: " + imu_error[0])

            elapsed = time.monotonic() - run_started
            captured = int(camera._capture_sequence - capture_sequence_start)
            processed = len(frame_rows)
            dropped = max(0, captured - processed)
            status = "passed"
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        identity = 0
        gyro_bias = np.zeros(3, dtype=np.float32)
        captured = processed = dropped = 0

    _write_csv(output / "frames.csv", frame_rows)
    _write_csv(output / "predictions.csv", prediction_rows)
    _write_csv(output / "resources.csv", resource_rows)

    frame_timestamps = [row["timestamp"] for row in frame_rows]
    frame_intervals_ms = (
        np.diff(np.asarray(frame_timestamps, dtype=np.float64)) * 1000.0
        if len(frame_timestamps) >= 2 else np.asarray([])
    )
    summary = {
        "status": status,
        "failure": failure,
        "started": started_wall,
        "requested_seconds": float(seconds),
        "elapsed_seconds": float(elapsed),
        "bitstream": bitstream,
        "model_dir": model_dir,
        "mpu6050_who_am_i": f"0x{identity:02X}",
        "gyro_bias_deg_s": np.asarray(gyro_bias).tolist(),
        "processed_frames": len(frame_rows),
        "captured_frames": int(captured),
        "intentionally_dropped_stale_frames": int(dropped),
        "processing_fps": float(len(frame_rows) / elapsed) if elapsed > 0 else None,
        "imu_samples_retained_at_end": len(imu_rows),
        "predictions": len(prediction_rows),
        "valid_target_ratio": (
            float(np.mean([row["valid"] for row in frame_rows])) if frame_rows else None
        ),
        "frame_interval_ms": _percentiles(frame_intervals_ms),
        "dma_ms": _percentiles([row["dma_ms"] for row in frame_rows]),
        "opencv_feature_ms": _percentiles([row["feature_ms"] for row in frame_rows]),
        "frame_total_ms": _percentiles([row["total_ms"] for row in frame_rows]),
        "cnn_inference_ms": _percentiles([row["inference_ms"] for row in prediction_rows]),
        "cpu_percent": _percentiles([row["cpu_percent"] for row in resource_rows]),
        "rss_mb": _percentiles([row["rss_mb"] for row in resource_rows]),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if status != "passed":
        raise RuntimeError(failure)
    return summary


if __name__ == "__main__":
    run_benchmark()
