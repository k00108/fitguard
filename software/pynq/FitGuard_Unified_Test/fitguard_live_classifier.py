"""Dependency-free NumPy CNN inference for the PYNQ FitGuard demo."""

import json
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

LABELS = ("Stable", "Mild Abnormality", "Marked Abnormality")
IMU_FIELDS = ("ax", "ay", "az", "gx", "gy", "gz")
VISION_FIELDS = ("cx", "cy", "area", "width", "height", "vx", "vy")
IMU_SAMPLES = 128
VISION_SAMPLES = 64
WINDOW_SECONDS = 2.56


def _resample(rows, fields, end_time, samples, duration):
    if len(rows) < 2:
        return None
    timestamp = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
    start_time = end_time - duration
    if timestamp[0] > start_time or timestamp[-1] < end_time:
        return None
    target = np.linspace(start_time, end_time, samples, endpoint=False)
    values = np.asarray([[row[field] for field in fields] for row in rows])
    return np.column_stack(
        [np.interp(target, timestamp, values[:, index]) for index in range(len(fields))]
    ).astype(np.float32)


def _conv1d_same(x, weights, bias):
    kernel = weights.shape[0]
    left = (kernel - 1) // 2
    right = kernel - 1 - left
    padded = np.pad(x, ((left, right), (0, 0)))
    windows = np.stack([padded[index:index + kernel] for index in range(len(x))])
    return np.maximum(np.tensordot(windows, weights, axes=([1, 2], [0, 1])) + bias, 0)


def _branch(x, weights, prefix):
    x = _conv1d_same(x, weights[f"{prefix}_conv1_w"], weights[f"{prefix}_conv1_b"])
    usable = len(x) - len(x) % 2
    x = x[:usable].reshape(usable // 2, 2, x.shape[1]).max(axis=1)
    x = _conv1d_same(x, weights[f"{prefix}_conv2_w"], weights[f"{prefix}_conv2_b"])
    x = x.mean(axis=0)
    return np.maximum(x @ weights[f"{prefix}_dense_w"] + weights[f"{prefix}_dense_b"], 0)


def _canonicalize_waist_window(imu, vision):
    imu = imu.astype(np.float32, copy=True)
    vision = vision.astype(np.float32, copy=True)
    imu -= imu.mean(axis=0, keepdims=True)
    present = vision[:, 2] > 1e-8
    if present.any():
        baseline = np.median(vision[present, :5], axis=0)
        vision[present, 0:2] -= baseline[0:2]
        vision[present, 2:5] = (
            vision[present, 2:5] / np.maximum(baseline[2:5], 1e-4) - 1.0
        )
        vision[~present] = 0.0
    else:
        vision[:] = 0.0
    return imu, vision


def _motion_stats(x):
    rms = np.sqrt(np.mean(np.square(x), axis=0) + 1e-6)
    change = np.mean(np.abs(x[1:] - x[:-1]), axis=0)
    return np.concatenate((rms, change)).astype(np.float32)


class FusionNumpyCNN:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        weight_path = model_dir / "fitguard_fusion_weights.npz"
        norm_path = model_dir / "normalization.npz"
        if not weight_path.exists() or not norm_path.exists():
            raise FileNotFoundError(
                f"Copy fitguard_fusion_weights.npz and normalization.npz to {model_dir}"
            )
        with np.load(weight_path, allow_pickle=False) as source:
            self.weights = {key: source[key].astype(np.float32) for key in source.files if key not in ("model_name", "labels")}
            labels = source["labels"].astype(str).tolist()
        if tuple(labels) != LABELS:
            raise ValueError(f"Model label order does not match runtime: {labels}")
        with np.load(norm_path, allow_pickle=False) as source:
            self.imu_mean = source["imu_mean"].astype(np.float32)
            self.imu_std = source["imu_std"].astype(np.float32)
            self.vision_mean = source["vision_mean"].astype(np.float32)
            self.vision_std = source["vision_std"].astype(np.float32)
            self.preprocessing = (
                str(source["preprocessing"])
                if "preprocessing" in source.files
                else "none"
            )

    def predict(self, imu, vision):
        if self.preprocessing == "waist_centered_v1":
            imu, vision = _canonicalize_waist_window(imu, vision)
        imu = (imu - self.imu_mean) / self.imu_std
        vision = (vision - self.vision_mean) / self.vision_std
        imu = np.clip(imu, -8.0, 8.0)
        vision = np.clip(vision, -8.0, 8.0)
        imu_features = _branch(imu, self.weights, "imu")
        vision_features = _branch(vision, self.weights, "vision")
        fused = np.concatenate((imu_features, vision_features))
        expected_features = self.weights["fusion_dense_w"].shape[0]
        if expected_features == fused.size + 26:
            fused = np.concatenate(
                (fused, _motion_stats(imu), _motion_stats(vision))
            )
        elif expected_features != fused.size:
            raise ValueError(
                f"Unsupported fusion feature size: weights expect {expected_features}, "
                f"runtime produced {fused.size}"
            )
        fused = np.maximum(
            fused @ self.weights["fusion_dense_w"] + self.weights["fusion_dense_b"], 0
        )
        logits = fused @ self.weights["fusion_output_w"] + self.weights["fusion_output_b"]
        shifted = logits - logits.max()
        probability = np.exp(shifted)
        return probability / probability.sum()


def run_live(
    bitstream="fitguard_unified.bit",
    model_dir="trained_models",
    seconds=60,
    min_area=200,
    preview_fps=8,
    prediction_stride=0.32,
    smoothing_alpha=0.60,
):
    import cv2
    from IPython.display import Image, display
    from fitguard_camera_runtime import FitGuardCamera
    from fitguard_mpu6050 import MPU6050

    model = FusionNumpyCNN(model_dir)
    imu_rows, vision_rows = deque(maxlen=512), deque(maxlen=256)
    predictions = []
    stop_event, imu_error = threading.Event(), []
    last_prediction = np.ones(len(LABELS), dtype=np.float32) / len(LABELS)
    prediction_ready = False
    last_inference = 0.0

    try:
        import ipywidgets as widgets
        widget = widgets.Image(format="jpeg", layout=widgets.Layout(width="640px"))
        display(widget)
    except Exception:
        widget = None
    display_handle = None
    next_preview = 0.0

    with FitGuardCamera(bitstream=bitstream) as camera:
        imu = MPU6050(camera.overlay.axi_iic_0)
        imu.initialize()
        print("Keep MPU6050 still for 2 seconds...")
        imu.calibrate_gyro(100, 0.02)

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

        thread = threading.Thread(target=sample_imu, daemon=True)
        thread.start()
        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline and not stop_event.is_set():
                _, mask, _, annotated, record = camera.process_frame(min_area)
                vision_rows.append(asdict(record))
                end_time = min(
                    imu_rows[-1]["timestamp"] if imu_rows else 0,
                    vision_rows[-1]["timestamp"],
                )
                if end_time - last_inference >= prediction_stride:
                    imu_window = _resample(list(imu_rows), IMU_FIELDS, end_time, IMU_SAMPLES, WINDOW_SECONDS)
                    vision_window = _resample(list(vision_rows), VISION_FIELDS, end_time, VISION_SAMPLES, WINDOW_SECONDS)
                    if imu_window is not None and vision_window is not None:
                        started = time.monotonic()
                        raw_probability = model.predict(imu_window, vision_window)
                        if prediction_ready:
                            last_prediction = (
                                smoothing_alpha * raw_probability
                                + (1.0 - smoothing_alpha) * last_prediction
                            )
                        else:
                            last_prediction = raw_probability
                            prediction_ready = True
                        inference_ms = (time.monotonic() - started) * 1000
                        prediction = {
                            "timestamp": end_time,
                            "label": LABELS[int(last_prediction.argmax())],
                            "confidence": float(last_prediction.max()),
                            "probabilities": last_prediction.tolist(),
                            "raw_probabilities": raw_probability.tolist(),
                            "inference_ms": inference_ms,
                        }
                        predictions.append(prediction)
                        last_inference = end_time

                now = time.monotonic()
                if now >= next_preview:
                    canvas = annotated.copy()
                    label_index = int(last_prediction.argmax())
                    cv2.rectangle(canvas, (0, 0), (320, 84), (0, 0, 0), -1)
                    cv2.putText(canvas, LABELS[label_index], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
                    cv2.putText(canvas, f"confidence {last_prediction[label_index]:.1%}", (8, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                    cv2.putText(
                        canvas,
                        f"S {last_prediction[0]:.0%}  M {last_prediction[1]:.0%}  X {last_prediction[2]:.0%}",
                        (8, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (180, 255, 180), 1,
                    )
                    preview = np.hstack((cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), canvas))
                    ok, jpeg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok:
                        if widget is not None:
                            widget.value = jpeg.tobytes()
                        else:
                            image = Image(data=jpeg.tobytes(), format="jpeg")
                            if display_handle is None:
                                display_handle = display(image, display_id=True)
                            else:
                                display_handle.update(image)
                    next_preview = now + 1.0 / max(preview_fps, 1)
        finally:
            stop_event.set()
            thread.join(timeout=2)

    if imu_error:
        raise RuntimeError("MPU6050 sampling failed") from imu_error[0]
    Path("live_predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(predictions)} predictions to live_predictions.json")
    return predictions
