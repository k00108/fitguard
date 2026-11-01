"""Live dual-modal demo using the isolated ESP32 BLE-to-USB gateway path."""

import json
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fitguard_live_classifier import (
    FusionNumpyCNN,
    IMU_FIELDS,
    IMU_SAMPLES,
    LABELS,
    VISION_FIELDS,
    VISION_SAMPLES,
    WINDOW_SECONDS,
    _resample,
)


def run_live_gateway(
    bitstream="fitguard_unified.bit",
    model_dir="trained_models_waist_v2",
    seconds=60,
    min_area=200,
    preview_fps=8,
    prediction_stride=0.32,
    smoothing_alpha=0.60,
    port="/dev/ttyUSB0",
):
    import cv2
    from IPython.display import Image, display
    from fitguard_camera_runtime import FitGuardCamera
    from fitguard_gateway_imu import GatewayMPU6050

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

    imu = GatewayMPU6050(port=port)
    with FitGuardCamera(bitstream=bitstream) as camera:
        try:
            identity = imu.initialize()
            print(f"ESP32 gateway connected (WHO_AM_I=0x{identity:02X}). Keep the waist unit still...")
            imu.calibrate_gyro(100)

            def sample_imu():
                try:
                    while not stop_event.is_set():
                        imu_rows.append(asdict(imu.read_sample()))
                except Exception as exc:
                    imu_error.append(exc)
                    stop_event.set()

            thread = threading.Thread(target=sample_imu, daemon=True)
            thread.start()
            deadline = time.monotonic() + float(seconds)
            try:
                while time.monotonic() < deadline and not stop_event.is_set():
                    _, mask, _, annotated, record = camera.process_frame(min_area)
                    vision_rows.append(asdict(record))
                    end_time = min(
                        imu_rows[-1]["timestamp"] if imu_rows else 0,
                        vision_rows[-1]["timestamp"],
                    )
                    if end_time - last_inference >= prediction_stride:
                        imu_window = _resample(
                            list(imu_rows), IMU_FIELDS, end_time, IMU_SAMPLES, WINDOW_SECONDS
                        )
                        vision_window = _resample(
                            list(vision_rows), VISION_FIELDS, end_time, VISION_SAMPLES, WINDOW_SECONDS
                        )
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
                            predictions.append({
                                "timestamp": end_time,
                                "label": LABELS[int(last_prediction.argmax())],
                                "confidence": float(last_prediction.max()),
                                "probabilities": last_prediction.tolist(),
                                "raw_probabilities": raw_probability.tolist(),
                                "inference_ms": (time.monotonic() - started) * 1000.0,
                            })
                            last_inference = end_time

                    now = time.monotonic()
                    if now >= next_preview:
                        canvas = annotated.copy()
                        label_index = int(last_prediction.argmax())
                        cv2.rectangle(canvas, (0, 0), (345, 101), (0, 0, 0), -1)
                        cv2.putText(canvas, "ESP32 USB gateway", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1)
                        cv2.putText(canvas, LABELS[label_index], (8, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
                        cv2.putText(canvas, f"confidence {last_prediction[label_index]:.1%}", (8, 69), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                        cv2.putText(canvas, f"S {last_prediction[0]:.0%}  M {last_prediction[1]:.0%}  X {last_prediction[2]:.0%}", (8, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (180, 255, 180), 1)
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
                thread.join(timeout=2.0)
        finally:
            gateway_stats = imu.diagnostics()
            imu.close()

    if imu_error:
        raise RuntimeError("Gateway MPU6050 sampling failed") from imu_error[0]
    result = {"predictions": predictions, "gateway": gateway_stats}
    Path("live_predictions_gateway.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(gateway_stats, ensure_ascii=False, indent=2))
    print(f"Saved {len(predictions)} predictions to live_predictions_gateway.json")
    return result
