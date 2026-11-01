"""FitGuard CNN V8 PL live path with safe NumPy fallback.

The validated V6/V7 decision logic and motion_tracker_v2 stay unchanged. Only
the six convolution blocks may run in PL; dense fusion and safeguards stay PS.
"""

import json
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fitguard_live_classifier import (
    IMU_FIELDS,
    IMU_SAMPLES,
    LABELS,
    VISION_FIELDS,
    VISION_SAMPLES,
    WINDOW_SECONDS,
    _resample,
)
from fitguard_live_classifier_gateway_cnn_v6 import (
    OutputStateFilter,
    _recent_vision_health,
    _reserve_run_record,
)
from fitguard_motion_m01_m06_cnn_runtime_pl import RangeFusionCNNPL
from fitguard_event_runtime import MarkedEventRecorder
from fitguard_web_dashboard import FitGuardDashboardServer
from motion_tracker_v2 import MotionPersonTracker


VISION_AUX_FIELDS = ("valid", "quality", "motion_score")


def _percentiles(values):
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _motion_indicators(imu_window, vision_window, vision_aux_window, turn_context):
    imu = np.asarray(imu_window, dtype=np.float32)
    centered = imu - np.median(imu, axis=0, keepdims=True)
    acceleration = np.linalg.norm(centered[:, :3], axis=1)
    gyroscope = np.linalg.norm(centered[:, 3:6], axis=1)
    jerk = np.linalg.norm(np.diff(centered[:, :3], axis=0), axis=1)
    valid = np.asarray(vision_aux_window[:, 0] > 0.5)
    if valid.any():
        speed = np.linalg.norm(np.asarray(vision_window)[valid, 5:7], axis=1)
        visual_speed_rms = float(np.sqrt(np.mean(speed * speed)))
    else:
        visual_speed_rms = 0.0
    result = {
        "dynamic_acceleration_rms_g": float(np.sqrt(np.mean(acceleration ** 2))),
        "gyroscope_rms_deg_s": float(np.sqrt(np.mean(gyroscope ** 2))),
        "acceleration_jerk_rms_per_sample": float(np.sqrt(np.mean(jerk ** 2))),
        "visual_speed_rms_normalized_s": visual_speed_rms,
        "turn_context": bool(turn_context),
    }
    result["summary"] = (
        f"A {result['dynamic_acceleration_rms_g']:.2f}g  "
        f"G {result['gyroscope_rms_deg_s']:.1f}d/s  "
        f"V {visual_speed_rms:.2f}/s"
    )
    return result


def _compose_dashboard_frame(cv2, annotated, status, scale=2):
    frame = annotated.copy()
    height, width = frame.shape[:2]
    panel_height = 118
    panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
    panel[:] = (13, 16, 22)
    sensor_state = status.get("sensor_state", "STARTING")
    display_state = status.get("display_state")
    probabilities = status.get("probabilities", [0.0, 0.0, 0.0])
    if sensor_state != "OK":
        headline = sensor_state.replace("_", " ")
        color = (0, 170, 255) if sensor_state != "LINK_FAULT" else (20, 20, 255)
    else:
        headline = display_state or "CONFIRMING STATE"
        color = (20, 20, 255) if display_state == LABELS[2] else (50, 220, 90)
    cv2.putText(panel, "FitGuard CNN V8 PL / M01-M06", (7, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, headline, (7, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    cv2.putText(panel, f"S {probabilities[0]:.0%}  M {probabilities[1]:.0%}  X {probabilities[2]:.0%}", (7, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    motion = status.get("motion_indicators", {})
    cv2.putText(panel, motion.get("summary", "motion indicators warming up"), (7, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 230, 190), 1, cv2.LINE_AA)
    cv2.putText(panel, f"track Q {status.get('vision_quality', 0.0):.2f}  FPS {status.get('camera_fps', 0.0):.1f}  CNN {status.get('inference_ms', 0.0):.1f} ms", (7, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (190, 205, 230), 1, cv2.LINE_AA)
    event = status.get("event", {})
    event_text = f"MARKED EVENT {event.get('event_id', 'ACTIVE')}" if event.get("active") else "event recorder ready"
    cv2.putText(panel, f"{event_text} | backend {status.get('backend', 'numpy_ps')}", (7, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (60, 80, 255) if event.get("active") else (160, 170, 185), 1, cv2.LINE_AA)
    canvas = np.concatenate([panel, frame], axis=0)
    if int(scale) > 1:
        canvas = cv2.resize(canvas, (width * int(scale), (height + panel_height) * int(scale)), interpolation=cv2.INTER_NEAREST)
    return canvas


def run_live_gateway_cnn_pl(
    bitstream="fitguard_unified_cnn_pl.bit",
    model_dir="trained_models_motion_v2_cnn_v6_candidate",
    seconds=None,
    preview_fps=8,
    prediction_stride=0.32,
    smoothing_alpha=0.60,
    port="/dev/ttyUSB0",
    vision_valid_ratio_min=0.55,
    vision_quality_min=0.20,
    imu_stale_limit_s=0.60,
    log_dir="live_predictions_gateway_cnn_pl_runs",
    event_dir="fitguard_marked_events",
    web_port=None,
    backend="auto",
):
    import cv2
    from IPython.display import display
    import ipywidgets as widgets
    from fitguard_camera_runtime import FitGuardCamera
    from fitguard_gateway_imu import GatewayMPU6050

    # FitGuardCamera programs this same combined overlay. The PL model attaches
    # lazily on the first complete sensor window, so no second download occurs.
    model = RangeFusionCNNPL(model_dir, bitstream=bitstream, backend=backend, download=False)
    tracker = MotionPersonTracker()
    state_filter = OutputStateFilter()
    event_recorder = MarkedEventRecorder(event_dir)
    dashboard = FitGuardDashboardServer(port=web_port) if web_port else None
    dashboard_status = dashboard.start() if dashboard else {"running": False}
    if dashboard_status.get("running"):
        print(f"Independent Web dashboard: http://<PYNQ-IP>:{web_port}/")
    elif dashboard_status.get("error"):
        print(f"Web dashboard disabled: {dashboard_status['error']}")

    imu_rows, vision_rows = deque(maxlen=512), deque(maxlen=256)
    predictions = deque(maxlen=20000)
    sensor_events = deque(maxlen=512)
    frame_times = deque(maxlen=60)
    inference_times = deque(maxlen=512)
    prediction_count_total = 0
    stop_event = threading.Event()
    imu_health = {
        "last_sample_monotonic": 0.0,
        "error_since": None,
        "last_error": None,
    }
    last_prediction = np.ones(len(LABELS), dtype=np.float32) / len(LABELS)
    prediction_ready = False
    last_inference = 0.0
    last_sensor_state = None
    startup_ready = False
    vision_degraded_since = None
    vision_good_streak = 0
    vision_grace_s = 1.40
    display_state = None
    sensor_state = "WARMING_UP"
    motion_indicators = {}
    current_event = {"active": False, "event_id": None}
    backend_run_status = None
    run_started = time.monotonic()
    stop_reason = "manual_interrupt" if seconds is None or float(seconds) <= 0 else "duration_complete"
    imu = GatewayMPU6050(port=port)
    run_id, output, started_board_time = _reserve_run_record(log_dir)
    print(f"Independent V7 run record reserved: {output}")

    widget = widgets.Image(format="jpeg", layout=widgets.Layout(width="100%", height="auto", object_fit="contain"))
    fullscreen = widgets.HTML(value="""
    <button type="button" style="margin:4px 0 8px;padding:8px 18px;font-size:15px;cursor:pointer"
      onclick="(function(b){var box=b.parentElement.parentElement;if(document.fullscreenElement){document.exitFullscreen();}else{box.requestFullscreen();}})(this)">全屏查看（Esc退出）</button>
    """)
    display(widgets.VBox([fullscreen, widget], layout=widgets.Layout(width="100%", max_width="1500px")))
    next_preview = 0.0

    try:
        with FitGuardCamera(bitstream=bitstream) as camera:
            prime_status = model.prime()
            print(
                "CNN PL backend ready: "
                f"{prime_status.get('active')} "
                f"(accelerated={prime_status.get('pl_acceleration_active')})"
            )
            identity = imu.initialize()
            print(f"CNN V8 PL READY | gateway WHO_AM_I=0x{identity:02X}. Keep waist unit still...")
            imu.calibrate_gyro(100)

            def sample_imu():
                last_reopen_attempt = 0.0
                while not stop_event.is_set():
                    try:
                        imu_rows.append(asdict(imu.read_sample()))
                        imu_health["last_sample_monotonic"] = time.monotonic()
                        imu_health["error_since"] = None
                        imu_health["last_error"] = None
                    except Exception as exc:
                        failure_time = time.monotonic()
                        if imu_health["error_since"] is None:
                            imu_health["error_since"] = failure_time
                        imu_health["last_error"] = repr(exc)
                        # Brief BLE pauses retry on the current descriptor. Sustained
                        # failures also reopen CH340 so USB unplug/replug can recover.
                        if failure_time - last_reopen_attempt >= 1.0:
                            last_reopen_attempt = failure_time
                            try:
                                imu.close()
                            except Exception:
                                pass
                            try:
                                imu.initialize(timeout_s=1.0)
                            except Exception as reopen_exc:
                                imu_health["last_error"] = repr(reopen_exc)
                        time.sleep(0.05)

            thread = threading.Thread(target=sample_imu, daemon=True)
            thread.start()
            deadline = None if seconds is None or float(seconds) <= 0 else time.monotonic() + float(seconds)
            try:
                while (deadline is None or time.monotonic() < deadline) and not stop_event.is_set():
                    timestamp, raw = camera.capture()
                    frame_times.append(time.monotonic())
                    pl_mask, _ = camera.run_pl(raw)
                    record, _, annotated = tracker.update(raw, pl_mask, timestamp)
                    vision_rows.append(asdict(record))
                    now = time.monotonic()
                    imu_age = now - imu_health["last_sample_monotonic"] if imu_health["last_sample_monotonic"] else float("inf")
                    valid_ratio, quality, recent_count = _recent_vision_health(list(vision_rows), timestamp)
                    # WARMING_UP is a one-time startup phase.  Once the synchronized
                    # vision window has filled, a borderline 5/6-frame count must not
                    # send the UI back to WARMING_UP on every camera-frame fluctuation.
                    if recent_count >= 6:
                        startup_ready = True
                    current_vision_good = bool(getattr(record, "valid", False)) and (
                        float(getattr(record, "quality", 0.0)) >= vision_quality_min
                    )
                    vision_good_streak = vision_good_streak + 1 if current_vision_good else 0
                    vision_window_good = (
                        recent_count >= 6
                        and valid_ratio >= vision_valid_ratio_min
                        and quality >= vision_quality_min
                    )
                    # Asymmetric hysteresis: reacquire a returning person quickly,
                    # but require a sustained loss before declaring insufficient data.
                    if last_sensor_state == "DATA_INSUFFICIENT":
                        vision_unhealthy = vision_good_streak < 3
                    else:
                        vision_unhealthy = not (vision_window_good or vision_good_streak >= 2)
                    imu_error_age = (
                        now - imu_health["error_since"]
                        if imu_health["error_since"] is not None else 0.0
                    )
                    imu_fault = bool(
                        (imu_health["last_sample_monotonic"] and imu_age > imu_stale_limit_s)
                        or (
                            not imu_health["last_sample_monotonic"]
                            and imu_health["error_since"] is not None
                            and imu_error_age > imu_stale_limit_s
                        )
                    )
                    if imu_fault:
                        sensor_state, sensor_reason = "LINK_FAULT", "wireless IMU stale or disconnected; retrying"
                    elif not imu_health["last_sample_monotonic"] or not startup_ready:
                        sensor_state, sensor_reason = "WARMING_UP", "collecting initial synchronized window"
                    elif vision_unhealthy:
                        if vision_degraded_since is None:
                            vision_degraded_since = now
                        if now - vision_degraded_since >= vision_grace_s:
                            sensor_state, sensor_reason = "DATA_INSUFFICIENT", "person tracking unreliable"
                        else:
                            sensor_state = last_sensor_state if last_sensor_state in ("OK", "DATA_INSUFFICIENT") else "OK"
                            sensor_reason = "brief vision fluctuation within grace period"
                    else:
                        vision_degraded_since = None
                        sensor_state, sensor_reason = "OK", "sensors healthy"
                    if sensor_state != last_sensor_state:
                        sensor_events.append({"timestamp": timestamp, "state": sensor_state, "reason": sensor_reason})
                        print(f"SENSOR STATE: {sensor_state} - {sensor_reason}")
                        last_sensor_state = sensor_state
                    if sensor_state != "OK" and current_event.get("active"):
                        current_event = event_recorder.update({"timestamp": timestamp, "label": None, "sensor_state": sensor_state})

                    end_time = min(imu_rows[-1]["timestamp"] if imu_rows else 0.0, vision_rows[-1]["timestamp"])
                    if sensor_state == "OK" and end_time - last_inference >= prediction_stride:
                        imu_window = _resample(list(imu_rows), IMU_FIELDS, end_time, IMU_SAMPLES, WINDOW_SECONDS)
                        vision_window = _resample(list(vision_rows), VISION_FIELDS, end_time, VISION_SAMPLES, WINDOW_SECONDS)
                        aux_window = _resample(list(vision_rows), VISION_AUX_FIELDS, end_time, VISION_SAMPLES, WINDOW_SECONDS)
                        if imu_window is not None and vision_window is not None and aux_window is not None:
                            raw_probability, profile = model.predict_profiled(imu_window, vision_window, aux_window)
                            inference_times.append(profile["inference_ms"])
                            turn = bool(model.turn_guard and __import__("fitguard_motion_m01_m06_cnn_runtime").coherent_turn_present(vision_window, aux_window))
                            motion_indicators = _motion_indicators(imu_window, vision_window, aux_window, turn)
                            last_prediction = smoothing_alpha * raw_probability + (1.0 - smoothing_alpha) * last_prediction if prediction_ready else raw_probability
                            prediction_ready = True
                            filtered = state_filter.update(last_prediction, raw_marked_probability=float(raw_probability[2]))
                            display_state = filtered["label"]
                            prediction_record = {
                                "timestamp": float(end_time), "model": "cnn_v8_pl_v6_weights",
                                "label": display_state, "raw_label": LABELS[int(raw_probability.argmax())],
                                "confidence": float(last_prediction.max()), "probabilities": last_prediction.tolist(),
                                "raw_probabilities": raw_probability.tolist(), "sensor_state": sensor_state,
                                "vision_valid_ratio_recent": float(valid_ratio), "vision_quality_recent": float(quality),
                                "motion_indicators": motion_indicators, "state_filter": filtered,
                                "inference_ms": profile["inference_ms"], "backend": profile["backend"],
                            }
                            snapshot_ok, snapshot = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
                            current_event = event_recorder.update(prediction_record, snapshot.tobytes() if snapshot_ok else None)
                            prediction_record["event"] = dict(current_event)
                            predictions.append(prediction_record)
                            prediction_count_total += 1
                            last_inference = end_time

                    camera_fps = 0.0
                    if len(frame_times) >= 2 and frame_times[-1] > frame_times[0]:
                        camera_fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                    status = {
                        "run_id": run_id, "sensor_state": sensor_state, "display_state": display_state,
                        "probabilities": last_prediction.tolist(), "vision_quality": float(quality),
                        "vision_valid_ratio": float(valid_ratio), "camera_fps": float(camera_fps),
                        "inference_ms": float(inference_times[-1]) if inference_times else 0.0,
                        "motion_indicators": motion_indicators, "event": current_event,
                        "backend": model.backend_name,
                    }
                    if now >= next_preview:
                        canvas = _compose_dashboard_frame(cv2, annotated, status, scale=2)
                        ok, jpeg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 86])
                        if ok:
                            jpeg_bytes = jpeg.tobytes()
                            widget.value = jpeg_bytes
                            if dashboard:
                                dashboard.update(jpeg_bytes, status)
                        next_preview = now + 1.0 / max(preview_fps, 1)
            except KeyboardInterrupt:
                stop_reason = "manual_interrupt"
                print("Manual stop received; releasing resources safely...")
            finally:
                stop_event.set()
                thread.join(timeout=2.0)
    finally:
        # Release CMA/DMA buffers first; later logging or peripheral cleanup
        # must not be able to strand them and make the next notebook run hang.
        backend_run_status = model.backend_status()
        model.close()
        event_recorder.close(stop_reason)
        gateway_stats = imu.diagnostics()
        imu.close()
        if dashboard:
            dashboard.stop()

    result = {
        "run_id": run_id, "record_status": "fault" if imu_health["last_error"] else "completed",
        "started_board_time": started_board_time, "board_time_is_authoritative": False,
        "model": "cnn_v8_pl_v6_weights", "model_dir": str(Path(model_dir).resolve()),
        "backend": backend_run_status or model.backend_status(), "stop_reason": stop_reason,
        "run_duration_s": float(time.monotonic() - run_started),
        "prediction_count_total": prediction_count_total, "predictions": list(predictions),
        "sensor_events": list(sensor_events), "inference_ms": _percentiles(inference_times),
        "event_directory": str(Path(event_dir).resolve()),
        "dashboard": dashboard_status, "gateway": gateway_stats,
        "terminal_fault": imu_health["last_error"],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    Path("live_predictions_gateway_cnn_pl.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "predictions"}, ensure_ascii=False, indent=2))
    return result

