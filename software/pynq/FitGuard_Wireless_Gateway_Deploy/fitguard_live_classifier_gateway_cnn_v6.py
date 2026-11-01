"""FitGuard live demo for the M01-M06 CNN V6 candidate.

This is intentionally a new entry point.  It does not replace the legacy
gateway demo, collector, random-forest candidate, or motion tracker.
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
from fitguard_motion_m01_m06_cnn_runtime import RangeFusionCNN, coherent_turn_present
from motion_tracker_v2 import MotionPersonTracker


VISION_AUX_FIELDS = ("valid", "quality", "motion_score")


class OutputStateFilter:
    """Debounce class changes without delaying a high-confidence marked event."""

    def __init__(
        self,
        high_marked_threshold=0.88,
        initial_confirmations=2,
        escalation_confirmations=2,
        recovery_confirmations=3,
        marked_recovery_confirmations=4,
    ):
        self.high_marked_threshold = float(high_marked_threshold)
        self.initial_confirmations = int(initial_confirmations)
        self.escalation_confirmations = int(escalation_confirmations)
        self.recovery_confirmations = int(recovery_confirmations)
        self.marked_recovery_confirmations = int(marked_recovery_confirmations)
        self.current_index = None
        self.pending_index = None
        self.pending_count = 0

    def update(self, probability, raw_marked_probability=None):
        probability = np.asarray(probability, dtype=np.float32)
        candidate = int(np.argmax(probability))
        raw_marked = float(
            probability[2] if raw_marked_probability is None else raw_marked_probability
        )
        changed = False
        fast_marked = candidate == 2 and raw_marked >= self.high_marked_threshold

        if fast_marked:
            changed = self.current_index != 2
            self.current_index = 2
            self.pending_index = None
            self.pending_count = 0
        elif self.current_index == candidate:
            self.pending_index = None
            self.pending_count = 0
        else:
            if self.pending_index == candidate:
                self.pending_count += 1
            else:
                self.pending_index = candidate
                self.pending_count = 1

            if self.current_index is None:
                required = self.initial_confirmations
            elif self.current_index == 2 and candidate != 2:
                required = self.marked_recovery_confirmations
            elif candidate > self.current_index:
                required = self.escalation_confirmations
            else:
                required = self.recovery_confirmations
            if self.pending_count >= required:
                changed = self.current_index != candidate
                self.current_index = candidate
                self.pending_index = None
                self.pending_count = 0

        return {
            "label": LABELS[self.current_index] if self.current_index is not None else None,
            "label_index": self.current_index,
            "candidate_label": LABELS[candidate],
            "pending_label": LABELS[self.pending_index] if self.pending_index is not None else None,
            "pending_count": self.pending_count,
            "changed": changed,
            "fast_marked": fast_marked,
        }


def _recent_vision_health(rows, current_timestamp, horizon_s=1.0):
    recent = [row for row in rows if current_timestamp - row["timestamp"] <= horizon_s]
    if len(recent) < 6:
        return 0.0, 0.0, len(recent)
    valid = [row for row in recent if int(row.get("valid", 0))]
    valid_ratio = len(valid) / len(recent)
    mean_quality = float(np.mean([row.get("quality", 0.0) for row in valid])) if valid else 0.0
    return float(valid_ratio), mean_quality, len(recent)


def _reserve_run_record(log_dir):
    """Reserve a monotonically numbered record without relying on board time."""
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    sequence = 0
    for path in root.glob("cnn_v6_run_*.json"):
        try:
            sequence = max(sequence, int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    sequence += 1
    run_id = f"cnn_v6_run_{sequence:04d}"
    output = root / f"{run_id}.json"
    placeholder = {
        "run_id": run_id,
        "record_status": "running",
        "started_board_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "Board clock may be inaccurate; run_id sequence is authoritative.",
    }
    output.write_text(json.dumps(placeholder, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_id, output, placeholder["started_board_time"]


def run_live_gateway_cnn_v6(
    bitstream="fitguard_unified.bit",
    model_dir="trained_models_motion_v2_cnn_v6_candidate",
    seconds=60,
    min_area=200,
    preview_fps=8,
    prediction_stride=0.32,
    smoothing_alpha=0.60,
    port="/dev/ttyUSB0",
    vision_valid_ratio_min=0.55,
    vision_quality_min=0.20,
    imu_stale_limit_s=0.60,
    log_dir="live_predictions_gateway_cnn_v6_runs",
):
    """Run the isolated CNN V6 live path using the best V2 motion tracker."""
    import cv2
    from IPython.display import display
    import ipywidgets as widgets
    from fitguard_camera_runtime import FitGuardCamera
    from fitguard_gateway_imu import GatewayMPU6050

    model = RangeFusionCNN(model_dir)
    tracker = MotionPersonTracker()
    imu_rows, vision_rows = deque(maxlen=512), deque(maxlen=256)
    # Keep memory bounded during an unlimited exhibition run.  At the default
    # stride this retains roughly the latest 100 minutes of predictions.
    predictions = deque(maxlen=20000)
    prediction_count_total = 0
    stop_event, imu_error = threading.Event(), []
    imu_health = {"last_sample_monotonic": 0.0}
    state_filter = OutputStateFilter()
    sensor_events = deque(maxlen=512)
    last_sensor_state = None
    last_prediction = np.ones(len(LABELS), dtype=np.float32) / len(LABELS)
    prediction_ready = False
    last_inference = 0.0
    last_turn_context = False
    display_state = None
    sensor_state = "WARMING_UP"
    sensor_reason = "collecting a full synchronized window"
    vision_valid_ratio_recent = 0.0
    vision_quality_recent = 0.0
    imu_age_s = float("inf")
    link_fault_started = None
    run_started = time.monotonic()
    stop_reason = (
        "manual_interrupt" if seconds is None or float(seconds) <= 0 else "duration_complete"
    )
    run_id, output, started_board_time = _reserve_run_record(log_dir)
    print(f"Independent run record reserved: {output}")

    widget = widgets.Image(
        format="jpeg",
        layout=widgets.Layout(width="100%", height="auto", object_fit="contain"),
    )
    fullscreen_control = widgets.HTML(
        value="""
        <button type="button" style="margin:4px 0 8px 0;padding:7px 16px;font-size:14px;cursor:pointer;"
          onclick="(function(button){
            var box=button.closest('.widget-box') || button.parentElement.parentElement.parentElement;
            if(!box){return;}
            if(document.fullscreenElement){document.exitFullscreen();return;}
            box.dataset.fitguardOldWidth=box.style.width || '';
            box.dataset.fitguardOldHeight=box.style.height || '';
            box.style.width='100vw'; box.style.height='100vh'; box.style.background='#000';
            var image=box.querySelector('img');
            if(image){image.style.width='100%';image.style.height='calc(100vh - 48px)';image.style.objectFit='contain';}
            box.requestFullscreen().catch(function(error){alert('浏览器未允许全屏：'+error.message);});
            var restore=function(){
              if(!document.fullscreenElement){
                box.style.width=box.dataset.fitguardOldWidth; box.style.height=box.dataset.fitguardOldHeight;
                box.style.background='';
                if(image){image.style.width='100%';image.style.height='auto';image.style.objectFit='contain';}
                document.removeEventListener('fullscreenchange',restore);
              }
            };
            document.addEventListener('fullscreenchange',restore);
          })(this)">全屏查看（Esc退出）</button>
        """
    )
    preview_box = widgets.VBox(
        [fullscreen_control, widget],
        layout=widgets.Layout(width="100%", max_width="1400px"),
    )
    display(preview_box)
    next_preview = 0.0

    imu = GatewayMPU6050(port=port)
    with FitGuardCamera(bitstream=bitstream) as camera:
        try:
            identity = imu.initialize()
            print(
                f"CNN V6 ACTIVE | ESP32 gateway connected "
                f"(WHO_AM_I=0x{identity:02X}). Keep the waist unit still..."
            )
            if seconds is None or float(seconds) <= 0:
                print("Unlimited demo mode: click the stop square or use Kernel > Interrupt to finish safely.")
            imu.calibrate_gyro(100)

            def sample_imu():
                try:
                    while not stop_event.is_set():
                        imu_rows.append(asdict(imu.read_sample()))
                        imu_health["last_sample_monotonic"] = time.monotonic()
                except Exception as exc:
                    imu_error.append(exc)

            thread = threading.Thread(target=sample_imu, daemon=True)
            thread.start()
            unlimited = seconds is None or float(seconds) <= 0
            deadline = None if unlimited else time.monotonic() + float(seconds)
            try:
                while (deadline is None or time.monotonic() < deadline) and not stop_event.is_set():
                    timestamp, raw = camera.capture()
                    pl_mask, _ = camera.run_pl(raw)
                    record, _, annotated = tracker.update(raw, pl_mask, timestamp)
                    vision_rows.append(asdict(record))

                    now = time.monotonic()
                    imu_age_s = (
                        now - imu_health["last_sample_monotonic"]
                        if imu_health["last_sample_monotonic"] > 0
                        else float("inf")
                    )
                    vision_valid_ratio_recent, vision_quality_recent, recent_count = (
                        _recent_vision_health(list(vision_rows), timestamp)
                    )

                    if imu_error or (
                        imu_health["last_sample_monotonic"] > 0
                        and imu_age_s > float(imu_stale_limit_s)
                    ):
                        sensor_state = "LINK_FAULT"
                        sensor_reason = "wireless IMU data are stale or disconnected"
                        if link_fault_started is None:
                            link_fault_started = now
                    elif imu_health["last_sample_monotonic"] <= 0 or recent_count < 6:
                        sensor_state = "WARMING_UP"
                        sensor_reason = "collecting a full synchronized window"
                    elif (
                        vision_valid_ratio_recent < float(vision_valid_ratio_min)
                        or vision_quality_recent < float(vision_quality_min)
                    ):
                        sensor_state = "DATA_INSUFFICIENT"
                        sensor_reason = "person tracking is missing or unreliable"
                        link_fault_started = None
                    else:
                        sensor_state = "OK"
                        sensor_reason = "sensors healthy"
                        link_fault_started = None

                    if sensor_state != last_sensor_state:
                        sensor_events.append(
                            {
                                "timestamp": timestamp,
                                "state": sensor_state,
                                "reason": sensor_reason,
                                "imu_age_s": None if not np.isfinite(imu_age_s) else float(imu_age_s),
                                "vision_valid_ratio_recent": vision_valid_ratio_recent,
                                "vision_quality_recent": vision_quality_recent,
                            }
                        )
                        print(f"SENSOR STATE: {sensor_state} - {sensor_reason}")
                        last_sensor_state = sensor_state

                    end_time = min(
                        imu_rows[-1]["timestamp"] if imu_rows else 0.0,
                        vision_rows[-1]["timestamp"],
                    )
                    if sensor_state == "OK" and end_time - last_inference >= prediction_stride:
                        imu_window = _resample(
                            list(imu_rows), IMU_FIELDS, end_time, IMU_SAMPLES, WINDOW_SECONDS
                        )
                        vision_window = _resample(
                            list(vision_rows), VISION_FIELDS, end_time, VISION_SAMPLES, WINDOW_SECONDS
                        )
                        vision_aux_window = _resample(
                            list(vision_rows), VISION_AUX_FIELDS, end_time, VISION_SAMPLES, WINDOW_SECONDS
                        )
                        if (
                            imu_window is not None
                            and vision_window is not None
                            and vision_aux_window is not None
                        ):
                            started = time.monotonic()
                            raw_probability = model.predict(
                                imu_window, vision_window, vision_aux_window
                            )
                            last_turn_context = coherent_turn_present(
                                vision_window, vision_aux_window
                            )
                            if prediction_ready:
                                last_prediction = (
                                    smoothing_alpha * raw_probability
                                    + (1.0 - smoothing_alpha) * last_prediction
                                )
                            else:
                                last_prediction = raw_probability
                                prediction_ready = True
                            filtered = state_filter.update(
                                last_prediction,
                                raw_marked_probability=float(raw_probability[2]),
                            )
                            display_state = filtered["label"]
                            predictions.append(
                                {
                                    "timestamp": end_time,
                                    "model": "cnn_v6_no_excursion_m01_m06",
                                    "raw_label": LABELS[int(raw_probability.argmax())],
                                    "smoothed_label": LABELS[int(last_prediction.argmax())],
                                    "label": display_state,
                                    "confidence": float(last_prediction.max()),
                                    "probabilities": last_prediction.tolist(),
                                    "raw_probabilities": raw_probability.tolist(),
                                    "turn_context": bool(last_turn_context),
                                    "sensor_state": sensor_state,
                                    "imu_age_s": float(imu_age_s),
                                    "vision_valid_ratio_recent": vision_valid_ratio_recent,
                                    "vision_quality_recent": vision_quality_recent,
                                    "state_filter": filtered,
                                    "inference_ms": (time.monotonic() - started) * 1000.0,
                                }
                            )
                            prediction_count_total += 1
                            last_inference = end_time

                    if now >= next_preview:
                        canvas = annotated.copy()
                        label_index = int(last_prediction.argmax())
                        cv2.rectangle(canvas, (0, 0), (390, 110), (0, 0, 0), -1)
                        cv2.putText(
                            canvas, "CNN V6 ACTIVE / M01-M06", (8, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 255), 1,
                        )
                        if sensor_state == "OK" and display_state is not None:
                            cv2.putText(
                                canvas, display_state, (8, 43),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2,
                            )
                            cv2.putText(
                                canvas, f"confidence {last_prediction[label_index]:.1%}", (8, 66),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.49, (255, 255, 255), 1,
                            )
                            turn_text = "  TURN" if last_turn_context else ""
                            cv2.putText(
                                canvas,
                                f"S {last_prediction[0]:.0%} M {last_prediction[1]:.0%} X {last_prediction[2]:.0%}{turn_text}",
                                (8, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (180, 255, 180), 1,
                            )
                            cv2.putText(
                                canvas, "debounced output", (8, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 210, 160), 1,
                            )
                        else:
                            confirming = sensor_state == "OK" and display_state is None
                            warning = "CONFIRMING STATE" if confirming else {
                                "WARMING_UP": "WARMING UP",
                                "DATA_INSUFFICIENT": "DATA INSUFFICIENT",
                                "LINK_FAULT": "LINK FAULT",
                            }.get(sensor_state, sensor_state)
                            warning_color = (0, 80, 255) if sensor_state == "LINK_FAULT" else (0, 220, 255)
                            cv2.putText(
                                canvas, warning, (8, 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.63, warning_color, 2,
                            )
                            detail = "waiting for stable CNN decision" if confirming else (
                                "IMU disconnected/stale" if sensor_state == "LINK_FAULT" else (
                                "person not reliably tracked" if sensor_state == "DATA_INSUFFICIENT"
                                else "waiting for synchronized data"
                                )
                            )
                            cv2.putText(
                                canvas, detail, (8, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                            )
                            cv2.putText(
                                canvas,
                                f"vision valid {vision_valid_ratio_recent:.0%}  Q {vision_quality_recent:.2f}",
                                (8, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 255, 180), 1,
                            )
                        ok, jpeg = cv2.imencode(
                            ".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 84]
                        )
                        if ok:
                            widget.value = jpeg.tobytes()
                        next_preview = now + 1.0 / max(preview_fps, 1)
                    if link_fault_started is not None and now - link_fault_started >= 1.5:
                        print("Stopping after displaying LINK FAULT for 1.5 seconds.")
                        stop_reason = "link_fault"
                        break
            except KeyboardInterrupt:
                stop_reason = "manual_interrupt"
                print("Manual stop received; releasing camera and IMU safely...")
            finally:
                stop_event.set()
                thread.join(timeout=2.0)
        finally:
            gateway_stats = imu.diagnostics()
            imu.close()

    result = {
        "run_id": run_id,
        "record_status": "fault" if stop_reason == "link_fault" else "completed",
        "started_board_time": started_board_time,
        "ended_board_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "board_time_is_authoritative": False,
        "model": "cnn_v6_no_excursion_m01_m06",
        "model_dir": str(Path(model_dir).resolve()),
        "stop_reason": stop_reason,
        "run_duration_s": time.monotonic() - run_started,
        "prediction_count_total": prediction_count_total,
        "prediction_buffer_limit": predictions.maxlen,
        "predictions": list(predictions),
        "sensor_events": list(sensor_events),
        "terminal_fault": str(imu_error[0]) if imu_error else None,
        "state_filter": {
            "initial_confirmations": state_filter.initial_confirmations,
            "escalation_confirmations": state_filter.escalation_confirmations,
            "recovery_confirmations": state_filter.recovery_confirmations,
            "marked_recovery_confirmations": state_filter.marked_recovery_confirmations,
            "high_marked_threshold": state_filter.high_marked_threshold,
        },
        "gateway": gateway_stats,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_output = Path("live_predictions_gateway_cnn_v6.json")
    latest_output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(gateway_stats, ensure_ascii=False, indent=2))
    if imu_error:
        print(f"LINK FAULT recorded: {imu_error[0]}")
    print(
        f"CNN V6 completed ({stop_reason}); saved the latest {len(predictions)} of "
        f"{prediction_count_total} predictions to independent record {output}"
    )
    return result

