from __future__ import annotations

import json
import os
import sys
from pathlib import Path


BASE = Path("/home/xilinx/jupyter_notebooks")
UNIFIED = BASE / "FitGuard_Unified_Test"
DEPLOY = BASE / "FitGuard_Wireless_Gateway_Deploy"
for path in (str(UNIFIED), str(DEPLOY)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(str(UNIFIED))

from fitguard_live_classifier_gateway_cnn_pl import run_live_gateway_cnn_pl


result = run_live_gateway_cnn_pl(
    bitstream=str(UNIFIED / "fitguard_unified_cnn_pl.bit"),
    model_dir=str(UNIFIED / "trained_models_motion_v2_cnn_v6_candidate"),
    seconds=5,
    preview_fps=2,
    port="/dev/ttyUSB0",
    web_port=None,
    backend="pl",
)

evidence = {
    "run_id": result.get("run_id"),
    "record_status": result.get("record_status"),
    "stop_reason": result.get("stop_reason"),
    "prediction_count_total": result.get("prediction_count_total"),
    "backend": result.get("backend"),
    "inference_ms": result.get("inference_ms"),
    "sensor_events": result.get("sensor_events"),
    "terminal_fault": result.get("terminal_fault"),
}
print("FITGUARD_PL_LIVE_ACCEPTANCE=" + json.dumps(evidence, ensure_ascii=False))
