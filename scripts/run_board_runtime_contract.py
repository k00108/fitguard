"""Board-side import and interface contract for the final CNN-PL live path."""

import inspect
import json
from pathlib import Path
import sys


root = Path("/home/xilinx/jupyter_notebooks")
unified = root / "FitGuard_Unified_Test"
gateway = root / "FitGuard_Wireless_Gateway_Deploy"
sys.path[:0] = [str(gateway), str(unified)]

from fitguard_live_classifier_gateway_cnn_pl import run_live_gateway_cnn_pl


parameters = inspect.signature(run_live_gateway_cnn_pl).parameters
expected = {
    "bitstream", "model_dir", "seconds", "preview_fps", "prediction_stride",
    "smoothing_alpha", "port", "vision_valid_ratio_min", "vision_quality_min",
    "imu_stale_limit_s", "log_dir", "event_dir", "web_port", "backend",
}
checks = {
    "live_entrypoint_imports": callable(run_live_gateway_cnn_pl),
    "interface_is_frozen": set(parameters) == expected,
    "supports_auto_pl_backend": parameters["backend"].default == "auto",
    "supports_manual_unlimited_run": parameters["seconds"].default is None,
    "supports_sensor_fault_threshold": "imu_stale_limit_s" in parameters,
    "supports_state_smoothing": "smoothing_alpha" in parameters,
}
result = {"checks": checks, "passed": all(checks.values())}
report = unified / "fitguard_cnn_pl_board_runtime_contract.json"
report.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
if not result["passed"]:
    raise RuntimeError("CNN-PL runtime contract failed")
