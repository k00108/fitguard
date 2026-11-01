from pathlib import Path
import json
import sys
import warnings

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "board_snapshot_20261012" / "FitGuard_Unified_Test"
MODEL = BASE / "trained_models_motion_v2_cnn_v6_candidate"
BIT = ROOT / "hardware" / "bitstream" / "fitguard_unified_cnn_pl.bit"
sys.path[:0] = [str(HERE), str(BASE)]

from fitguard_motion_m01_m06_cnn_runtime_v7 import RangeFusionCNNV7
from fitguard_motion_m01_m06_cnn_runtime_pl import RangeFusionCNNPL


raw = np.load(MODEL / "cnn_v6_golden_batch.npz", allow_pickle=False)
reference = RangeFusionCNNV7(MODEL)
auto = RangeFusionCNNPL(MODEL, bitstream=BIT, backend="auto", download=False)
max_error = 0.0
class_matches = 0
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    for imu, vision, aux in zip(raw["imu"], raw["vision"], raw["vision_aux"]):
        expected = reference.predict(imu, vision, aux)
        actual = auto.predict(imu, vision, aux)
        max_error = max(max_error, float(np.max(np.abs(expected - actual))))
        class_matches += int(np.argmax(expected) == np.argmax(actual))

strict_failed = False
try:
    strict = RangeFusionCNNPL(MODEL, bitstream=BIT, backend="pl", download=False)
    strict.predict(raw["imu"][0], raw["vision"][0], raw["vision_aux"][0])
except RuntimeError:
    strict_failed = True

status = auto.backend_status()
report = {
    "cases": int(len(raw["imu"])),
    "class_matches": int(class_matches),
    "max_probability_error": max_error,
    "auto_backend_after_unavailable_pl": status["active"],
    "fallback_reason_recorded": bool(status["fallback_reason"]),
    "strict_pl_rejects_unavailable_backend": strict_failed,
    "passed": bool(
        class_matches == len(raw["imu"])
        and max_error == 0.0
        and status["active"] == "numpy_ps_fallback"
        and status["fallback_reason"]
        and strict_failed
    ),
}
(HERE / "fallback_contract_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise SystemExit(2)
