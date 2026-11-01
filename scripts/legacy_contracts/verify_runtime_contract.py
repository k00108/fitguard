from pathlib import Path
import json
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "board_snapshot_20261012" / "FitGuard_Unified_Test"
MODEL = BASE / "trained_models_motion_v2_cnn_v6_candidate"
sys.path[:0] = [str(HERE), str(BASE)]

from fitguard_motion_m01_m06_cnn_runtime_v7 import RangeFusionCNNV7
from fitguard_motion_m01_m06_cnn_runtime_pl import RangeFusionCNNPL


golden_raw = np.load(MODEL / "cnn_v6_golden_batch.npz", allow_pickle=False)
golden_pl = np.load(HERE / "generated" / "fitguard_cnn_pl_golden.npz", allow_pickle=False)
reference = RangeFusionCNNV7(MODEL)
runtime = RangeFusionCNNPL(MODEL, backend="numpy")
max_error = 0.0
class_matches = 0
for imu, vision, aux, gap in zip(
    golden_raw["imu"], golden_raw["vision"], golden_raw["vision_aux"], golden_pl["outputs"]
):
    expected = reference.predict(imu, vision, aux)
    _, _, ranges, turn = runtime._prepare(imu, vision, aux)
    actual = runtime._finish_from_gap(gap, ranges, turn)
    max_error = max(max_error, float(np.max(np.abs(expected - actual))))
    class_matches += int(np.argmax(expected) == np.argmax(actual))
report = {
    "cases": int(len(golden_raw["imu"])),
    "class_matches": int(class_matches),
    "max_probability_error": max_error,
    "passed": bool(class_matches == len(golden_raw["imu"]) and max_error <= 0.002),
}
(HERE / "runtime_contract_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
if not report["passed"]:
    raise SystemExit(2)
