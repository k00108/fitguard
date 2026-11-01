#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED = [
    "README.md",
    "LICENSE",
    "hardware/hls/fitguard_cnn_conv.cpp",
    "hardware/vivado/rebuild_final_project.tcl",
    "hardware/bitstream/fitguard_unified_cnn_pl.bit",
    "hardware/bitstream/fitguard_unified_cnn_pl.hwh",
    "software/pynq/FitGuard_Unified_Test/fitguard_cnn_pl_backend.py",
    "verification/board_selftest/fitguard_cnn_pl_board_selftest.json",
    "verification/live/live_predictions_gateway_cnn_pl.json",
    "verification/live/live_predictions_gateway_long_run.json",
]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_artifacts.py <release-root>")
        return 2
    root = Path(sys.argv[1]).resolve()
    errors = [f"missing: {rel}" for rel in REQUIRED if not (root / rel).is_file()]
    selftest_path = root / "verification/board_selftest/fitguard_cnn_pl_board_selftest.json"
    if selftest_path.is_file():
        data = json.loads(selftest_path.read_text(encoding="utf-8"))
        if not data.get("passed"):
            errors.append("board self-test is not passed")
    live_path = root / "verification/live/live_predictions_gateway_cnn_pl.json"
    if live_path.is_file():
        data = json.loads(live_path.read_text(encoding="utf-8"))
        if data.get("backend", {}).get("active") != "pl_cnn_conv1d":
            errors.append("live backend is not pl_cnn_conv1d")
        if data.get("backend", {}).get("pl_acceleration_active") is not True:
            errors.append("live evidence does not mark PL acceleration active")
        if int(data.get("backend", {}).get("pl", {}).get("calls", 0)) <= 0:
            errors.append("live evidence has no PL calls")
    long_run_path = root / "verification/live/live_predictions_gateway_long_run.json"
    if long_run_path.is_file():
        data = json.loads(long_run_path.read_text(encoding="utf-8"))
        gateway = data.get("gateway", {})
        labels = {item.get("label") for item in data.get("predictions", []) if item.get("label")}
        if data.get("run_id") != "cnn_v6_run_0011":
            errors.append("long-run evidence has an unexpected run_id")
        if int(data.get("prediction_count_total", 0)) != 737:
            errors.append("long-run prediction count is not 737")
        if int(gateway.get("samples", 0)) != 40400:
            errors.append("long-run IMU sample count is not 40400")
        if labels != {"Stable", "Mild Abnormality", "Marked Abnormality"}:
            errors.append("long-run evidence does not contain all three final classes")
    print(json.dumps({"passed": not errors, "errors": errors}, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
