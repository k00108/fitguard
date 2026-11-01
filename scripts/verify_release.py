#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REQUIRED = [
    "README.md",
    "LICENSE",
    "hardware/hls/fitguard_cnn_conv.cpp",
    "hardware/hls/fitguard_cnn_conv_tb.cpp",
    "hardware/hls/run_hls.tcl",
    "hardware/vivado/rebuild_final_project.tcl",
    "hardware/bitstream/fitguard_unified_cnn_pl.bit",
    "hardware/bitstream/fitguard_unified_cnn_pl.hwh",
    "software/pynq/FitGuard_Unified_Test/fitguard_cnn_pl_backend.py",
    "software/pynq/FitGuard_Unified_Test/FitGuard_CNN_PL_Selftest.ipynb",
    "software/pynq/FitGuard_Wireless_Gateway_Deploy/fitguard_live_classifier_gateway_cnn_pl.py",
    "software/pynq/FitGuard_Wireless_Gateway_Deploy/FitGuard_Wireless_Live_Demo_CNN_PL.ipynb",
    "models/cnn_v6/range_fusion_v2_weights.npz",
    "models/random_forest_baseline/fitguard_motion_v2_rf_base_numpy.npz",
    "verification/board_selftest/fitguard_cnn_pl_board_selftest.json",
    "verification/live/live_predictions_gateway_cnn_pl.json",
    "verification/live/live_predictions_gateway_long_run.json",
    "skill/SKILL.md",
    "skill/README.md",
]

FORBIDDEN_PARTS = {
    "__pycache__",
    ".ipynb_checkpoints",
    "dataset_motion_v2",
    "fitguard_marked_events",
}
FORBIDDEN_SUFFIXES = {".crdownload", ".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the FitGuard release tree")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    errors: list[str] = []
    for rel in REQUIRED:
        if not (root / rel).is_file():
            errors.append(f"missing required file: {rel}")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        rel = path.relative_to(root)
        if any(part in FORBIDDEN_PARTS for part in rel.parts):
            errors.append(f"forbidden path: {rel.as_posix()}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden generated file: {rel.as_posix()}")

    selftest_path = root / "verification/board_selftest/fitguard_cnn_pl_board_selftest.json"
    if selftest_path.is_file():
        selftest = json.loads(selftest_path.read_text(encoding="utf-8"))
        if not selftest.get("passed"):
            errors.append("board self-test is not marked passed")
        if int(selftest.get("cases", 0)) != 18:
            errors.append("board self-test does not contain 18 cases")
        if float(selftest.get("max_abs_error", 1.0)) > float(selftest.get("threshold", 0.0)):
            errors.append("board self-test error exceeds threshold")

    live_path = root / "verification/live/live_predictions_gateway_cnn_pl.json"
    if live_path.is_file():
        live = json.loads(live_path.read_text(encoding="utf-8"))
        backend = live.get("backend", {})
        if backend.get("active") != "pl_cnn_conv1d":
            errors.append("live evidence did not activate pl_cnn_conv1d")
        if backend.get("pl_acceleration_active") is not True:
            errors.append("live evidence does not mark PL acceleration active")
        if int(backend.get("pl", {}).get("calls", 0)) <= 0:
            errors.append("live evidence has no PL calls")

    long_run_path = root / "verification/live/live_predictions_gateway_long_run.json"
    if long_run_path.is_file():
        long_run = json.loads(long_run_path.read_text(encoding="utf-8"))
        gateway = long_run.get("gateway", {})
        predictions = long_run.get("predictions", [])
        final_labels = [item.get("label") for item in predictions if item.get("label")]
        expected_labels = {"Stable", "Mild Abnormality", "Marked Abnormality"}
        if long_run.get("run_id") != "cnn_v6_run_0011":
            errors.append("long-run evidence has an unexpected run_id")
        if abs(float(long_run.get("run_duration_s", 0.0)) - 817.591307483) > 1e-6:
            errors.append("long-run duration does not match the report evidence")
        if int(long_run.get("prediction_count_total", 0)) != 737 or len(predictions) != 737:
            errors.append("long-run prediction count does not match the report evidence")
        if len(final_labels) != 736 or set(final_labels) != expected_labels:
            errors.append("long-run evidence does not contain the expected final three-class output")
        if int(gateway.get("samples", 0)) != 40400:
            errors.append("long-run IMU sample count does not match the report evidence")
        if abs(float(gateway.get("receive_ratio", 0.0)) - 0.9988132911392406) > 1e-12:
            errors.append("long-run receive ratio does not match the report evidence")
        if int(gateway.get("bad_crc", -1)) != 1 or int(gateway.get("lost_samples", -1)) != 48:
            errors.append("long-run CRC/lost-sample counts do not match the report evidence")
        if int(gateway.get("out_of_order_samples", -1)) != 0 or int(gateway.get("queue_drops", -1)) != 0:
            errors.append("long-run ordering/queue counters do not match the report evidence")
        if long_run.get("stop_reason") != "manual_interrupt":
            errors.append("long-run evidence was not stopped manually")

    if args.write_manifest and not errors:
        manifest_path = root / "SHA256SUMS.txt"
        manifest_files = [path for path in files if path != manifest_path]
        lines = [f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in manifest_files]
        manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = {
        "root": str(root),
        "file_count": len(files),
        "errors": errors,
        "passed": not errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
