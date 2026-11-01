"""FitGuard startup self-check that is safe to run without camera or IMU."""

import glob
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reserve_report(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sequence = 1
    existing = []
    for path in root.glob("preflight_*.json"):
        try:
            existing.append(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            pass
    if existing:
        sequence = max(existing) + 1
    return root / f"preflight_{sequence:04d}.json"


def run_preflight(deploy_dir=None, base_dir=None, output_dir="preflight_reports"):
    deploy = Path(deploy_dir or Path(__file__).resolve().parent).resolve()
    base = Path(base_dir or (deploy.parent / "FitGuard_Unified_Test")).resolve()
    for path in (deploy, base):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    bitstream = base / "fitguard_unified.bit"
    hwh = base / "fitguard_unified.hwh"
    model_dir = base / "trained_models_motion_v2_cnn_v6_candidate"
    required = [
        bitstream,
        hwh,
        model_dir / "range_fusion_v2_weights.npz",
        model_dir / "range_fusion_v2_normalization.npz",
        model_dir / "deployment_policy.json",
        model_dir / "cnn_v6_golden_batch.npz",
        base / "fitguard_motion_m01_m06_cnn_runtime.py",
        base / "fitguard_motion_m01_m06_cnn_runtime_v7.py",
        deploy / "motion_tracker_v2.py",
        deploy / "fitguard_live_classifier_gateway_cnn_v7.py",
    ]
    files = {
        str(path): {
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else None,
        }
        for path in required
    }
    file_check = all(item["exists"] and item["size"] > 0 for item in files.values())

    golden = {"passed": False, "error": None}
    try:
        from fitguard_motion_m01_m06_cnn_runtime_v7 import RangeFusionCNNV7

        model = RangeFusionCNNV7(model_dir)
        with np.load(model_dir / "cnn_v6_golden_batch.npz", allow_pickle=False) as data:
            actual = np.asarray(
                [
                    model.predict(imu, vision, aux)
                    for imu, vision, aux in zip(
                        data["imu"], data["vision"], data["vision_aux"]
                    )
                ],
                dtype=np.float32,
            )
            expected = data["expected_probability"].astype(np.float32)
            error = float(np.max(np.abs(actual - expected)))
            labels_match = bool(
                np.array_equal(np.argmax(actual, axis=1), np.argmax(expected, axis=1))
            )
        golden = {
            "passed": labels_match and error <= 2e-6,
            "max_abs_probability_error": error,
            "class_agreement": labels_match,
            "windows": int(len(actual)),
            "backend": model.backend_status(),
        }
    except Exception as exc:
        golden = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}

    camera_nodes = sorted(glob.glob("/dev/video*"))
    gateway_nodes = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
    report = {
        "report_format_version": 1,
        "board_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "board_time_is_authoritative": False,
        "deploy_dir": str(deploy),
        "base_dir": str(base),
        "files": files,
        "file_check_passed": file_check,
        "bitstream_sha256": _sha256(bitstream) if bitstream.exists() else None,
        "model_golden_check": golden,
        "camera_nodes": camera_nodes,
        "gateway_nodes": gateway_nodes,
        "offline_ready": bool(file_check and golden.get("passed")),
        "live_ready": bool(
            file_check and golden.get("passed") and camera_nodes and gateway_nodes
        ),
        "live_blockers": [
            message
            for condition, message in (
                (not camera_nodes, "USB camera is not connected"),
                (not gateway_nodes, "ESP32 USB gateway is not connected"),
            )
            if condition
        ],
    }
    output = _reserve_report(deploy / output_dir)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["saved_path"] = str(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    run_preflight()
