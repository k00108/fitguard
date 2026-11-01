#!/usr/bin/env python3
"""Requirement-by-requirement gate for the final independent CNN-PL package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    root = args.package_root.resolve()
    verify = root / "engineering" / "verification"
    unified = root / "deploy" / "FitGuard_Unified_Test"
    gateway = root / "deploy" / "FitGuard_Wireless_Gateway_Deploy"
    audit = load(verify / "completion_audit.json")
    hardware = load(verify / "hardware_contract_report.json")
    live = load(verify / "fitguard_cnn_pl_live_acceptance.json")

    deploy_files = [
        unified / "fitguard_unified_cnn_pl.bit",
        unified / "fitguard_unified_cnn_pl.hwh",
        unified / "fitguard_cnn_pl_backend.py",
        unified / "fitguard_motion_m01_m06_cnn_runtime_pl.py",
        unified / "fitguard_cnn_pl_selftest.py",
        unified / "fitguard_cnn_pl_golden.npz",
        unified / "FitGuard_CNN_PL_Selftest.ipynb",
        gateway / "fitguard_live_classifier_gateway_cnn_pl.py",
        gateway / "FitGuard_Wireless_Live_Demo_CNN_PL.ipynb",
    ]
    passed_reports = [
        "runtime_contract_report.json",
        "fallback_contract_report.json",
        "hardware_contract_report.json",
        "installer_contract_report.json",
        "mock_pynq_contract_report.json",
        "prime_contract_report.json",
        "fitguard_cnn_pl_board_selftest.json",
        "stable_preservation_report.json",
    ]
    backend = live.get("backend", {})
    pl = backend.get("pl", {})
    checks = {
        "nine_steps_complete": len(audit.get("steps", [])) == 9
        and all(step.get("status") == "complete" for step in audit["steps"])
        and audit.get("overall_status") == "complete",
        "nine_addon_files_present": all(path.is_file() and path.stat().st_size > 0 for path in deploy_files),
        "stable_bitstream_not_packaged": not (unified / "fitguard_unified.bit").exists(),
        "all_contract_reports_pass": all(load(verify / name).get("passed") is True for name in passed_reports),
        "live_backend_is_pl": backend.get("active") == "pl_cnn_conv1d",
        "live_acceleration_active": backend.get("pl_acceleration_active") is True,
        "live_dma_exercised": pl.get("dma") == "axi_dma_cnn_0" and int(pl.get("calls", 0)) >= 1,
        "live_has_no_fallback_reason": backend.get("fallback_reason") is None,
        "bit_hash_matches_contract": sha256(unified / "fitguard_unified_cnn_pl.bit")
        == hardware["details"]["bit_sha256"],
        "hwh_hash_matches_contract": sha256(unified / "fitguard_unified_cnn_pl.hwh")
        == hardware["details"]["hwh_sha256"],
    }
    report = {"package_root": str(root), "checks": checks, "passed": all(checks.values())}
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
