#!/usr/bin/env python3
"""Verify that the add-on installer is complete, atomic at preflight, and non-overwriting."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile


def digest_tree(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    package_root = args.package_root.resolve()
    spec = importlib.util.spec_from_file_location("fitguard_installer", package_root / "install_to_pynq.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    checks: dict[str, bool] = {}
    # Keep the temporary tree beside the report so sandboxed CI environments
    # do not redirect it to an unwritable per-app Windows temp directory.
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="fitguard_installer_", dir=args.report.parent.resolve()
    ) as temporary:
        root = Path(temporary)
        (root / "FitGuard_Unified_Test").mkdir()
        (root / "FitGuard_Wireless_Gateway_Deploy").mkdir()

        module.install(root, check_only=True)
        checks["check_only_writes_nothing"] = not any(path.is_file() for path in root.rglob("*"))
        module.install(root, check_only=False)
        first = digest_tree(root)
        checks["all_nine_addon_files_installed"] = len(first) == 9
        checks["stable_bitstream_not_created_or_replaced"] = not (
            root / "FitGuard_Unified_Test" / "fitguard_unified.bit"
        ).exists()
        rejected = False
        try:
            module.install(root, check_only=False)
        except FileExistsError:
            rejected = True
        second = digest_tree(root)
        checks["repeat_install_rejected"] = rejected
        checks["repeat_install_changed_nothing"] = first == second
        checks["no_staging_files_left"] = not any(
            path.name.endswith(".installing") for path in root.rglob("*")
        )

    report = {"checks": checks, "passed": all(checks.values())}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
