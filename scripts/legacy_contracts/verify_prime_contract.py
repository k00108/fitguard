#!/usr/bin/env python3
"""Verify live PL priming and its strict/automatic failure semantics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "install_test_root" / "FitGuard_Unified_Test"))
sys.path.insert(0, str(ROOT.parent / "board_current"))
sys.path.insert(0, str(ROOT))

from fitguard_motion_m01_m06_cnn_runtime_pl import RangeFusionCNNPL


class FakePl:
    def __init__(self, fail=False):
        self.fail = bool(fail)
        self.calls = 0
        self.closed = False

    def infer(self, imu, vision):
        assert np.asarray(imu).shape == (128, 8)
        assert np.asarray(vision).shape == (64, 7)
        self.calls += 1
        if self.fail:
            raise RuntimeError("intentional prime failure")
        return np.zeros(160, dtype=np.float32)

    def status(self):
        return {"calls": self.calls, "active": self.calls > 0}

    def close(self):
        self.closed = True


def shell(requested, pl):
    model = RangeFusionCNNPL.__new__(RangeFusionCNNPL)
    model.backend_requested = requested
    model.backend_name = "pl_cnn_pending"
    model.pl_acceleration_active = False
    model._pl = pl
    model._pl_failure = None
    return model


success_pl = FakePl()
success = shell("pl", success_pl)
success_status = success.prime()

fallback_pl = FakePl(fail=True)
fallback = shell("auto", fallback_pl)
fallback_status = fallback.prime()

strict_pl = FakePl(fail=True)
strict = shell("pl", strict_pl)
strict_failed = False
try:
    strict.prime()
except RuntimeError:
    strict_failed = True

checks = {
    "successful_prime_activates_pl": success_status["active"] == "pl_cnn_conv1d",
    "successful_prime_marks_acceleration": success_status["pl_acceleration_active"] is True,
    "successful_prime_runs_once": success_pl.calls == 1,
    "auto_prime_failure_falls_back": fallback_status["active"] == "numpy_ps_fallback",
    "auto_prime_failure_records_reason": bool(fallback_status["fallback_reason"]),
    "auto_prime_failure_releases_pl": fallback_pl.closed and fallback._pl is None,
    "strict_prime_failure_raises": strict_failed,
    "strict_prime_failure_releases_pl": strict_pl.closed and strict._pl is None,
}
report = {"checks": checks, "passed": all(checks.values())}
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 1)
