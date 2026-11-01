"""Board-side golden regression and latency check for the CNN PL backend."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from fitguard_cnn_pl_backend import FitGuardCnnPlBackend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitstream", required=True)
    parser.add_argument("--golden", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-error", type=float, default=0.001)
    args = parser.parse_args()
    golden = np.load(args.golden, allow_pickle=False)
    backend = FitGuardCnnPlBackend(args.bitstream, download=True)
    errors, times = [], []
    warmup_ms = None
    try:
        # The first DMA transaction after programming the overlay includes a
        # one-time driver/CMA warm-up. Report it separately so steady-state
        # inference is not misrepresented in the acceptance result.
        first_packet = golden["inputs"][0]
        warmup_started = time.monotonic()
        backend.infer(
            first_packet[:1024].reshape(128, 8),
            first_packet[1024:].reshape(64, 7),
        )
        warmup_ms = (time.monotonic() - warmup_started) * 1000.0
        for packet, expected in zip(golden["inputs"], golden["outputs"]):
            started = time.monotonic()
            actual = backend.infer(packet[:1024].reshape(128, 8), packet[1024:].reshape(64, 7))
            times.append((time.monotonic() - started) * 1000.0)
            errors.append(float(np.max(np.abs(actual - expected))))
    finally:
        backend.close()
    report = {
        "cases": len(errors),
        "max_abs_error": max(errors),
        "warmup_latency_ms": float(warmup_ms),
        "mean_latency_ms": float(np.mean(times)),
        "p50_latency_ms": float(np.percentile(times, 50)),
        "p95_latency_ms": float(np.percentile(times, 95)),
        "max_latency_ms": max(times),
        "passed": bool(max(errors) <= args.max_error),
        "threshold": args.max_error,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
