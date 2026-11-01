"""Board-side, like-for-like PS NumPy versus PL Conv1D benchmark.

This benchmark does not alter the trained model, runtime selection policy, or
bitstream.  It uses the normalized 18-case golden input set and times exactly
the two six-layer Conv1D branches through GAP, whose 160-value output is the
PL accelerator contract.
"""

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np

from fitguard_cnn_pl_backend import FitGuardCnnPlBackend


def _conv_relu(x, weight, bias):
    kernel = weight.shape[0]
    left = (kernel - 1) // 2
    right = kernel - 1 - left
    padded = np.pad(x, ((left, right), (0, 0)))
    windows = np.stack([padded[index : index + kernel] for index in range(len(x))])
    return np.maximum(
        np.tensordot(windows, weight, axes=([1, 2], [0, 1])) + bias,
        0.0,
    )


def _pool2(x):
    usable = len(x) - len(x) % 2
    return x[:usable].reshape(usable // 2, 2, x.shape[1]).max(axis=1)


class NumpyConvGapBackend:
    """Reference implementation matching the PL boundary, without Dense layers."""

    def __init__(self, weights_path):
        with np.load(weights_path, allow_pickle=False) as data:
            self.weights = {
                key: data[key].astype(np.float32)
                for key in data.files
                if key not in ("model_kind", "labels", "range_names")
            }

    def _branch_gap(self, x, prefix):
        w = self.weights
        x = _conv_relu(x, w[f"{prefix}_conv1_0"], w[f"{prefix}_conv1_1"])
        x = _pool2(x)
        x = _conv_relu(x, w[f"{prefix}_conv2_0"], w[f"{prefix}_conv2_1"])
        x = _pool2(x)
        x = _conv_relu(x, w[f"{prefix}_conv3_0"], w[f"{prefix}_conv3_1"])
        return x.mean(axis=0).astype(np.float32)

    def infer(self, imu_normalized, vision_normalized):
        imu = np.asarray(imu_normalized, dtype=np.float32)
        vision = np.asarray(vision_normalized, dtype=np.float32)
        if imu.shape != (128, 8) or vision.shape != (64, 7):
            raise ValueError(f"unexpected input shapes: {imu.shape}, {vision.shape}")
        return np.concatenate(
            (self._branch_gap(imu, "imu"), self._branch_gap(vision, "vision"))
        )


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def _measure(backend, packets, repeats):
    times = []
    outputs = []
    for _ in range(repeats):
        for packet in packets:
            imu = packet[:1024].reshape(128, 8)
            vision = packet[1024:].reshape(64, 7)
            started = time.perf_counter_ns()
            output = backend.infer(imu, vision)
            times.append((time.perf_counter_ns() - started) / 1_000_000.0)
            outputs.append(np.asarray(output, dtype=np.float32))
    return times, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--bitstream", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--pl-max-error", type=float, default=0.001)
    parser.add_argument("--numpy-max-error", type=float, default=0.002)
    parser.add_argument("--numpy-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")

    with np.load(args.golden, allow_pickle=False) as golden:
        packets = golden["inputs"].astype(np.float32)
        expected = golden["outputs"].astype(np.float32)

    numpy_backend = NumpyConvGapBackend(args.model_dir / "range_fusion_v2_weights.npz")
    _measure(numpy_backend, packets[:1], 1)
    numpy_times, numpy_outputs = _measure(numpy_backend, packets, args.repeats)
    numpy_first_pass = np.stack(numpy_outputs[: len(packets)])
    numpy_error = float(np.max(np.abs(numpy_first_pass - expected)))

    report = {
        "benchmark_scope": "six Conv1D/ReLU layers, two pooling stages per branch, and GAP",
        "input_contract": "same normalized 18-case golden tensors for PS and PL",
        "timing_boundary": "one backend infer call; excludes input normalization, Dense fusion, softmax, sensors, and UI",
        "cases": int(len(packets)),
        "repeats": int(args.repeats),
        "platform": platform.platform(),
        "numpy_ps": {
            **_summary(numpy_times),
            "max_abs_error_vs_golden": numpy_error,
        },
    }

    if not args.numpy_only:
        if args.bitstream is None:
            raise ValueError("--bitstream is required unless --numpy-only is used")
        pl_backend = FitGuardCnnPlBackend(args.bitstream, download=True)
        try:
            warmup_started = time.perf_counter_ns()
            pl_backend.infer(
                packets[0, :1024].reshape(128, 8),
                packets[0, 1024:].reshape(64, 7),
            )
            warmup_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000.0
            pl_times, pl_outputs = _measure(pl_backend, packets, args.repeats)
        finally:
            pl_backend.close()
        pl_first_pass = np.stack(pl_outputs[: len(packets)])
        pl_error = float(np.max(np.abs(pl_first_pass - expected)))
        cross_error = float(np.max(np.abs(pl_first_pass - numpy_first_pass)))
        pl_summary = _summary(pl_times)
        report["pl"] = {
            **pl_summary,
            "warmup_ms": float(warmup_ms),
            "max_abs_error_vs_golden": pl_error,
            "max_abs_error_vs_numpy_ps": cross_error,
        }
        report["speedup_numpy_ps_over_pl"] = float(
            report["numpy_ps"]["mean_ms"] / pl_summary["mean_ms"]
        )
        report["passed"] = bool(
            numpy_error <= args.numpy_max_error
            and pl_error <= args.pl_max_error
            and cross_error <= args.numpy_max_error
        )
    else:
        report["passed"] = bool(numpy_error <= args.numpy_max_error)

    report["thresholds"] = {
        "pl_vs_golden": float(args.pl_max_error),
        "numpy_vs_golden": float(args.numpy_max_error),
        "pl_vs_numpy": float(args.numpy_max_error),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
