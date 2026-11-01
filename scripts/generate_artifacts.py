from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SNAP = ROOT / "software" / "pynq" / "FitGuard_Unified_Test"
MODEL = ROOT / "models" / "cnn_v6"
sys.path.insert(0, str(SNAP))

from fitguard_motion_m01_m06_cnn_runtime import RangeFusionCNN, canonicalize  # noqa: E402


TOTAL_BITS = 18
FRAC_BITS = 13


def q(x):
    scale = 1 << FRAC_BITS
    lo = -(1 << (TOTAL_BITS - 1)) / scale
    hi = ((1 << (TOTAL_BITS - 1)) - 1) / scale
    return np.clip(np.rint(np.asarray(x, np.float64) * scale) / scale, lo, hi)


def conv_relu(x, w, b):
    k = w.shape[0]
    padded = np.pad(x, (((k - 1) // 2, k // 2), (0, 0)))
    out = np.empty((len(x), w.shape[2]), np.float64)
    for t in range(len(x)):
        out[t] = np.tensordot(padded[t:t+k], w, axes=([0, 1], [0, 1])) + b
    return q(np.maximum(out, 0.0))


def pool2(x):
    return np.maximum(x[0::2], x[1::2])


def branch_gap(x, weights, prefix):
    x = q(x)
    for layer in (1, 2, 3):
        w = q(weights[f"{prefix}_conv{layer}_0"])
        b = q(weights[f"{prefix}_conv{layer}_1"])
        x = conv_relu(x, w, b)
        if layer < 3:
            x = pool2(x)
    return q(x.mean(axis=0))


def c_array(name, arr, ctype="weight_t"):
    arr = np.asarray(arr)
    dims = "".join(f"[{n}]" for n in arr.shape)
    values = ",".join(f"{float(v):.10g}" for v in arr.reshape(-1))
    return f"static const {ctype} {name}{dims} = {{{values}}};\n"


def main():
    out = ROOT / "hardware" / "hls" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    model = RangeFusionCNN(MODEL)
    names = [
        "imu_conv1_0", "imu_conv1_1", "imu_conv2_0", "imu_conv2_1", "imu_conv3_0", "imu_conv3_1",
        "vision_conv1_0", "vision_conv1_1", "vision_conv2_0", "vision_conv2_1",
        "vision_conv3_0", "vision_conv3_1",
    ]
    header = [
        "#pragma once\n",
        '#include "ap_fixed.h"\n',
        f"typedef ap_fixed<{TOTAL_BITS},{TOTAL_BITS-FRAC_BITS},AP_RND,AP_SAT> weight_t;\n",
    ]
    for name in names:
        header.append(c_array(name, q(model.w[name])))
    (out / "fitguard_cnn_weights.hpp").write_text("".join(header), encoding="utf-8")

    golden = np.load(MODEL / "cnn_v6_golden_batch.npz", allow_pickle=False)
    inputs = []
    outputs = []
    for imu_raw, vis_raw, aux in zip(golden["imu"], golden["vision"], golden["vision_aux"]):
        imu, vis, _ = canonicalize(imu_raw, vis_raw, aux)
        imu = np.clip((imu - model.imu_mean) / model.imu_std, -8, 8)
        vis = np.clip((vis - model.vis_mean) / model.vis_std, -8, 8)
        inputs.append(np.concatenate([imu.reshape(-1), vis.reshape(-1)]).astype(np.float32))
        outputs.append(np.concatenate([branch_gap(imu, model.w, "imu"), branch_gap(vis, model.w, "vision")]).astype(np.float32))
    test_header = ["#pragma once\n", f"#define GOLDEN_CASES {len(inputs)}\n"]
    test_header.append(c_array("golden_input", np.asarray(inputs), "float"))
    test_header.append(c_array("golden_output", np.asarray(outputs), "float"))
    (out / "fitguard_cnn_golden.hpp").write_text("".join(test_header), encoding="utf-8")
    np.savez_compressed(
        out / "fitguard_cnn_pl_golden.npz",
        inputs=np.asarray(inputs, dtype=np.float32),
        outputs=np.asarray(outputs, dtype=np.float32),
    )
    manifest = {
        "fixed_format": f"ap_fixed<{TOTAL_BITS},{TOTAL_BITS-FRAC_BITS},AP_RND,AP_SAT>",
        "input_floats": int(np.asarray(inputs).shape[1]),
        "output_floats": int(np.asarray(outputs).shape[1]),
        "golden_cases": len(inputs),
        "model_dir": str(MODEL),
        "weights": {name: list(model.w[name].shape) for name in names},
    }
    (out / "generation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
