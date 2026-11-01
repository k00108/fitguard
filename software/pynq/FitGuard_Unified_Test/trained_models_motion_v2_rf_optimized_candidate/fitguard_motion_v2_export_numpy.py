"""Train the final overlap-window RF and export an sklearn-free runtime."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import fitguard_motion_v2_fit as core
import fitguard_motion_v2_numpy_runtime as runtime


def export_forest(model, path):
    left, right, feature, threshold, value = [], [], [], [], []
    offsets = [0]
    for estimator in model.estimators_:
        tree = estimator.tree_
        offset = offsets[-1]
        left.append(np.where(tree.children_left < 0, -1, tree.children_left + offset))
        right.append(np.where(tree.children_right < 0, -1, tree.children_right + offset))
        feature.append(tree.feature)
        threshold.append(tree.threshold)
        value.append(tree.value[:, 0, :])
        offsets.append(offset + tree.node_count)
    np.savez_compressed(
        path,
        children_left=np.concatenate(left).astype(np.int32),
        children_right=np.concatenate(right).astype(np.int32),
        feature=np.concatenate(feature).astype(np.int32),
        # Keep sklearn's float64 thresholds and leaf values. Casting thresholds
        # to float32 can move a sample across an exact split boundary.
        threshold=np.concatenate(threshold).astype(np.float64),
        value=np.concatenate(value).astype(np.float64),
        offsets=np.asarray(offsets, dtype=np.int32),
        labels=np.asarray(core.LABELS),
        feature_version=np.asarray("fitguard_motion_v2_robust_stats_v1"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = core.load_windows(core.discover(args.root))
    x = np.stack([row["x"] for row in rows])
    y = np.asarray([row["y"] for row in rows])
    model = RandomForestClassifier(
        n_estimators=args.trees, max_features="sqrt", min_samples_leaf=1,
        class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1,
    )
    model.fit(x, y)
    joblib.dump(model, args.output / "fitguard_motion_v2_overlap_rf.joblib", compress=3)
    npz_path = args.output / "fitguard_motion_v2_overlap_rf_numpy.npz"
    export_forest(model, npz_path)
    pure = runtime.NumpyRandomForest(npz_path)
    sklearn_probability = model.predict_proba(x)
    started = time.perf_counter()
    numpy_probability = pure.predict_proba_features(x)
    elapsed = time.perf_counter() - started
    max_error = float(np.max(np.abs(sklearn_probability - numpy_probability)))
    agreement = float(np.mean(sklearn_probability.argmax(axis=1) == numpy_probability.argmax(axis=1)))
    if max_error > 1e-5 or agreement != 1.0:
        raise RuntimeError(f"NumPy export mismatch: error={max_error}, agreement={agreement}")
    feature_errors = []
    for row in rows[:25]:
        feature_errors.append(float(np.max(np.abs(row["x"] - runtime.extract_features(row["imu"], row["vision"], row["aux"])))))
    if max(feature_errors) > 1e-6:
        raise RuntimeError(f"Feature extractor mismatch: {max(feature_errors)}")
    report = {
        "labels": list(core.LABELS),
        "trees": args.trees,
        "windows_fitted": len(rows),
        "subjects": list(core.SUBJECTS),
        "model_format": "pure NumPy flattened random forest",
        "sklearn_required_on_pynq": False,
        "feature_dimensions": int(x.shape[1]),
        "feature_parity_max_abs_error": max(feature_errors),
        "probability_parity_max_abs_error": max_error,
        "class_agreement": agreement,
        "local_batch_inference_seconds": elapsed,
        "local_ms_per_window": elapsed * 1000.0 / len(x),
    }
    (args.output / "numpy_export_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
