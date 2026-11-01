"""Evaluate and export a two-forest FitGuard robustness blend.

This is a candidate-only artifact.  It keeps the accurate incumbent-style RF
and blends it with a trial-balanced, high-feature RF to reduce subject-specific
errors.  Both forests use the same 311-dimensional deployed feature extractor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np

import fitguard_motion_v2_fit as core
from fitguard_motion_v2_export_numpy import export_forest
from fitguard_motion_v2_numpy_runtime import NumpyRandomForest
from fitguard_motion_v2_optimize_rf import equal_trial_weights, evaluate, make_model


BASE_CONFIG = {
    "max_features": "sqrt", "min_samples_leaf": 1, "max_depth": None,
    "criterion": "gini", "max_samples": None,
}
ROBUST_CONFIG = {
    "max_features": 0.6, "min_samples_leaf": 1, "max_depth": None,
    "criterion": "entropy", "max_samples": None,
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def fit_pair(train, base_trees, robust_trees):
    x = np.stack([r["x"] for r in train])
    y = np.asarray([r["y"] for r in train])
    base = make_model(BASE_CONFIG, base_trees, 20261008)
    robust = make_model(ROBUST_CONFIG, robust_trees, 20261010)
    base.fit(x, y)
    robust.fit(x, y, sample_weight=equal_trial_weights(train))
    return base, robust


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--base-trees", type=int, default=500)
    p.add_argument("--robust-trees", type=int, default=600)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = core.load_windows(core.discover(args.root))

    records = []
    for held_out in core.SUBJECTS:
        train = [r for r in rows if r["subject"] != held_out]
        test = [r for r in rows if r["subject"] == held_out]
        base, robust = fit_pair(train, args.base_trees, args.robust_trees)
        x = np.stack([r["x"] for r in test])
        for row, pb, pr in zip(test, base.predict_proba(x), robust.predict_proba(x)):
            records.append({"row": row, "base": pb, "robust": pr})

    candidates = []
    for robust_weight in np.linspace(0.0, 1.0, 21):
        probability = np.stack([
            (1.0 - robust_weight) * r["base"] + robust_weight * r["robust"]
            for r in records
        ])
        y = np.asarray([r["row"]["y"] for r in records])
        trials = [r["row"]["trial_path"] for r in records]
        pooled = evaluate(y, probability, trials)
        folds = []
        for subject in core.SUBJECTS:
            indices = [i for i, r in enumerate(records) if r["row"]["subject"] == subject]
            folds.append({
                "held_out_subject": subject,
                **evaluate(y[indices], probability[indices], [trials[i] for i in indices]),
            })
        candidates.append({
            "robust_weight": float(robust_weight),
            "pooled": pooled,
            "folds": folds,
            "minimum_subject_window_macro_f1": min(f["window_macro_f1"] for f in folds),
        })
    candidates.sort(key=lambda r: (
        r["pooled"]["trial_macro_f1"],
        r["minimum_subject_window_macro_f1"],
        r["pooled"]["window_macro_f1"],
        -r["pooled"]["window_log_loss"],
    ), reverse=True)
    best = candidates[0]

    base, robust = fit_pair(rows, args.base_trees, args.robust_trees)
    base_joblib = args.output / "fitguard_motion_v2_rf_base.joblib"
    robust_joblib = args.output / "fitguard_motion_v2_rf_robust.joblib"
    base_npz = args.output / "fitguard_motion_v2_rf_base_numpy.npz"
    robust_npz = args.output / "fitguard_motion_v2_rf_robust_numpy.npz"
    joblib.dump(base, base_joblib, compress=3)
    joblib.dump(robust, robust_joblib, compress=3)
    export_forest(base, base_npz)
    export_forest(robust, robust_npz)

    x = np.stack([r["x"] for r in rows])
    pb = NumpyRandomForest(base_npz).predict_proba_features(x)
    pr = NumpyRandomForest(robust_npz).predict_proba_features(x)
    sklearn_pb, sklearn_pr = base.predict_proba(x), robust.predict_proba(x)
    max_error = float(max(np.max(np.abs(pb - sklearn_pb)), np.max(np.abs(pr - sklearn_pr))))
    agreement = float(np.mean(
        ((1 - best["robust_weight"]) * pb + best["robust_weight"] * pr).argmax(1)
        == ((1 - best["robust_weight"]) * sklearn_pb + best["robust_weight"] * sklearn_pr).argmax(1)
    ))
    if max_error > 1e-5 or agreement != 1.0:
        raise RuntimeError(f"NumPy parity failure: {max_error}, {agreement}")

    deployment = {
        "feature_version": "fitguard_motion_v2_robust_stats_v1",
        "labels": list(core.LABELS),
        "base_model": base_npz.name,
        "robust_model": robust_npz.name,
        "robust_weight": best["robust_weight"],
        "base_weight": 1.0 - best["robust_weight"],
    }
    (args.output / "rf_blend_config.json").write_text(
        json.dumps(deployment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "status": "candidate_only_incumbent_preserved",
        "selection_limit": "Blend weight selected on M01-M03; must be re-evaluated with M04-M06",
        "windows": len(rows),
        "trials": len(set(r["trial_path"] for r in rows)),
        "best": best,
        "all_blend_weights": candidates,
        "numpy_parity_max_abs_error": max_error,
        "numpy_class_agreement": agreement,
        "sha256": {
            base_npz.name: sha256(base_npz), robust_npz.name: sha256(robust_npz),
        },
    }
    (args.output / "rf_blend_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"deployment": deployment, "best": best, "parity": {
        "max_abs_error": max_error, "class_agreement": agreement,
    }}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
