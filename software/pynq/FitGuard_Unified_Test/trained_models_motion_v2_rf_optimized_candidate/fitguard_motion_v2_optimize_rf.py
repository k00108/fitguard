"""Robust, subject-isolated optimization of the FitGuard fusion random forest.

The incumbent model is never overwritten.  Search models use whole-subject
holdout folds, and every source trial receives the same total training weight
so overlap-window count cannot make one trial dominate another.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

import fitguard_motion_v2_fit as core
from fitguard_motion_v2_export_numpy import export_forest
from fitguard_motion_v2_numpy_runtime import NumpyRandomForest, extract_features


def equal_trial_weights(rows):
    counts = defaultdict(int)
    for row in rows:
        counts[row["trial_path"]] += 1
    # Mean one keeps sklearn's regularisation scale easy to interpret.
    raw = np.asarray([1.0 / counts[row["trial_path"]] for row in rows])
    return raw / raw.mean()


def evaluate(y, probability, trials):
    pred = probability.argmax(axis=1)
    grouped_p = defaultdict(list)
    grouped_y = {}
    for target, prob, trial in zip(y, probability, trials):
        grouped_p[trial].append(prob)
        grouped_y[trial] = int(target)
    ordered = sorted(grouped_p)
    trial_y = np.asarray([grouped_y[t] for t in ordered])
    trial_p = np.asarray([np.mean(grouped_p[t], axis=0) for t in ordered])
    trial_pred = trial_p.argmax(axis=1)
    return {
        "window_accuracy": float(accuracy_score(y, pred)),
        "window_macro_f1": float(f1_score(y, pred, average="macro")),
        "window_log_loss": float(log_loss(y, probability, labels=np.arange(3))),
        "window_confusion_matrix": confusion_matrix(y, pred, labels=np.arange(3)).tolist(),
        "trial_accuracy": float(accuracy_score(trial_y, trial_pred)),
        "trial_macro_f1": float(f1_score(trial_y, trial_pred, average="macro")),
        "trial_confusion_matrix": confusion_matrix(trial_y, trial_pred, labels=np.arange(3)).tolist(),
        "windows": int(len(y)),
        "trials": int(len(trial_y)),
    }


def make_model(config, trees, seed):
    return RandomForestClassifier(
        n_estimators=trees,
        max_features=config["max_features"],
        min_samples_leaf=config["min_samples_leaf"],
        max_depth=config["max_depth"],
        criterion=config["criterion"],
        max_samples=config["max_samples"],
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )


def candidate_configs():
    # Deliberately compact, regularisation-oriented search.  It includes the
    # incumbent configuration and avoids tuning hundreds of near-duplicates to
    # only three held-out subjects.
    result = []
    for max_features in ("sqrt", 0.25, 0.4, 0.6):
        for leaf in (1, 2, 3):
            for depth in (None, 18):
                for criterion in ("gini", "entropy"):
                    result.append({
                        "max_features": max_features,
                        "min_samples_leaf": leaf,
                        "max_depth": depth,
                        "criterion": criterion,
                        "max_samples": None,
                    })
    return result


def loso(rows, config, trees, seed, use_trial_weights):
    pooled_y, pooled_p, pooled_trials = [], [], []
    folds = []
    for fold_index, held_out in enumerate(core.SUBJECTS):
        train = [r for r in rows if r["subject"] != held_out]
        test = [r for r in rows if r["subject"] == held_out]
        x_train = np.stack([r["x"] for r in train])
        y_train = np.asarray([r["y"] for r in train])
        x_test = np.stack([r["x"] for r in test])
        y_test = np.asarray([r["y"] for r in test])
        model = make_model(config, trees, seed + fold_index)
        weights = equal_trial_weights(train) if use_trial_weights else None
        model.fit(x_train, y_train, sample_weight=weights)
        probability = model.predict_proba(x_test)
        trials = [r["trial_path"] for r in test]
        folds.append({"held_out_subject": held_out, **evaluate(y_test, probability, trials)})
        pooled_y.extend(y_test.tolist())
        pooled_p.extend(probability.tolist())
        pooled_trials.extend(trials)
    pooled = evaluate(np.asarray(pooled_y), np.asarray(pooled_p), pooled_trials)
    minimum_subject_f1 = min(f["window_macro_f1"] for f in folds)
    return {"folds": folds, "pooled": pooled, "minimum_subject_window_macro_f1": minimum_subject_f1}


def ranking_key(item):
    metrics = item["metrics"]
    pooled = metrics["pooled"]
    # Robustness takes priority over a tiny pooled gain.  Trial accuracy is
    # first because a complete 30 s recording is the project's actual unit.
    return (
        pooled["trial_macro_f1"],
        metrics["minimum_subject_window_macro_f1"],
        pooled["window_macro_f1"],
        -pooled["window_log_loss"],
        -item["config"]["min_samples_leaf"],
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--search-trees", type=int, default=180)
    parser.add_argument("--final-trees", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = core.load_windows(core.discover(args.root))
    if not rows:
        raise RuntimeError("No usable windows found")
    print(f"Loaded {len(rows)} windows from {len(set(r['trial_path'] for r in rows))} trials")

    results = []
    started = time.perf_counter()
    for weighted in (False, True):
        for index, config in enumerate(candidate_configs(), 1):
            metrics = loso(rows, config, args.search_trees, args.seed, weighted)
            result = {"config": config, "equal_trial_weighting": weighted, "metrics": metrics}
            results.append(result)
            print(
                f"weighted={weighted} {index:02d}/48 "
                f"minF1={metrics['minimum_subject_window_macro_f1']:.6f} "
                f"pooledF1={metrics['pooled']['window_macro_f1']:.6f} "
                f"logloss={metrics['pooled']['window_log_loss']:.6f}"
            )

    results.sort(key=ranking_key, reverse=True)
    best = results[0]
    # Re-evaluate the winner with the exact final tree count.
    final_loso = loso(
        rows, best["config"], args.final_trees, args.seed,
        best["equal_trial_weighting"],
    )

    x = np.stack([r["x"] for r in rows])
    y = np.asarray([r["y"] for r in rows])
    final = make_model(best["config"], args.final_trees, args.seed)
    weights = equal_trial_weights(rows) if best["equal_trial_weighting"] else None
    final.fit(x, y, sample_weight=weights)

    joblib_path = args.output / "fitguard_motion_v2_rf_optimized.joblib"
    npz_path = args.output / "fitguard_motion_v2_rf_optimized_numpy.npz"
    joblib.dump(final, joblib_path, compress=3)
    export_forest(final, npz_path)

    pure = NumpyRandomForest(npz_path)
    sklearn_probability = final.predict_proba(x)
    inference_started = time.perf_counter()
    numpy_probability = pure.predict_proba_features(x)
    inference_seconds = time.perf_counter() - inference_started
    probability_error = float(np.max(np.abs(sklearn_probability - numpy_probability)))
    agreement = float(np.mean(sklearn_probability.argmax(1) == numpy_probability.argmax(1)))
    feature_error = max(
        float(np.max(np.abs(r["x"] - extract_features(r["imu"], r["vision"], r["aux"]))))
        for r in rows[:25]
    )
    if probability_error > 1e-5 or agreement != 1.0 or feature_error > 1e-6:
        raise RuntimeError(
            f"Export parity failed: probability={probability_error}, "
            f"agreement={agreement}, feature={feature_error}"
        )

    report = {
        "status": "candidate_only_incumbent_preserved",
        "selection_note": "M01-M03 optimization evidence; re-evaluate after M04-M06",
        "labels": list(core.LABELS),
        "subjects": list(core.SUBJECTS),
        "windows": len(rows),
        "trials": len(set(r["trial_path"] for r in rows)),
        "feature_dimensions": int(x.shape[1]),
        "search_trees": args.search_trees,
        "final_trees": args.final_trees,
        "best_search_result": best,
        "final_loso": final_loso,
        "export_validation": {
            "feature_parity_max_abs_error": feature_error,
            "probability_parity_max_abs_error": probability_error,
            "class_agreement": agreement,
            "local_ms_per_window": inference_seconds * 1000.0 / len(rows),
            "joblib_sha256": sha256(joblib_path),
            "numpy_sha256": sha256(npz_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "all_search_results": results,
    }
    (args.output / "rf_optimization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "best_config": best["config"],
        "equal_trial_weighting": best["equal_trial_weighting"],
        "final_loso": final_loso,
        "export_validation": report["export_validation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
