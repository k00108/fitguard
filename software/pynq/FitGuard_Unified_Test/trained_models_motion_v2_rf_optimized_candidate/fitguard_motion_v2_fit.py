"""Audit and fit FitGuard motion-v2 data without touching the raw dataset.

The board clock check is intentionally ignored. All other acquisition quality
checks remain active. Evaluation is leave-one-subject-out (M01/M02/M03), and
augmentation is applied only to training folds to prevent leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABELS = ("Stable", "Mild Abnormality", "Marked Abnormality")
LABEL_TO_INDEX = {name: index for index, name in enumerate(LABELS)}
SUBJECTS = ("M01", "M02", "M03")


def failed_checks(metadata):
    checks = metadata.get("quality_checks", {})
    return sorted(
        key
        for key, value in checks.items()
        if not bool(value) and key != "clock_year_at_least_2026"
    )


def discover(root):
    trials = []
    for path in sorted(root.glob("M0[1-3]/**/windows.npz")):
        metadata_path = path.with_name("metadata.json")
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        subject = metadata.get("subject")
        label = metadata.get("label")
        if subject not in SUBJECTS or label not in LABEL_TO_INDEX:
            continue
        trials.append(
            {
                "path": path,
                "metadata_path": metadata_path,
                "metadata": metadata,
                "subject": subject,
                "label": label,
                "trial": str(metadata.get("trial", "")),
                "failed_checks_excluding_clock": failed_checks(metadata),
            }
        )
    return trials


def robust_stats(x):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    rms = np.sqrt(np.mean(x * x, axis=0) + 1e-12)
    q10, q25, q50, q75, q90 = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90], axis=0)
    delta = np.diff(x, axis=0)
    mad_delta = np.mean(np.abs(delta), axis=0)
    rms_delta = np.sqrt(np.mean(delta * delta, axis=0) + 1e-12)
    peak_delta = np.max(np.abs(delta), axis=0)
    return np.concatenate(
        [mean, std, rms, q10, q25, q50, q75, q90, q75 - q25,
         mad_delta, rms_delta, peak_delta]
    )


def spectral_features(x):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    power = power[1:]
    if len(power) == 0:
        return np.zeros(x.shape[1] * 5, dtype=np.float64)
    total = power.sum(axis=0) + 1e-12
    n = len(power)
    cuts = [0, max(1, n // 8), max(2, n // 4), max(3, n // 2), n]
    bands = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        bands.append(power[left:right].sum(axis=0) / total)
    dominant = np.argmax(power, axis=0) / max(n - 1, 1)
    return np.concatenate(bands + [dominant])


def correlation_features(x):
    corr = np.corrcoef(np.asarray(x, dtype=np.float64), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr[np.triu_indices(corr.shape[0], 1)]


def canonicalize(imu, vision, aux):
    imu = np.asarray(imu, dtype=np.float64).copy()
    vision = np.asarray(vision, dtype=np.float64).copy()
    aux = np.asarray(aux, dtype=np.float64).copy()

    # Remove person/mounting-specific DC offsets but retain dynamic magnitude.
    imu_centered = imu - np.median(imu, axis=0, keepdims=True)
    valid = aux[:, 0] > 0.5
    if valid.any():
        baseline = np.median(vision[valid, :5], axis=0)
        vision[valid, :2] -= baseline[:2]
        vision[valid, 2:5] = vision[valid, 2:5] / np.maximum(baseline[2:5], 1e-4) - 1.0
    vision[~valid] = 0.0
    return imu_centered, vision, aux, valid


def extract_features(imu, vision, aux):
    imu_c, vis_c, aux_c, valid = canonicalize(imu, vision, aux)
    acc_norm = np.linalg.norm(imu_c[:, :3], axis=1, keepdims=True)
    gyro_norm = np.linalg.norm(imu_c[:, 3:], axis=1, keepdims=True)
    imu_full = np.concatenate([imu_c, acc_norm, gyro_norm], axis=1)

    if valid.any():
        vis_valid = vis_c[valid]
        aux_valid = aux_c[valid]
    else:
        vis_valid = np.zeros((2, vis_c.shape[1]))
        aux_valid = np.zeros((2, aux_c.shape[1]))

    parts = [
        robust_stats(imu_full),
        spectral_features(imu_full),
        correlation_features(imu_c),
        robust_stats(vis_valid),
        spectral_features(vis_valid),
        robust_stats(aux_valid),
        np.asarray([
            valid.mean(),
            aux_valid[:, 1].mean(),
            aux_valid[:, 1].std(),
            aux_valid[:, 2].mean(),
            aux_valid[:, 2].std(),
        ]),
    ]
    result = np.concatenate(parts).astype(np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def augment_window(imu, vision, aux, rng):
    imu = np.asarray(imu, dtype=np.float32).copy()
    vision = np.asarray(vision, dtype=np.float32).copy()
    aux = np.asarray(aux, dtype=np.float32).copy()
    shift_imu = int(rng.integers(-5, 6))
    shift_vis = int(rng.integers(-2, 3))
    imu = np.roll(imu, shift_imu, axis=0)
    vision = np.roll(vision, shift_vis, axis=0)
    aux = np.roll(aux, shift_vis, axis=0)
    imu[:, :3] *= rng.uniform(0.94, 1.06, size=(1, 3))
    imu[:, 3:] *= rng.uniform(0.92, 1.08, size=(1, 3))
    imu += rng.normal(0.0, np.maximum(imu.std(axis=0), 1e-3) * 0.012, imu.shape)
    valid = aux[:, 0] > 0.5
    if valid.any():
        vision[valid] += rng.normal(
            0.0, np.maximum(vision[valid].std(axis=0), 1e-4) * 0.018,
            vision[valid].shape,
        )
        candidates = np.flatnonzero(valid)
        drop_count = min(len(candidates) // 20, int(rng.integers(0, 3)))
        if drop_count:
            dropped = rng.choice(candidates, size=drop_count, replace=False)
            vision[dropped] = 0.0
            aux[dropped] = 0.0
    return imu, vision, aux


def load_windows(trials):
    rows = []
    for item in trials:
        if item["failed_checks_excluding_clock"]:
            continue
        with np.load(item["path"], allow_pickle=False) as data:
            imu = data["imu"].astype(np.float32)
            vision = data["vision"].astype(np.float32)
            aux = data["vision_aux"].astype(np.float32)
        for index in range(len(imu)):
            rows.append(
                {
                    "imu": imu[index],
                    "vision": vision[index],
                    "aux": aux[index],
                    "x": extract_features(imu[index], vision[index], aux[index]),
                    "y": LABEL_TO_INDEX[item["label"]],
                    "subject": item["subject"],
                    "trial_path": str(item["path"].parent),
                }
            )
    return rows


def augmented_training(rows, rng, repeats):
    features, labels = [], []
    for row in rows:
        features.append(row["x"])
        labels.append(row["y"])
        for _ in range(repeats):
            imu, vision, aux = augment_window(row["imu"], row["vision"], row["aux"], rng)
            features.append(extract_features(imu, vision, aux))
            labels.append(row["y"])
    return np.stack(features), np.asarray(labels, dtype=np.int64)


def candidates(seed):
    return {
        "logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.35, max_iter=4000, class_weight="balanced", random_state=seed),
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed),
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=700, max_features=0.7, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=700, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        ),
    }


def score_predictions(y, probability, trial_paths):
    prediction = probability.argmax(axis=1)
    trial_prob = defaultdict(list)
    trial_y = {}
    for label, prob, path in zip(y, probability, trial_paths):
        trial_prob[path].append(prob)
        trial_y[path] = int(label)
    ordered = sorted(trial_prob)
    ty = np.asarray([trial_y[path] for path in ordered])
    tp = np.asarray([np.mean(trial_prob[path], axis=0).argmax() for path in ordered])
    return {
        "window_accuracy": float(accuracy_score(y, prediction)),
        "window_macro_f1": float(f1_score(y, prediction, average="macro")),
        "window_confusion_matrix": confusion_matrix(y, prediction, labels=range(3)).tolist(),
        "trial_accuracy": float(accuracy_score(ty, tp)),
        "trial_macro_f1": float(f1_score(ty, tp, average="macro")),
        "trial_confusion_matrix": confusion_matrix(ty, tp, labels=range(3)).tolist(),
        "test_windows": int(len(y)),
        "test_trials": int(len(ty)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--augment", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    trials = discover(args.root)
    if not trials:
        raise RuntimeError("No M01/M02/M03 trials found")
    audit = []
    for item in trials:
        metadata = item["metadata"]
        audit.append(
            {
                "subject": item["subject"],
                "label": item["label"],
                "trial": item["trial"],
                "trial_path": str(item["path"].parent.relative_to(args.root)),
                "usable": not item["failed_checks_excluding_clock"],
                "failed_checks_excluding_clock": item["failed_checks_excluding_clock"],
                "aligned_windows": metadata.get("aligned_windows"),
                "vision_valid_ratio": metadata.get("vision_valid_ratio"),
                "mean_tracking_quality": metadata.get("mean_tracking_quality"),
                "lost_samples": metadata.get("gateway", {}).get("lost_samples"),
                "bad_crc": metadata.get("gateway", {}).get("bad_crc"),
                "max_abs_gyro_bias": max(map(abs, metadata.get("gyro_bias_deg_s", [0]))),
            }
        )
    (args.output / "trial_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output / "trial_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit[0].keys())
        writer.writeheader()
        writer.writerows(audit)

    rows = load_windows(trials)
    print("usable_trials", sum(row["usable"] for row in audit), "total_trials", len(audit))
    print("windows", len(rows), "distribution", Counter((r["subject"], r["y"]) for r in rows))

    report = {
        "labels": list(LABELS),
        "subjects": list(SUBJECTS),
        "clock_check_ignored": True,
        "raw_trials_untouched": True,
        "usable_trials": sum(row["usable"] for row in audit),
        "excluded_trials": sum(not row["usable"] for row in audit),
        "usable_windows": len(rows),
        "augmentation_repeats_training_only": args.augment,
        "evaluation": "leave-one-subject-out; no trial/window leakage",
        "models": {},
    }
    pooled = defaultdict(lambda: {"y": [], "p": [], "trial": []})
    for held_out in SUBJECTS:
        train_rows = [row for row in rows if row["subject"] != held_out]
        test_rows = [row for row in rows if row["subject"] == held_out]
        rng = np.random.default_rng(args.seed + int(held_out[-1]))
        train_x, train_y = augmented_training(train_rows, rng, args.augment)
        test_x = np.stack([row["x"] for row in test_rows])
        test_y = np.asarray([row["y"] for row in test_rows])
        test_trials = [row["trial_path"] for row in test_rows]
        for name, model in candidates(args.seed).items():
            model.fit(train_x, train_y)
            probability = model.predict_proba(test_x)
            metrics = score_predictions(test_y, probability, test_trials)
            report["models"].setdefault(name, {"folds": []})["folds"].append(
                {"held_out_subject": held_out, **metrics}
            )
            pooled[name]["y"].extend(test_y.tolist())
            pooled[name]["p"].extend(probability.tolist())
            pooled[name]["trial"].extend(test_trials)
            print(name, held_out, metrics["window_macro_f1"], metrics["trial_macro_f1"])

    for name, payload in pooled.items():
        report["models"][name]["pooled"] = score_predictions(
            np.asarray(payload["y"]), np.asarray(payload["p"]), payload["trial"]
        )
    ranking = sorted(
        report["models"],
        key=lambda name: (
            report["models"][name]["pooled"]["trial_macro_f1"],
            report["models"][name]["pooled"]["window_macro_f1"],
        ),
        reverse=True,
    )
    report["ranking"] = ranking
    report["selected_model"] = ranking[0]

    # Final candidate is fitted on every usable real subject. Augmentation never
    # changes the audit set and the raw trials remain untouched.
    rng = np.random.default_rng(args.seed + 100)
    full_x, full_y = augmented_training(rows, rng, args.augment)
    final_model = candidates(args.seed)[ranking[0]]
    final_model.fit(full_x, full_y)
    joblib.dump(
        {
            "model": final_model,
            "labels": LABELS,
            "feature_version": "fitguard_motion_v2_robust_stats_v1",
            "source_subjects": SUBJECTS,
            "report": report,
        },
        args.output / "fitguard_motion_v2_candidate.joblib",
        compress=3,
    )
    (args.output / "fit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({name: report["models"][name]["pooled"] for name in ranking}, indent=2))
    print("selected", ranking[0])


if __name__ == "__main__":
    main()
