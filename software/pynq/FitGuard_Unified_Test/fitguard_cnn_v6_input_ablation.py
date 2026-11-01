"""Post-training input ablation for the validated CNN V6/V7 weights.

This measures model dependency by replacing one normalized modality with its
training-set mean. It is not a substitute for retraining separate unimodal
models, so the report names the test "input_ablation" explicitly.
"""

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from fitguard_motion_m01_m06_cnn_runtime_v7 import LABELS, RangeFusionCNNV7


LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def _valid_trial(metadata):
    if metadata.get("subject") not in {f"M{i:02d}" for i in range(1, 7)}:
        return False
    checks = metadata.get("quality_checks", {})
    return all(
        value
        for name, value in checks.items()
        if name != "clock_year_at_least_2026"
    )


def _confusion(y_true, y_pred):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


def _metrics(y_true, y_pred):
    matrix = _confusion(y_true, y_pred)
    f1 = []
    for index in range(len(LABELS)):
        tp = float(matrix[index, index])
        fp = float(matrix[:, index].sum() - tp)
        fn = float(matrix[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": matrix.tolist(),
    }


def run_input_ablation(
    dataset_root="dataset_motion_v2",
    model_dir="trained_models_motion_v2_cnn_v6_candidate",
    output_path="validation_results/cnn_v6_input_ablation.json",
):
    root = Path(dataset_root)
    model = RangeFusionCNNV7(model_dir)
    modes = {"full": None, "imu_mean_replaced": "imu", "vision_mean_replaced": "vision"}
    truth = []
    prediction = {name: [] for name in modes}
    trial_rows = []
    subjects = Counter()
    started = time.monotonic()

    for metadata_path in sorted(root.glob("M??/**/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not _valid_trial(metadata):
            continue
        windows_path = metadata_path.with_name("windows.npz")
        if not windows_path.exists():
            continue
        with np.load(windows_path, allow_pickle=False) as data:
            imu = data["imu"].astype(np.float32)
            vision = data["vision"].astype(np.float32)
            aux = data["vision_aux"].astype(np.float32)
            labels = data["label"].astype(str)
        trial_truth = [LABEL_TO_INDEX[label] for label in labels]
        truth.extend(trial_truth)
        subjects[metadata["subject"]] += len(labels)
        trial_entry = {
            "subject": metadata["subject"],
            "label": metadata["label"],
            "trial": metadata.get("trial"),
            "windows": int(len(labels)),
            "predictions": {},
        }
        for mode_name, ablate in modes.items():
            probabilities = np.asarray(
                [
                    model.predict(i, v, a, ablate=ablate)
                    for i, v, a in zip(imu, vision, aux)
                ]
            )
            predicted = np.argmax(probabilities, axis=1).astype(int)
            prediction[mode_name].extend(predicted.tolist())
            mean_probability = probabilities.mean(axis=0)
            trial_entry["predictions"][mode_name] = {
                "label": LABELS[int(np.argmax(mean_probability))],
                "probabilities": mean_probability.tolist(),
            }
        trial_rows.append(trial_entry)

    truth_array = np.asarray(truth, dtype=int)
    report = {
        "report_type": "post_training_input_ablation",
        "warning": (
            "Ablated branches are replaced with their normalized training mean. "
            "This tests dependency of the deployed fusion model; it is not a "
            "comparison with separately retrained unimodal models."
        ),
        "model": "cnn_v6_no_excursion_m01_m06",
        "backend": model.backend_status(),
        "subjects": dict(subjects),
        "trials": len(trial_rows),
        "windows": int(len(truth_array)),
        "window_metrics": {
            name: _metrics(truth_array, np.asarray(values, dtype=int))
            for name, values in prediction.items()
        },
        "trial_metrics": {},
        "elapsed_seconds": float(time.monotonic() - started),
        "trials_detail": trial_rows,
    }
    trial_truth = [LABEL_TO_INDEX[row["label"]] for row in trial_rows]
    for mode_name in modes:
        trial_pred = [
            LABEL_TO_INDEX[row["predictions"][mode_name]["label"]]
            for row in trial_rows
        ]
        report["trial_metrics"][mode_name] = _metrics(trial_truth, trial_pred)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "trials_detail"}, ensure_ascii=False, indent=2))
    print(f"saved: {output}")
    return report


if __name__ == "__main__":
    run_input_ablation()
