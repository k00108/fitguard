"""Train FitGuard waist models with real-subject LOSO evaluation.

Synthetic samples are permitted only for training folds. Validation and test
sets always contain untouched real windows from held-out subjects.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from fitguard_train_models import (
    LABELS,
    LABEL_TO_INDEX,
    build_models,
    branch,
    export_numpy_weights,
    export_tflite,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def load_root(root, expect_synthetic):
    imu, vision, labels, subjects, trials, synthetic = [], [], [], [], [], []
    for path in sorted(Path(root).glob("W*/*/trial_*/windows.npz")):
        with np.load(path, allow_pickle=False) as data:
            subject = str(data["subject"][0])
            label = str(data["label"][0])
            if not subject.startswith("W") or label not in LABEL_TO_INDEX:
                continue
            imu_item = data["imu"].astype(np.float32)
            vision_item = data["vision"].astype(np.float32)
            count = len(imu_item)
            is_synthetic = bool(
                "synthetic" in data.files and np.asarray(data["synthetic"]).any()
            )
            if is_synthetic != expect_synthetic:
                raise ValueError(f"Unexpected synthetic flag in {path}")
            if imu_item.shape != (count, 128, 6):
                raise ValueError(f"Bad IMU shape in {path}: {imu_item.shape}")
            if vision_item.shape != (count, 64, 7):
                raise ValueError(f"Bad vision shape in {path}: {vision_item.shape}")
            if not np.isfinite(imu_item).all() or not np.isfinite(vision_item).all():
                raise ValueError(f"Non-finite sample in {path}")
            imu.append(imu_item)
            vision.append(vision_item)
            labels.extend([LABEL_TO_INDEX[label]] * count)
            subjects.extend([subject] * count)
            trials.extend([str(path.parent)] * count)
            synthetic.extend([is_synthetic] * count)
    if not imu:
        raise RuntimeError(f"No data found below {root}")
    return {
        "imu": np.concatenate(imu),
        "vision": np.concatenate(vision),
        "y": np.asarray(labels, dtype=np.int64),
        "subject": np.asarray(subjects),
        "trial": np.asarray(trials),
        "synthetic": np.asarray(synthetic, dtype=bool),
    }


def canonicalize_waist_windows(imu, vision):
    """Remove subject/mounting offsets while retaining motion magnitude."""
    imu = imu.astype(np.float32, copy=True)
    vision = vision.astype(np.float32, copy=True)
    # Gravity projection, gyro bias and sensor zero offsets vary with mounting.
    # The within-window dynamic component is the transferable stability cue.
    imu -= imu.mean(axis=1, keepdims=True)
    for index in range(len(vision)):
        present = vision[index, :, 2] > 1e-8
        if not present.any():
            vision[index] = 0.0
            continue
        baseline = np.median(vision[index, present, :5], axis=0)
        vision[index, present, 0:2] -= baseline[0:2]
        scale = np.maximum(baseline[2:5], 1e-4)
        vision[index, present, 2:5] = (
            vision[index, present, 2:5] / scale - 1.0
        )
        vision[index, ~present] = 0.0
    return imu, vision


def normalization(real_imu, real_vision):
    imu_mean = real_imu.mean(axis=(0, 1))
    imu_std = np.maximum(real_imu.std(axis=(0, 1)), 1e-6)
    vis_mean = real_vision.mean(axis=(0, 1))
    vis_std = np.maximum(real_vision.std(axis=(0, 1)), 1e-6)
    return imu_mean, imu_std, vis_mean, vis_std


def normalize(imu, vision, stats):
    imu_mean, imu_std, vis_mean, vis_std = stats
    imu = ((imu - imu_mean) / imu_std).astype(np.float32)
    vision = ((vision - vis_mean) / vis_std).astype(np.float32)
    return np.clip(imu, -8.0, 8.0), np.clip(vision, -8.0, 8.0)


def training_weights(y, synthetic, synthetic_weight):
    counts = np.bincount(y, minlength=len(LABELS)).astype(np.float64)
    class_weight = len(y) / (len(LABELS) * np.maximum(counts, 1))
    weights = class_weight[y]
    weights[synthetic] *= synthetic_weight
    return weights.astype(np.float32)


def model_inputs(name, imu, vision):
    if name == "imu":
        return imu
    if name == "vision":
        return vision
    return [imu, vision]


def build_waist_models():
    """Keep the required CNN branches and add interpretable motion statistics."""
    import tensorflow as tf

    models = build_models()
    imu_input, imu_features = branch((128, 6), "imu")
    vision_input, vision_features = branch((64, 7), "vision")

    def motion_stats(x):
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=1) + 1e-6)
        change = tf.reduce_mean(tf.abs(x[:, 1:, :] - x[:, :-1, :]), axis=1)
        return tf.concat([rms, change], axis=1)

    imu_stats = tf.keras.layers.Lambda(
        motion_stats, output_shape=(12,), name="imu_motion_stats"
    )(imu_input)
    vision_stats = tf.keras.layers.Lambda(
        motion_stats, output_shape=(14,), name="vision_motion_stats"
    )(vision_input)
    fused = tf.keras.layers.Concatenate(name="fusion_features")(
        [imu_features, vision_features, imu_stats, vision_stats]
    )
    fused = tf.keras.layers.Dense(64, activation="relu", name="fusion_dense")(fused)
    fused = tf.keras.layers.Dropout(0.25, name="fusion_dropout")(fused)
    outputs = tf.keras.layers.Dense(
        len(LABELS), activation="softmax", name="fusion_output"
    )(fused)
    fusion = tf.keras.Model(
        [imu_input, vision_input], outputs, name="fitguard_fusion"
    )
    fusion.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    models["fusion"] = fusion
    return models


def metrics_from_probabilities(y, probability, trials):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    prediction = probability.argmax(axis=1)
    matrix = confusion_matrix(y, prediction, labels=range(len(LABELS)))
    grouped_probability = defaultdict(list)
    grouped_label = {}
    for index, trial in enumerate(trials):
        grouped_probability[trial].append(probability[index])
        grouped_label[trial] = int(y[index])
    trial_y, trial_prediction = [], []
    for trial in sorted(grouped_probability):
        trial_y.append(grouped_label[trial])
        trial_prediction.append(np.mean(grouped_probability[trial], axis=0).argmax())
    trial_matrix = confusion_matrix(
        trial_y, trial_prediction, labels=range(len(LABELS))
    )
    return {
        "window_accuracy": float(accuracy_score(y, prediction)),
        "window_macro_f1": float(f1_score(y, prediction, average="macro")),
        "window_confusion_matrix": matrix.tolist(),
        "trial_accuracy": float(accuracy_score(trial_y, trial_prediction)),
        "trial_macro_f1": float(f1_score(trial_y, trial_prediction, average="macro")),
        "trial_confusion_matrix": trial_matrix.tolist(),
        "test_windows": len(y),
        "test_trials": len(trial_y),
    }


def train_waist_models(
    real_root,
    augmented_root,
    output_dir,
    epochs=60,
    synthetic_weight=0.35,
    seed=20261001,
):
    import tensorflow as tf

    set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    real = load_root(real_root, expect_synthetic=False)
    augmented = load_root(augmented_root, expect_synthetic=True)
    real["imu"], real["vision"] = canonicalize_waist_windows(
        real["imu"], real["vision"]
    )
    augmented["imu"], augmented["vision"] = canonicalize_waist_windows(
        augmented["imu"], augmented["vision"]
    )
    subjects = sorted(np.unique(real["subject"]).tolist())
    if len(subjects) < 3:
        raise RuntimeError("At least three real subjects are required")

    report = {
        "labels": list(LABELS),
        "evaluation": "leave-one-subject-out; validation and test are real only",
        "real_subjects": subjects,
        "real_windows": len(real["y"]),
        "synthetic_training_pool_windows": len(augmented["y"]),
        "synthetic_sample_weight": synthetic_weight,
        "seed": seed,
        "preprocessing": "waist_centered_v1",
        "folds": [],
        "models": {},
    }
    all_predictions = {name: [] for name in ("imu", "vision", "fusion")}
    all_y = {name: [] for name in ("imu", "vision", "fusion")}
    all_trials = {name: [] for name in ("imu", "vision", "fusion")}
    epoch_history = {name: [] for name in ("imu", "vision", "fusion")}

    for fold_index, test_subject in enumerate(subjects):
        val_subject = subjects[(fold_index + 1) % len(subjects)]
        train_subjects = [s for s in subjects if s not in (test_subject, val_subject)]
        real_train = np.isin(real["subject"], train_subjects)
        real_val = real["subject"] == val_subject
        real_test = real["subject"] == test_subject
        aug_train = np.isin(augmented["subject"], train_subjects)

        train_imu = np.concatenate([real["imu"][real_train], augmented["imu"][aug_train]])
        train_vision = np.concatenate([
            real["vision"][real_train], augmented["vision"][aug_train]
        ])
        train_y = np.concatenate([real["y"][real_train], augmented["y"][aug_train]])
        train_synthetic = np.concatenate([
            np.zeros(real_train.sum(), dtype=bool),
            np.ones(aug_train.sum(), dtype=bool),
        ])
        stats = normalization(real["imu"][real_train], real["vision"][real_train])
        train_imu, train_vision = normalize(train_imu, train_vision, stats)
        val_imu, val_vision = normalize(
            real["imu"][real_val], real["vision"][real_val], stats
        )
        test_imu, test_vision = normalize(
            real["imu"][real_test], real["vision"][real_test], stats
        )
        weights = training_weights(train_y, train_synthetic, synthetic_weight)
        fold_result = {
            "test_subject": test_subject,
            "validation_subject": val_subject,
            "training_subjects": train_subjects,
            "real_training_windows": int(real_train.sum()),
            "synthetic_training_windows": int(aug_train.sum()),
            "real_validation_windows": int(real_val.sum()),
            "real_test_windows": int(real_test.sum()),
            "models": {},
        }

        for model_index, name in enumerate(("imu", "vision", "fusion")):
            tf.keras.backend.clear_session()
            set_seed(seed + fold_index * 10 + model_index)
            model = build_waist_models()[name]
            callback = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            )
            history = model.fit(
                model_inputs(name, train_imu, train_vision),
                train_y,
                sample_weight=weights,
                validation_data=(
                    model_inputs(name, val_imu, val_vision),
                    real["y"][real_val],
                ),
                epochs=epochs,
                batch_size=64,
                callbacks=[callback],
                verbose=0,
            )
            probability = model.predict(
                model_inputs(name, test_imu, test_vision), verbose=0
            )
            fold_metrics = metrics_from_probabilities(
                real["y"][real_test], probability, real["trial"][real_test]
            )
            fold_metrics["epochs_run"] = len(history.history["loss"])
            fold_result["models"][name] = fold_metrics
            epoch_history[name].append(len(history.history["loss"]))
            all_predictions[name].append(probability)
            all_y[name].append(real["y"][real_test])
            all_trials[name].append(real["trial"][real_test])
            print(
                f"fold={fold_index + 1}/5 test={test_subject} model={name} "
                f"window_acc={fold_metrics['window_accuracy']:.4f} "
                f"trial_acc={fold_metrics['trial_accuracy']:.4f}",
                flush=True,
            )
        report["folds"].append(fold_result)

    for name in ("imu", "vision", "fusion"):
        report["models"][name] = metrics_from_probabilities(
            np.concatenate(all_y[name]),
            np.concatenate(all_predictions[name]),
            np.concatenate(all_trials[name]),
        )
        report["models"][name]["fold_epochs"] = epoch_history[name]
        np.savetxt(
            output / f"loso_confusion_{name}.csv",
            np.asarray(report["models"][name]["window_confusion_matrix"]),
            delimiter=",",
            fmt="%d",
        )

    # Final deployable models use all real subjects and all labelled training-only
    # augmentation. The median LOSO stopping epoch avoids tuning on the test folds.
    stats = normalization(real["imu"], real["vision"])
    np.savez(
        output / "normalization.npz",
        imu_mean=stats[0], imu_std=stats[1], vision_mean=stats[2], vision_std=stats[3],
        preprocessing=np.asarray("waist_centered_v1"),
    )
    final_imu = np.concatenate([real["imu"], augmented["imu"]])
    final_vision = np.concatenate([real["vision"], augmented["vision"]])
    final_y = np.concatenate([real["y"], augmented["y"]])
    final_synthetic = np.concatenate([
        np.zeros(len(real["y"]), dtype=bool),
        np.ones(len(augmented["y"]), dtype=bool),
    ])
    final_imu, final_vision = normalize(final_imu, final_vision, stats)
    final_weights = training_weights(final_y, final_synthetic, synthetic_weight)
    report["final_models"] = {}
    for model_index, name in enumerate(("imu", "vision", "fusion")):
        tf.keras.backend.clear_session()
        set_seed(seed + 100 + model_index)
        model = build_waist_models()[name]
        final_epochs = max(5, int(round(np.median(epoch_history[name]))))
        model.fit(
            model_inputs(name, final_imu, final_vision),
            final_y,
            sample_weight=final_weights,
            epochs=final_epochs,
            batch_size=64,
            verbose=0,
        )
        model.save(output / f"fitguard_{name}.keras")
        export_tflite(model, output / f"fitguard_{name}.tflite")
        export_numpy_weights(model, name, output / f"fitguard_{name}_weights.npz")
        report["final_models"][name] = {
            "epochs": final_epochs,
            "keras_file": f"fitguard_{name}.keras",
            "tflite_file": f"fitguard_{name}.tflite",
            "numpy_weights_file": f"fitguard_{name}_weights.npz",
        }
        print(f"final model={name} epochs={final_epochs} exported", flush=True)

    (output / "evaluation_loso_real_subjects.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "MODEL_SCOPE.txt").write_text(
        "Evaluation uses untouched real W01-W05 test subjects in five LOSO folds.\n"
        "Synthetic variants are training-only and never enter validation or test sets.\n"
        "This remains an engineering prototype, not a clinical diagnostic model.\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True)
    parser.add_argument("--augmented", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--synthetic-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    result = train_waist_models(
        args.real,
        args.augmented,
        args.output,
        epochs=args.epochs,
        synthetic_weight=args.synthetic_weight,
        seed=args.seed,
    )
    print(json.dumps(result["models"], ensure_ascii=False, indent=2))
