"""Train and compare FitGuard IMU, vision, and late-fusion 1D CNNs."""

import argparse
import json
import random
from pathlib import Path

import numpy as np


LABELS = ("Stable", "Mild Abnormality", "Marked Abnormality")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def load_dataset(root):
    imu, vision, labels, subjects, trials = [], [], [], [], []
    synthetic_detected = False
    files = sorted(Path(root).glob("*/*/trial_*/windows.npz"))
    for path in files:
        data = np.load(path, allow_pickle=False)
        if "synthetic" in data.files and np.asarray(data["synthetic"]).any():
            synthetic_detected = True
        trial_labels = data["label"].astype(str)
        trial_subjects = data["subject"].astype(str)
        if len(trial_labels) == 0 or trial_subjects[0] == "SMOKE":
            continue
        if trial_labels[0] not in LABEL_TO_INDEX:
            raise ValueError(f"Unknown label {trial_labels[0]!r} in {path}")
        count = data["imu"].shape[0]
        if data["imu"].shape != (count, 128, 6):
            raise ValueError(f"Bad IMU shape in {path}: {data['imu'].shape}")
        if data["vision"].shape != (count, 64, 7):
            raise ValueError(f"Bad vision shape in {path}: {data['vision'].shape}")
        imu.append(data["imu"].astype(np.float32))
        vision.append(data["vision"].astype(np.float32))
        labels.extend([LABEL_TO_INDEX[value] for value in trial_labels])
        subjects.extend(trial_subjects.tolist())
        trials.extend([str(path.parent)] * count)
    if not imu:
        raise RuntimeError(f"No formal windows.npz files found below {root}")
    return (
        np.concatenate(imu),
        np.concatenate(vision),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subjects),
        np.asarray(trials),
        synthetic_detected,
    )


def split_indices(y, subjects, trials, seed):
    from sklearn.model_selection import GroupShuffleSplit

    # Prefer subject-independent evaluation.  Until at least three subjects
    # exist, keep whole trials together so overlapping windows never leak.
    unique_subjects = np.unique(subjects)
    groups = subjects if len(unique_subjects) >= 3 else trials
    split_kind = "subject-independent" if len(unique_subjects) >= 3 else "trial-independent"
    if len(np.unique(groups)) < 5:
        raise RuntimeError(
            "At least five subjects/trials are needed for train/validation/test splitting"
        )
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_val, test = next(outer.split(np.zeros(len(y)), y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed + 1)
    train_rel, val_rel = next(
        inner.split(np.zeros(len(train_val)), y[train_val], groups[train_val])
    )
    train, val = train_val[train_rel], train_val[val_rel]
    if set(np.unique(y[train])) != set(range(len(LABELS))):
        raise RuntimeError("Training split does not contain all three labels; collect more trials")
    return train, val, test, split_kind


def channel_normalize(train, val, test):
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)
    return (
        ((train - mean) / std).astype(np.float32),
        ((val - mean) / std).astype(np.float32),
        ((test - mean) / std).astype(np.float32),
        mean.reshape(-1),
        std.reshape(-1),
    )


def branch(input_shape, name):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=input_shape, name=f"{name}_input")
    x = tf.keras.layers.Conv1D(
        32, 5, padding="same", activation="relu", name=f"{name}_conv1"
    )(inputs)
    x = tf.keras.layers.MaxPooling1D(2, name=f"{name}_pool")(x)
    x = tf.keras.layers.Conv1D(
        64, 3, padding="same", activation="relu", name=f"{name}_conv2"
    )(x)
    x = tf.keras.layers.GlobalAveragePooling1D(name=f"{name}_gap")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name=f"{name}_dense")(x)
    return inputs, x


def build_models():
    import tensorflow as tf

    models = {}
    for name, shape in (("imu", (128, 6)), ("vision", (64, 7))):
        inputs, features = branch(shape, name)
        outputs = tf.keras.layers.Dense(
            len(LABELS), activation="softmax", name=f"{name}_output"
        )(features)
        models[name] = tf.keras.Model(inputs, outputs, name=f"fitguard_{name}")

    imu_input, imu_features = branch((128, 6), "imu")
    vision_input, vision_features = branch((64, 7), "vision")
    fused = tf.keras.layers.Concatenate()([imu_features, vision_features])
    fused = tf.keras.layers.Dense(64, activation="relu", name="fusion_dense")(fused)
    fused = tf.keras.layers.Dropout(0.25)(fused)
    outputs = tf.keras.layers.Dense(
        len(LABELS), activation="softmax", name="fusion_output"
    )(fused)
    models["fusion"] = tf.keras.Model(
        [imu_input, vision_input], outputs, name="fitguard_fusion"
    )
    for model in models.values():
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return models


def export_tflite(model, path):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    path.write_bytes(converter.convert())


def export_numpy_weights(model, model_name, path):
    """Export fixed CNN layers for dependency-free NumPy inference on PYNQ."""
    arrays = {"model_name": np.asarray(model_name), "labels": np.asarray(LABELS)}
    branches = (model_name,) if model_name in ("imu", "vision") else ("imu", "vision")
    for branch_name in branches:
        for layer_suffix in ("conv1", "conv2", "dense"):
            weights, bias = model.get_layer(f"{branch_name}_{layer_suffix}").get_weights()
            arrays[f"{branch_name}_{layer_suffix}_w"] = weights
            arrays[f"{branch_name}_{layer_suffix}_b"] = bias
    if model_name == "fusion":
        for layer_name in ("fusion_dense", "fusion_output"):
            weights, bias = model.get_layer(layer_name).get_weights()
            arrays[f"{layer_name}_w"] = weights
            arrays[f"{layer_name}_b"] = bias
    else:
        weights, bias = model.get_layer(f"{model_name}_output").get_weights()
        arrays[f"{model_name}_output_w"] = weights
        arrays[f"{model_name}_output_b"] = bias
    np.savez_compressed(path, **arrays)


def train_all(dataset_root="dataset", output_dir="trained_models", epochs=80, seed=42):
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    set_seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    imu, vision, y, subjects, trials, synthetic_detected = load_dataset(dataset_root)
    train, val, test, split_kind = split_indices(y, subjects, trials, seed)
    imu_train, imu_val, imu_test, imu_mean, imu_std = channel_normalize(
        imu[train], imu[val], imu[test]
    )
    vis_train, vis_val, vis_test, vis_mean, vis_std = channel_normalize(
        vision[train], vision[val], vision[test]
    )
    np.savez(
        output / "normalization.npz",
        imu_mean=imu_mean,
        imu_std=imu_std,
        vision_mean=vis_mean,
        vision_std=vis_std,
    )

    inputs = {
        "imu": (imu_train, imu_val, imu_test),
        "vision": (vis_train, vis_val, vis_test),
        "fusion": (
            [imu_train, vis_train],
            [imu_val, vis_val],
            [imu_test, vis_test],
        ),
    }
    results = {
        "labels": list(LABELS),
        "dataset_root": str(Path(dataset_root).resolve()),
        "synthetic_data": synthetic_detected,
        "result_status": (
            "DEMO ONLY - not a real-subject generalization result"
            if synthetic_detected else "REAL DATA EVALUATION"
        ),
        "split_kind": split_kind,
        "samples": {"train": len(train), "validation": len(val), "test": len(test)},
        "subjects": sorted(np.unique(subjects).tolist()),
        "models": {},
    }
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]
    for name, model in build_models().items():
        x_train, x_val, x_test = inputs[name]
        history = model.fit(
            x_train,
            y[train],
            validation_data=(x_val, y[val]),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=2,
        )
        probabilities = model.predict(x_test, verbose=0)
        prediction = probabilities.argmax(axis=1)
        matrix = confusion_matrix(y[test], prediction, labels=range(len(LABELS)))
        report = classification_report(
            y[test], prediction, labels=range(len(LABELS)), target_names=LABELS,
            output_dict=True, zero_division=0,
        )
        model.save(output / f"fitguard_{name}.keras")
        export_tflite(model, output / f"fitguard_{name}.tflite")
        export_numpy_weights(
            model, name, output / f"fitguard_{name}_weights.npz"
        )
        np.savetxt(output / f"confusion_{name}.csv", matrix, delimiter=",", fmt="%d")
        results["models"][name] = {
            "accuracy": float(accuracy_score(y[test], prediction)),
            "macro_f1": float(f1_score(y[test], prediction, average="macro")),
            "confusion_matrix": matrix.tolist(),
            "classification_report": report,
            "epochs_run": len(history.history["loss"]),
            "tflite_bytes": (output / f"fitguard_{name}.tflite").stat().st_size,
        }
        print(name, results["models"][name]["accuracy"], results["models"][name]["macro_f1"])

    (output / "evaluation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if synthetic_detected:
        (output / "DEMO_ONLY.txt").write_text(
            "Models were trained and evaluated on augmented data derived from one "
            "real trial per class. Do not report these metrics as clinical or "
            "cross-subject accuracy.\n",
            encoding="utf-8",
        )
    print("Training outputs saved in", output.resolve())
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--output", default="trained_models")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_all(args.dataset, args.output, args.epochs, args.seed)
