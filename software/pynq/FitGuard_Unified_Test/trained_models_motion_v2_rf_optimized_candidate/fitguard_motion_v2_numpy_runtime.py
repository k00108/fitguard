"""Pure-NumPy feature extraction and Random Forest inference for PYNQ."""

from __future__ import annotations

from pathlib import Path

import numpy as np


LABELS = ("Stable", "Mild Abnormality", "Marked Abnormality")


def _robust_stats(x):
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


def _spectral_features(x):
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(axis=0, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=0)) ** 2
    power = power[1:]
    if len(power) == 0:
        return np.zeros(x.shape[1] * 5, dtype=np.float64)
    total = power.sum(axis=0) + 1e-12
    n = len(power)
    cuts = [0, max(1, n // 8), max(2, n // 4), max(3, n // 2), n]
    bands = [power[left:right].sum(axis=0) / total for left, right in zip(cuts[:-1], cuts[1:])]
    dominant = np.argmax(power, axis=0) / max(n - 1, 1)
    return np.concatenate(bands + [dominant])


def _correlation_features(x):
    corr = np.corrcoef(np.asarray(x, dtype=np.float64), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr[np.triu_indices(corr.shape[0], 1)]


def extract_features(imu, vision, vision_aux):
    imu = np.asarray(imu, dtype=np.float64).copy()
    vision = np.asarray(vision, dtype=np.float64).copy()
    aux = np.asarray(vision_aux, dtype=np.float64).copy()
    imu_centered = imu - np.median(imu, axis=0, keepdims=True)
    valid = aux[:, 0] > 0.5
    if valid.any():
        baseline = np.median(vision[valid, :5], axis=0)
        vision[valid, :2] -= baseline[:2]
        vision[valid, 2:5] = vision[valid, 2:5] / np.maximum(baseline[2:5], 1e-4) - 1.0
    vision[~valid] = 0.0
    acc_norm = np.linalg.norm(imu_centered[:, :3], axis=1, keepdims=True)
    gyro_norm = np.linalg.norm(imu_centered[:, 3:], axis=1, keepdims=True)
    imu_full = np.concatenate([imu_centered, acc_norm, gyro_norm], axis=1)
    if valid.any():
        vis_valid = vision[valid]
        aux_valid = aux[valid]
    else:
        vis_valid = np.zeros((2, vision.shape[1]))
        aux_valid = np.zeros((2, aux.shape[1]))
    result = np.concatenate(
        [
            _robust_stats(imu_full),
            _spectral_features(imu_full),
            _correlation_features(imu_centered),
            _robust_stats(vis_valid),
            _spectral_features(vis_valid),
            _robust_stats(aux_valid),
            np.asarray([
                valid.mean(), aux_valid[:, 1].mean(), aux_valid[:, 1].std(),
                aux_valid[:, 2].mean(), aux_valid[:, 2].std(),
            ]),
        ]
    ).astype(np.float32)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


class NumpyRandomForest:
    def __init__(self, model_path):
        data = np.load(Path(model_path), allow_pickle=False)
        self.children_left = data["children_left"]
        self.children_right = data["children_right"]
        self.feature = data["feature"]
        self.threshold = data["threshold"]
        self.value = data["value"]
        self.offsets = data["offsets"]
        self.labels = tuple(data["labels"].tolist())
        self.feature_version = str(data["feature_version"])

    def predict_proba_features(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        output = np.zeros((len(x), len(self.labels)), dtype=np.float64)
        for row_index, row in enumerate(x):
            total = np.zeros(len(self.labels), dtype=np.float64)
            for tree_index in range(len(self.offsets) - 1):
                node = int(self.offsets[tree_index])
                while self.children_left[node] != -1:
                    if row[self.feature[node]] <= self.threshold[node]:
                        node = int(self.children_left[node])
                    else:
                        node = int(self.children_right[node])
                leaf = self.value[node]
                total += leaf / max(float(leaf.sum()), 1e-12)
            output[row_index] = total / max(len(self.offsets) - 1, 1)
        output /= np.maximum(output.sum(axis=1, keepdims=True), 1e-12)
        return output.astype(np.float32)

    def predict_window(self, imu, vision, vision_aux):
        probability = self.predict_proba_features(extract_features(imu, vision, vision_aux))[0]
        index = int(np.argmax(probability))
        return self.labels[index], probability
