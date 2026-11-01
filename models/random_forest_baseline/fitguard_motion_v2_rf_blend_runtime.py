"""Scikit-learn-free runtime for the FitGuard optimized RF blend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fitguard_motion_v2_numpy_runtime import NumpyRandomForest, extract_features


class NumpyRandomForestBlend:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "rf_blend_config.json").read_text(encoding="utf-8"))
        self.labels = tuple(config["labels"])
        self.base_weight = float(config["base_weight"])
        self.robust_weight = float(config["robust_weight"])
        self.base = NumpyRandomForest(model_dir / config["base_model"])
        self.robust = NumpyRandomForest(model_dir / config["robust_model"])

    def predict_proba_features(self, features):
        features = np.asarray(features, dtype=np.float32)
        return (
            self.base_weight * self.base.predict_proba_features(features)
            + self.robust_weight * self.robust.predict_proba_features(features)
        )

    def predict_window(self, imu, vision, vision_aux):
        probability = self.predict_proba_features(
            extract_features(imu, vision, vision_aux)
        )[0]
        index = int(np.argmax(probability))
        return {
            "label": self.labels[index],
            "confidence": float(probability[index]),
            "probabilities": probability.tolist(),
        }
