"""V7 runtime wrapper for the validated M01-M06 CNN.

The V6 NumPy implementation remains untouched and is the numerical reference.
This wrapper adds input-ablation support, backend reporting, and profiling hooks
without changing the trained weights or the normal prediction path.
"""

import time

import numpy as np

from fitguard_motion_m01_m06_cnn_runtime import (
    LABELS,
    RangeFusionCNN,
    _dense_relu,
    canonicalize,
    coherent_turn_present,
)


class RangeFusionCNNV7(RangeFusionCNN):
    """Backward-compatible V6 model with diagnostics and safe PS fallback."""

    def __init__(self, model_dir, backend="auto"):
        super().__init__(model_dir)
        requested = str(backend).lower()
        if requested not in {"auto", "numpy"}:
            raise ValueError(
                "No verified PL CNN bitstream is installed. Use backend='auto' "
                "or backend='numpy' until the HLS accelerator passes golden tests."
            )
        self.backend_requested = requested
        self.backend_name = "numpy_ps"
        self.pl_acceleration_active = False

    def _prepare(self, imu, vision, aux):
        turn = coherent_turn_present(vision, aux) if self.turn_guard else False
        imu8, vision7, ranges = canonicalize(imu, vision, aux)
        if self.no_excursion:
            ranges[7] = 0.0
        imu8 = np.clip((imu8 - self.imu_mean) / self.imu_std, -8, 8)
        vision7 = np.clip((vision7 - self.vis_mean) / self.vis_std, -8, 8)
        ranges = np.clip((ranges - self.range_mean) / self.range_std, -8, 8)
        return imu8, vision7, ranges, turn

    def _forward(self, imu8, vision7, ranges, turn):
        fi = self._branch(imu8, "imu")
        fv = self._branch(vision7, "vision")
        fr = _dense_relu(
            ranges, self.w["range_dense1_0"], self.w["range_dense1_1"]
        )
        fr = _dense_relu(
            fr, self.w["range_dense2_0"], self.w["range_dense2_1"]
        )
        x = np.concatenate([fi, fv, fr])
        x = _dense_relu(x, self.w["fusion_dense1_0"], self.w["fusion_dense1_1"])
        x = _dense_relu(x, self.w["fusion_dense2_0"], self.w["fusion_dense2_1"])
        logits = x @ self.w["fusion_output_0"] + self.w["fusion_output_1"]
        logits -= logits.max()
        probability = np.exp(logits)
        probability = (probability / probability.sum()).astype(np.float32)
        if (
            turn
            and int(np.argmax(probability)) == 2
            and float(probability[2] - probability[1]) < self.turn_guard_margin
        ):
            probability[1], probability[2] = probability[2], probability[1]
        return probability

    def predict(self, imu, vision, aux, ablate=None):
        """Predict with optional diagnostic input ablation.

        Ablation is performed after normalization by replacing a branch with
        its training-set mean (zero in normalized coordinates). This is for
        dependency analysis only and never runs in the live default path.
        """
        imu8, vision7, ranges, turn = self._prepare(imu, vision, aux)
        if ablate is not None:
            ablate = str(ablate).lower()
            if ablate == "imu":
                imu8 = np.zeros_like(imu8)
                ranges[:6] = 0.0
            elif ablate == "vision":
                vision7 = np.zeros_like(vision7)
                ranges[6:] = 0.0
                turn = False
            else:
                raise ValueError("ablate must be None, 'imu', or 'vision'")
        return self._forward(imu8, vision7, ranges, turn)

    def predict_profiled(self, imu, vision, aux):
        started = time.monotonic()
        probability = self.predict(imu, vision, aux)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return probability, {
            "backend": self.backend_name,
            "pl_acceleration_active": self.pl_acceleration_active,
            "inference_ms": float(elapsed_ms),
        }

    def backend_status(self):
        return {
            "requested": self.backend_requested,
            "active": self.backend_name,
            "pl_acceleration_active": self.pl_acceleration_active,
            "note": (
                "CNN V7 currently uses the verified NumPy PS reference. "
                "PL must not be claimed until an HLS bitstream passes the "
                "golden probability and latency checks."
            ),
        }


__all__ = ["LABELS", "RangeFusionCNNV7"]
