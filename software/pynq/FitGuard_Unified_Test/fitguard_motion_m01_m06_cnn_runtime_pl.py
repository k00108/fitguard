"""CNN V8 runtime: PL Conv1D branches with automatic NumPy fallback."""

import time
import warnings

import numpy as np

from fitguard_motion_m01_m06_cnn_runtime import (
    LABELS,
    _dense_relu,
    coherent_turn_present,
)
from fitguard_motion_m01_m06_cnn_runtime_v7 import RangeFusionCNNV7


class RangeFusionCNNPL(RangeFusionCNNV7):
    def __init__(self, model_dir, bitstream=None, backend="auto", download=False):
        super().__init__(model_dir, backend="numpy")
        requested = str(backend).lower()
        if requested not in {"auto", "pl", "numpy"}:
            raise ValueError("backend must be 'auto', 'pl', or 'numpy'")
        self.backend_requested = requested
        self.backend_name = "numpy_ps"
        self.pl_acceleration_active = False
        self._pl = None
        self._pl_failure = None
        if requested != "numpy":
            if bitstream is None:
                self._pl_failure = "no CNN-PL bitstream supplied"
                if requested == "pl":
                    raise ValueError(self._pl_failure)
            else:
                try:
                    from fitguard_cnn_pl_backend import FitGuardCnnPlBackend
                    self._pl = FitGuardCnnPlBackend(bitstream, download=download)
                    self.backend_name = "pl_cnn_pending"
                except Exception as exc:
                    self._pl_failure = f"PL backend setup failed: {exc}"
                    if requested == "pl":
                        raise

    def prime(self):
        """Attach and exercise the PL DMA before the first live sensor window.

        Priming moves the one-time DMA/overlay startup cost out of the first
        classification and gives the live run an immediate, explicit backend
        readiness result.  In ``auto`` mode a failed prime preserves the
        documented NumPy fallback; strict ``pl`` mode fails loudly.
        """
        if self._pl is None:
            return self.backend_status()
        try:
            gap = self._pl.infer(
                np.zeros((128, 8), dtype=np.float32),
                np.zeros((64, 7), dtype=np.float32),
            )
            gap = np.asarray(gap, dtype=np.float32)
            if gap.shape != (160,) or not np.isfinite(gap).all():
                raise RuntimeError(f"PL prime returned invalid GAP tensor {gap.shape}")
            self.backend_name = "pl_cnn_conv1d"
            self.pl_acceleration_active = True
            return self.backend_status()
        except Exception as exc:
            self._pl_failure = f"PL prime failed; NumPy fallback active: {exc}"
            self.backend_name = "numpy_ps_fallback"
            self.pl_acceleration_active = False
            failed_pl, self._pl = self._pl, None
            if failed_pl is not None:
                try:
                    failed_pl.close()
                except Exception as close_exc:
                    self._pl_failure += f"; PL cleanup also failed: {close_exc}"
            if self.backend_requested == "pl":
                raise RuntimeError(self._pl_failure) from exc
            warnings.warn(self._pl_failure, RuntimeWarning)
            return self.backend_status()

    def _finish_from_gap(self, gap, ranges, turn):
        gap = np.asarray(gap, dtype=np.float32)
        if gap.shape != (160,):
            raise ValueError(f"PL returned {gap.shape}, expected (160,)")
        fi = _dense_relu(gap[:96], self.w["imu_dense_0"], self.w["imu_dense_1"])
        fv = _dense_relu(gap[96:], self.w["vision_dense_0"], self.w["vision_dense_1"])
        fr = _dense_relu(ranges, self.w["range_dense1_0"], self.w["range_dense1_1"])
        fr = _dense_relu(fr, self.w["range_dense2_0"], self.w["range_dense2_1"])
        x = np.concatenate([fi, fv, fr])
        x = _dense_relu(x, self.w["fusion_dense1_0"], self.w["fusion_dense1_1"])
        x = _dense_relu(x, self.w["fusion_dense2_0"], self.w["fusion_dense2_1"])
        logits = x @ self.w["fusion_output_0"] + self.w["fusion_output_1"]
        logits -= logits.max()
        probability = np.exp(logits)
        probability = (probability / probability.sum()).astype(np.float32)
        if turn and int(np.argmax(probability)) == 2 and float(probability[2] - probability[1]) < self.turn_guard_margin:
            probability[1], probability[2] = probability[2], probability[1]
        return probability

    def predict(self, imu, vision, aux, ablate=None):
        if ablate is not None or self._pl is None:
            return super().predict(imu, vision, aux, ablate=ablate)
        imu8, vision7, ranges, turn = self._prepare(imu, vision, aux)
        try:
            gap = self._pl.infer(imu8, vision7)
            probability = self._finish_from_gap(gap, ranges, turn)
            self.backend_name = "pl_cnn_conv1d"
            self.pl_acceleration_active = True
            return probability
        except Exception as exc:
            self._pl_failure = f"PL inference failed; NumPy fallback active: {exc}"
            self.backend_name = "numpy_ps_fallback"
            self.pl_acceleration_active = False
            failed_pl, self._pl = self._pl, None
            if failed_pl is not None:
                try:
                    failed_pl.close()
                except Exception as close_exc:
                    self._pl_failure += f"; PL cleanup also failed: {close_exc}"
            if self.backend_requested == "pl":
                raise RuntimeError(self._pl_failure) from exc
            warnings.warn(self._pl_failure, RuntimeWarning)
            return super().predict(imu, vision, aux)

    def predict_profiled(self, imu, vision, aux):
        started = time.monotonic()
        probability = self.predict(imu, vision, aux)
        return probability, {
            "backend": self.backend_name,
            "pl_acceleration_active": self.pl_acceleration_active,
            "inference_ms": float((time.monotonic() - started) * 1000.0),
        }

    def backend_status(self):
        status = {
            "requested": self.backend_requested,
            "active": self.backend_name,
            "pl_acceleration_active": self.pl_acceleration_active,
            "fallback_reason": self._pl_failure,
        }
        if self._pl is not None:
            status["pl"] = self._pl.status()
        return status

    def close(self):
        """Release physically contiguous DMA buffers before a notebook rerun."""
        if self._pl is not None:
            self._pl.close()
        self._pl = None
        self.pl_acceleration_active = False


__all__ = ["LABELS", "RangeFusionCNNPL"]
