"""PYNQ DMA backend for the FitGuard six-Conv1D PL accelerator."""

from pathlib import Path
import time

import numpy as np


class FitGuardCnnPlBackend:
    INPUT_FLOATS = 1472
    OUTPUT_FLOATS = 160

    def __init__(self, bitstream, dma_name="axi_dma_cnn_0", download=False):
        self.bitstream = str(Path(bitstream).resolve())
        self.dma_name = str(dma_name)
        self.download = bool(download)
        self.overlay = None
        self.dma = None
        self.input_buffer = None
        self.output_buffer = None
        self.calls = 0
        self.last_ms = None

    def _attach(self):
        if self.dma is not None:
            return
        from pynq import Overlay, allocate

        self.overlay = Overlay(self.bitstream, download=self.download)
        if not hasattr(self.overlay, self.dma_name):
            raise RuntimeError(f"HWH has no DMA named {self.dma_name}")
        # The streaming-only HLS block has no AXI-Lite address and therefore
        # need not appear as a Python attribute in every PYNQ release. Its
        # presence is verified from the HWH at packaging time; DMA is the only
        # object software needs to drive at runtime.
        self.dma = getattr(self.overlay, self.dma_name)
        self.input_buffer = allocate(shape=(self.INPUT_FLOATS,), dtype=np.float32)
        self.output_buffer = allocate(shape=(self.OUTPUT_FLOATS,), dtype=np.float32)

    def infer(self, imu_normalized, vision_normalized):
        imu = np.asarray(imu_normalized, dtype=np.float32)
        vision = np.asarray(vision_normalized, dtype=np.float32)
        if imu.shape != (128, 8):
            raise ValueError(f"expected IMU (128,8), got {imu.shape}")
        if vision.shape != (64, 7):
            raise ValueError(f"expected vision (64,7), got {vision.shape}")
        self._attach()
        packet = np.concatenate((imu.reshape(-1), vision.reshape(-1)))
        self.input_buffer[:] = packet
        self.input_buffer.flush()
        started = time.monotonic()
        # Arm receive first so an ap_ctrl_none producer cannot back-pressure
        # before the S2MM channel is ready.
        self.dma.recvchannel.transfer(self.output_buffer)
        self.dma.sendchannel.transfer(self.input_buffer)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        self.output_buffer.invalidate()
        self.last_ms = (time.monotonic() - started) * 1000.0
        self.calls += 1
        return np.asarray(self.output_buffer, dtype=np.float32).copy()

    def status(self):
        return {
            "active": self.dma is not None,
            "bitstream": self.bitstream,
            "dma": self.dma_name,
            "calls": int(self.calls),
            "last_ms": None if self.last_ms is None else float(self.last_ms),
        }

    def close(self):
        for buffer in (self.input_buffer, self.output_buffer):
            if buffer is not None:
                buffer.freebuffer()
        self.input_buffer = self.output_buffer = None
        self.dma = None


__all__ = ["FitGuardCnnPlBackend"]
