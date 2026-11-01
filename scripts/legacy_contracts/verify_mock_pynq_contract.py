#!/usr/bin/env python3
"""Exercise the PL backend against a strict in-memory PYNQ/DMA simulator."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np


class MockBuffer(np.ndarray):
    def __new__(cls, shape, dtype):
        obj = np.zeros(shape, dtype=dtype).view(cls)
        obj.flushed = False
        obj.invalidated = False
        obj.freed = False
        return obj

    def flush(self):
        self.flushed = True

    def invalidate(self):
        self.invalidated = True

    def freebuffer(self):
        self.freed = True


class Channel:
    def __init__(self, dma, direction):
        self.dma = dma
        self.direction = direction

    def transfer(self, buffer):
        self.dma.events.append(f"{self.direction}_transfer")
        if self.direction == "recv":
            self.dma.output_buffer = buffer
            return
        if self.dma.output_buffer is None:
            raise RuntimeError("S2MM must be armed before MM2S")
        if not buffer.flushed:
            raise RuntimeError("input buffer was not flushed")
        packet = np.asarray(buffer, dtype=np.float32)
        distances = np.max(np.abs(self.dma.inputs - packet[None, :]), axis=1)
        index = int(np.argmin(distances))
        if float(distances[index]) != 0.0:
            raise RuntimeError("DMA received a packet outside the golden set")
        self.dma.output_buffer[:] = self.dma.outputs[index]

    def wait(self):
        self.dma.events.append(f"{self.direction}_wait")


class MockDma:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.events: list[str] = []
        self.output_buffer = None
        self.recvchannel = Channel(self, "recv")
        self.sendchannel = Channel(self, "send")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-dir", required=True, type=Path)
    parser.add_argument("--bitstream", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    golden = np.load(args.golden, allow_pickle=False)
    inputs = np.asarray(golden["inputs"], dtype=np.float32)
    outputs = np.asarray(golden["outputs"], dtype=np.float32)
    dma = MockDma(inputs, outputs)
    overlays: list[object] = []

    class Overlay:
        def __init__(self, bitstream, download=False):
            self.bitstream_name = str(bitstream)
            self.download = bool(download)
            self.axi_dma_cnn_0 = dma
            overlays.append(self)

    pynq = types.ModuleType("pynq")
    pynq.Overlay = Overlay
    pynq.allocate = lambda shape, dtype: MockBuffer(shape, dtype)
    sys.modules["pynq"] = pynq
    sys.path.insert(0, str(args.backend_dir.resolve()))
    backend_module = importlib.import_module("fitguard_cnn_pl_backend")
    backend = backend_module.FitGuardCnnPlBackend(args.bitstream, download=True)

    errors = []
    try:
        for packet, expected in zip(inputs, outputs):
            actual = backend.infer(packet[:1024].reshape(128, 8), packet[1024:].reshape(64, 7))
            errors.append(float(np.max(np.abs(actual - expected))))
        input_buffer, output_buffer = backend.input_buffer, backend.output_buffer
        status_before_close = backend.status()
    finally:
        backend.close()

    per_call = ["recv_transfer", "send_transfer", "send_wait", "recv_wait"]
    ordering_ok = all(dma.events[i:i + 4] == per_call for i in range(0, len(dma.events), 4))
    checks = {
        "all_golden_cases_executed": len(errors) == len(inputs) == 18,
        "outputs_exact": bool(errors and max(errors) == 0.0),
        "receive_armed_before_send": ordering_ok,
        "overlay_download_requested": bool(overlays and overlays[0].download),
        "input_flushed": bool(input_buffer.flushed),
        "output_invalidated": bool(output_buffer.invalidated),
        "buffers_freed": bool(input_buffer.freed and output_buffer.freed),
        "call_counter_correct": status_before_close["calls"] == len(inputs),
    }
    report = {
        "cases": len(errors),
        "max_abs_error": max(errors) if errors else None,
        "dma_event_count": len(dma.events),
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
