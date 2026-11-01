"""FitGuard wireless IMU received through an ESP32 BLE-to-USB gateway.

The module uses only Python's POSIX standard library plus NumPy already on the
PYNQ image.  It deliberately avoids pyserial and leaves the wired AXI-IIC path
untouched.  Protocol: AA 55, type, length, payload, CRC16-CCITT-FALSE.
"""

from collections import deque
from dataclasses import asdict
import os
import select
import struct
import termios
import time

import numpy as np

from fitguard_mpu6050 import ImuSample


SYNC = b"\xAA\x55"
TYPE_IMU = 0x01
TYPE_IMU_BATCH = 0x10
RECORD_BYTES = 19
MAX_PAYLOAD = 190
ACCEL_SCALE = 16384.0
GYRO_SCALE = 131.0


def crc16_ccitt_false(data):
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def build_frame(frame_type, payload):
    body = bytes((int(frame_type), len(payload))) + bytes(payload)
    crc = crc16_ccitt_false(body)
    return SYNC + body + struct.pack("<H", crc)


class FrameParser:
    def __init__(self):
        self.buffer = bytearray()
        self.bytes_received = 0
        self.frames = 0
        self.bad_crc = 0
        self.bad_length = 0

    def feed(self, data, arrival_time=None):
        arrival_time = time.monotonic() if arrival_time is None else float(arrival_time)
        self.bytes_received += len(data)
        self.buffer.extend(data)
        records = []
        while True:
            position = self.buffer.find(SYNC)
            if position < 0:
                if len(self.buffer) > 1:
                    del self.buffer[:-1]
                break
            if position:
                del self.buffer[:position]
            if len(self.buffer) < 4:
                break
            payload_length = self.buffer[3]
            if payload_length == 0 or payload_length > MAX_PAYLOAD:
                self.bad_length += 1
                del self.buffer[:2]
                continue
            total = payload_length + 6
            if len(self.buffer) < total:
                break
            expected = struct.unpack_from("<H", self.buffer, 4 + payload_length)[0]
            actual = crc16_ccitt_false(bytes(self.buffer[2:4 + payload_length]))
            if expected != actual:
                self.bad_crc += 1
                del self.buffer[:total]
                continue
            frame_type = self.buffer[2]
            payload = bytes(self.buffer[4:4 + payload_length])
            self.frames += 1
            if frame_type in (TYPE_IMU, TYPE_IMU_BATCH):
                if len(payload) % RECORD_BYTES:
                    self.bad_length += 1
                else:
                    for start in range(0, len(payload), RECORD_BYTES):
                        sequence, sender_us, ax, ay, az, gx, gy, gz, temperature = struct.unpack_from(
                            "<BIhhhhhhh", payload, start
                        )
                        records.append({
                            "sequence": sequence,
                            "sender_us": sender_us,
                            "arrival_time": arrival_time,
                            "raw": (ax, ay, az, gx, gy, gz),
                            "temperature_raw": temperature,
                        })
            del self.buffer[:total]
        return records


class SenderClock:
    def __init__(self):
        self.wrap = 0
        self.last_raw = None
        self.offsets = deque(maxlen=250)
        self.offset = None

    def unwrap(self, raw_us):
        if self.last_raw is not None and raw_us < self.last_raw and self.last_raw - raw_us > 0x80000000:
            self.wrap += 1 << 32
        self.last_raw = raw_us
        return raw_us + self.wrap

    def map_batch(self, records):
        expanded = []
        for record in records:
            sender64 = self.unwrap(record["sender_us"])
            expanded.append((record, sender64))
            self.offsets.append(record["arrival_time"] - sender64 / 1e6)
        if self.offsets:
            self.offset = min(self.offsets)
        return [(record, sender64 / 1e6 + self.offset, sender64) for record, sender64 in expanded]


class GatewayMPU6050:
    """MPU6050-compatible source backed by an ESP32 USB serial gateway."""

    def __init__(self, port="/dev/ttyUSB0", baud=921600, timeout_s=0.25, queue_size=512):
        self.port = str(port)
        self.baud = int(baud)
        self.timeout_s = float(timeout_s)
        self.queue_size = int(queue_size)
        self.fd = None
        self.parser = FrameParser()
        self.clock = SenderClock()
        self.samples = deque(maxlen=self.queue_size)
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        self.last_sequence = None
        self.lost_samples = 0
        self.out_of_order_samples = 0
        self.queue_drops = 0
        self.first_sender64 = None
        self.last_sender64 = None
        self.total_samples = 0
        self.last_arrival_time = None

    @staticmethod
    def available_ports():
        import glob
        return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))

    def _open_port(self):
        if self.baud != 921600 or not hasattr(termios, "B921600"):
            raise RuntimeError("This gateway backend requires termios B921600 support")
        self.fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attributes = termios.tcgetattr(self.fd)
        attributes[0] = 0
        attributes[1] = 0
        attributes[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        attributes[3] = 0
        attributes[4] = termios.B921600
        attributes[5] = termios.B921600
        attributes[6][termios.VMIN] = 0
        attributes[6][termios.VTIME] = 0
        termios.tcflush(self.fd, termios.TCIOFLUSH)
        termios.tcsetattr(self.fd, termios.TCSANOW, attributes)

    def _accept_records(self, records):
        if not records:
            return
        mapped = self.clock.map_batch(records)
        for record, timestamp, sender64 in mapped:
            sequence = record["sequence"]
            if self.last_sequence is not None:
                delta = (sequence - self.last_sequence) & 0xFF
                if delta == 0 or delta > 0x80:
                    self.out_of_order_samples += 1
                elif delta > 1:
                    self.lost_samples += delta - 1
            self.last_sequence = sequence
            raw = record["raw"]
            sample = ImuSample(
                timestamp,
                raw[0] / ACCEL_SCALE,
                raw[1] / ACCEL_SCALE,
                raw[2] / ACCEL_SCALE,
                raw[3] / GYRO_SCALE,
                raw[4] / GYRO_SCALE,
                raw[5] / GYRO_SCALE,
            )
            if len(self.samples) == self.samples.maxlen:
                self.queue_drops += 1
            self.samples.append(sample)
            self.total_samples += 1
            self.last_arrival_time = record["arrival_time"]
            self.first_sender64 = sender64 if self.first_sender64 is None else self.first_sender64
            self.last_sender64 = sender64

    def _pump(self, wait_seconds=0.02):
        if self.fd is None:
            raise RuntimeError("Gateway serial port is not open")
        readable, _, _ = select.select([self.fd], [], [], max(0.0, float(wait_seconds)))
        if not readable:
            return
        try:
            data = os.read(self.fd, 4096)
        except BlockingIOError:
            return
        if data:
            self._accept_records(self.parser.feed(data, time.monotonic()))

    def initialize(self, timeout_s=3.0):
        if self.fd is None:
            try:
                self._open_port()
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot open gateway {self.port}: {exc}. Available: {self.available_ports()}"
                ) from exc
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline and not self.samples:
            self._pump(0.05)
        if not self.samples:
            raise TimeoutError(
                f"No valid gateway frame in {timeout_s:.1f}s; bytes={self.parser.bytes_received}, "
                f"bad_crc={self.parser.bad_crc}, bad_length={self.parser.bad_length}"
            )
        return 0x68

    def who_am_i(self):
        return 0x68

    def read_sample(self, timeout=None):
        deadline = time.monotonic() + (self.timeout_s if timeout is None else float(timeout))
        while not self.samples:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("No wireless IMU sample received before the stale-data limit")
            self._pump(min(remaining, 0.02))
        sample = self.samples.popleft()
        gyro = np.asarray([sample.gx, sample.gy, sample.gz]) - self.gyro_bias
        return ImuSample(sample.timestamp, sample.ax, sample.ay, sample.az, *gyro.tolist())

    def calibrate_gyro(self, samples=100, interval=0.02):
        self.gyro_bias[:] = 0
        # Notifications already arrive at 50 Hz; sleeping again would consume at
        # 25 Hz and leave a one-second stale backlog after calibration.
        self.samples.clear()
        readings = []
        for _ in range(int(samples)):
            item = self.read_sample(timeout=max(self.timeout_s, 0.25))
            readings.append([item.gx, item.gy, item.gz])
        self.gyro_bias = np.mean(readings, axis=0)
        return self.gyro_bias.copy()

    def diagnostics(self):
        span = None
        if self.first_sender64 is not None and self.last_sender64 > self.first_sender64:
            span = (self.last_sender64 - self.first_sender64) / 1e6
        expected = self.total_samples + self.lost_samples
        return {
            "port": self.port,
            "bytes": self.parser.bytes_received,
            "frames": self.parser.frames,
            "bad_crc": self.parser.bad_crc,
            "bad_length": self.parser.bad_length,
            "samples": self.total_samples,
            "lost_samples": self.lost_samples,
            "out_of_order_samples": self.out_of_order_samples,
            "queue_drops": self.queue_drops,
            "receive_ratio": self.total_samples / expected if expected else None,
            "sender_hz": (self.total_samples - 1) / span if span and self.total_samples > 1 else None,
            "last_packet_age_s": time.monotonic() - self.last_arrival_time if self.last_arrival_time else None,
        }

    @staticmethod
    def as_row(sample):
        return asdict(sample)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
