"""MPU6050 reader for the AXI IIC block in the unified FitGuard overlay."""

import time
from dataclasses import asdict, dataclass

import numpy as np


MPU6050_ADDRESS = 0x68


@dataclass
class ImuSample:
    timestamp: float
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


class MPU6050:
    def __init__(self, axi_iic, address=MPU6050_ADDRESS):
        self.iic = axi_iic
        self.address = address
        self._ffi = axi_iic._ffi
        self.gyro_bias = np.zeros(3, dtype=np.float64)

    def _write(self, register, value):
        payload = self._ffi.new("unsigned char[]", [register, value])
        sent = self.iic.send(self.address, payload, 2, 0)
        if sent != 2:
            raise IOError(f"MPU6050 write failed at register 0x{register:02X}")

    def _read(self, register, length):
        register_buffer = self._ffi.new("unsigned char[]", [register])
        receive_buffer = self._ffi.new("unsigned char[]", length)
        sent = self.iic.send(
            self.address, register_buffer, 1, self.iic.REPEAT_START
        )
        received = self.iic.receive(self.address, receive_buffer, length, 0)
        if sent != 1 or received != length:
            raise IOError(
                f"MPU6050 read failed at register 0x{register:02X}: "
                f"sent={sent}, received={received}"
            )
        return bytes(self._ffi.buffer(receive_buffer, length))

    def initialize(self):
        # Wake up using the X-axis gyro PLL, enable DLPF, and produce 50 Hz.
        self._write(0x6B, 0x01)  # PWR_MGMT_1
        time.sleep(0.1)
        self._write(0x1A, 0x03)  # CONFIG: ~44 Hz bandwidth
        self._write(0x19, 19)    # SMPLRT_DIV: 1 kHz / (1 + 19) = 50 Hz
        self._write(0x1B, 0x00)  # GYRO_CONFIG: +/-250 deg/s
        self._write(0x1C, 0x00)  # ACCEL_CONFIG: +/-2 g
        identity = self.who_am_i()
        if identity != 0x68:
            raise RuntimeError(f"Unexpected MPU6050 WHO_AM_I: 0x{identity:02X}")
        return identity

    def who_am_i(self):
        return self._read(0x75, 1)[0]

    @staticmethod
    def _signed(high, low):
        value = (high << 8) | low
        return value - 65536 if value & 0x8000 else value

    def read_sample(self):
        raw = self._read(0x3B, 14)
        accel = np.asarray(
            [
                self._signed(raw[0], raw[1]),
                self._signed(raw[2], raw[3]),
                self._signed(raw[4], raw[5]),
            ],
            dtype=np.float64,
        ) / 16384.0
        gyro = np.asarray(
            [
                self._signed(raw[8], raw[9]),
                self._signed(raw[10], raw[11]),
                self._signed(raw[12], raw[13]),
            ],
            dtype=np.float64,
        ) / 131.0
        gyro -= self.gyro_bias
        return ImuSample(time.monotonic(), *accel.tolist(), *gyro.tolist())

    def calibrate_gyro(self, samples=100, interval=0.02):
        """Estimate gyro bias; the sensor must remain still during this call."""
        readings = []
        self.gyro_bias[:] = 0
        for _ in range(samples):
            item = self.read_sample()
            readings.append([item.gx, item.gy, item.gz])
            time.sleep(interval)
        self.gyro_bias = np.mean(readings, axis=0)
        return self.gyro_bias.copy()

    @staticmethod
    def as_row(sample):
        return asdict(sample)
