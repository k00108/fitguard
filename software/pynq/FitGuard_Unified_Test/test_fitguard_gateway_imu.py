"""Hardware-free protocol regression for the ESP32 USB gateway backend."""

import struct

from fitguard_gateway_imu import (
    FrameParser,
    GatewayMPU6050,
    TYPE_IMU_BATCH,
    build_frame,
)


def record(sequence, sender_us, ax, ay, az, gx, gy, gz, temp=0):
    return struct.pack("<BIhhhhhhh", sequence, sender_us, ax, ay, az, gx, gy, gz, temp)


def main():
    payload = b"".join((
        record(10, 1_000_000, 16384, -8192, 4096, 131, -262, 393, 1234),
        record(12, 1_020_000, 8192, 0, -16384, -131, 0, 262, 5678),
    ))
    frame = build_frame(TYPE_IMU_BATCH, payload)
    parser = FrameParser()
    assert parser.feed(frame[:7], arrival_time=50.0) == []
    rows = parser.feed(frame[7:], arrival_time=50.02)
    assert len(rows) == 2 and parser.bad_crc == 0

    imu = GatewayMPU6050()
    imu._accept_records(rows)
    first = imu.read_sample()
    second = imu.read_sample()
    assert abs(first.ax - 1.0) < 1e-9
    assert abs(first.gx - 1.0) < 1e-9
    assert abs(first.gy + 2.0) < 1e-9
    assert abs(first.gz - 3.0) < 1e-9
    assert second.timestamp > first.timestamp
    assert imu.lost_samples == 1

    damaged = bytearray(frame)
    damaged[-1] ^= 0x80
    assert FrameParser().feed(damaged, arrival_time=60.0) == []
    damaged_parser = FrameParser()
    damaged_parser.feed(damaged, arrival_time=60.0)
    assert damaged_parser.bad_crc == 1
    print("FITGUARD_GATEWAY_IMU_TEST_PASS", imu.diagnostics())


if __name__ == "__main__":
    main()
