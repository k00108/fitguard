"""Generate provenance-preserving, train-only waist dataset augmentation."""

import argparse
import json
from pathlib import Path

import numpy as np


def _rotation_matrix(rng, max_degrees=2.5):
    angles = np.deg2rad(rng.uniform(-max_degrees, max_degrees, size=3))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rx = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (rz @ ry @ rx).astype(np.float32)


def _time_warp(sequence, speed):
    length = sequence.shape[0]
    source = np.arange(length, dtype=np.float32)
    center = (length - 1) / 2.0
    target = np.clip((source - center) * speed + center, 0, length - 1)
    return np.column_stack(
        [np.interp(target, source, sequence[:, i]) for i in range(sequence.shape[1])]
    ).astype(np.float32)


def _smooth_noise(rng, shape, sigma):
    noise = rng.normal(0.0, sigma, size=shape)
    return np.apply_along_axis(
        lambda x: np.convolve(x, [0.25, 0.5, 0.25], mode="same"), 0, noise
    ).astype(np.float32)


def _augment_trial(imu, vision, rng):
    rotation = _rotation_matrix(rng)
    speed = float(rng.uniform(0.97, 1.03))
    accel_gain = float(rng.uniform(0.98, 1.02))
    gyro_gain = float(rng.uniform(0.98, 1.02))
    x_shift = float(rng.uniform(-1.5, 1.5) / 320.0)
    y_shift = float(rng.uniform(-1.5, 1.5) / 240.0)
    size_gain = float(rng.uniform(0.99, 1.01))

    imu_std = np.maximum(imu.std(axis=(0, 1)), 1e-5)
    imu_sigma = np.minimum(
        imu_std * 0.015,
        np.asarray([0.004, 0.004, 0.004, 0.06, 0.06, 0.06], dtype=np.float32),
    )
    vision_std = np.maximum(vision.std(axis=(0, 1)), 1e-6)

    imu_out, vision_out = [], []
    for imu_window, vision_window in zip(imu, vision):
        imu_item = _time_warp(imu_window, speed)
        accel_mean = imu_item[:, :3].mean(axis=0, keepdims=True)
        gyro_mean = imu_item[:, 3:].mean(axis=0, keepdims=True)
        imu_item[:, :3] = accel_mean + (imu_item[:, :3] - accel_mean) * accel_gain
        imu_item[:, 3:] = gyro_mean + (imu_item[:, 3:] - gyro_mean) * gyro_gain
        imu_item[:, :3] = imu_item[:, :3] @ rotation.T
        imu_item[:, 3:] = imu_item[:, 3:] @ rotation.T
        imu_item += _smooth_noise(rng, imu_item.shape, imu_sigma)

        vision_item = _time_warp(vision_window, speed)
        present = vision_item[:, 2] > 1e-8
        vision_item[present, 0] += x_shift + rng.normal(0, 0.35 / 320.0, present.sum())
        vision_item[present, 1] += y_shift + rng.normal(0, 0.35 / 240.0, present.sum())
        vision_item[present, 2:5] *= size_gain
        vision_item[present, 5:7] += rng.normal(
            0.0,
            np.maximum(vision_std[5:7] * 0.005, 1e-7),
            size=(present.sum(), 2),
        )
        vision_item[:, :5] = np.clip(vision_item[:, :5], 0.0, 1.0)
        vision_item[~present] = 0.0
        imu_out.append(imu_item.astype(np.float32))
        vision_out.append(vision_item.astype(np.float32))
    return np.stack(imu_out), np.stack(vision_out), {
        "speed": speed,
        "accel_gain": accel_gain,
        "gyro_gain": gyro_gain,
        "x_shift_pixels": x_shift * 320.0,
        "y_shift_pixels": y_shift * 240.0,
        "size_gain": size_gain,
    }


def augment_dataset(source_root, output_root, variants=10, seed=20261001):
    source_root, output_root = Path(source_root), Path(output_root)
    source_files = sorted(source_root.glob("W*/*/trial_*/windows.npz"))
    if not source_files:
        raise RuntimeError(f"No W* source trials found below {source_root}")
    if output_root.exists() and any(output_root.rglob("windows.npz")):
        raise FileExistsError(f"Refusing to overwrite existing augmentation: {output_root}")

    manifest = {
        "synthetic": True,
        "training_only": True,
        "real_test_windows_required": True,
        "seed": seed,
        "variants_per_real_trial": variants,
        "methods": [
            "synchronized multimodal time warp 0.97-1.03x",
            "trial-consistent sensor mounting rotation +/-2.5 degrees",
            "dynamic IMU gain 0.98-1.02 with bounded low-pass sensor noise",
            "camera centroid translation +/-1.5 pixels and size gain 0.99-1.01",
        ],
        "sources": [],
        "outputs": [],
    }
    total_windows = 0
    for source_number, path in enumerate(source_files):
        with np.load(path, allow_pickle=False) as data:
            imu = data["imu"].astype(np.float32)
            vision = data["vision"].astype(np.float32)
            label = str(data["label"][0])
            subject = str(data["subject"][0])
        label_slug = path.parent.parent.name
        source_trial = path.parent.name
        manifest["sources"].append(str(path))
        for variant in range(1, variants + 1):
            variant_seed = seed + source_number * 1000 + variant
            rng = np.random.default_rng(variant_seed)
            aug_imu, aug_vision, parameters = _augment_trial(imu, vision, rng)
            output = output_root / subject / label_slug / f"{source_trial}_aug_{variant:02d}"
            output.mkdir(parents=True, exist_ok=False)
            count = len(aug_imu)
            np.savez_compressed(
                output / "windows.npz",
                imu=aug_imu,
                vision=aug_vision,
                label=np.asarray([label] * count),
                subject=np.asarray([subject] * count),
                synthetic=np.ones(count, dtype=np.uint8),
                source_subject=np.asarray([subject] * count),
                source_trial=np.asarray([str(path)] * count),
                source_window_index=np.arange(count, dtype=np.int32),
                variant=np.full(count, variant, dtype=np.int16),
            )
            metadata = {
                "synthetic": True,
                "training_only": True,
                "source_subject": subject,
                "source_trial": str(path),
                "label": label,
                "variant": variant,
                "variant_seed": variant_seed,
                "windows": count,
                "parameters": parameters,
            }
            (output / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            manifest["outputs"].append(str(output / "windows.npz"))
            total_windows += count

    manifest["source_trials"] = len(source_files)
    manifest["synthetic_trials"] = len(manifest["outputs"])
    manifest["synthetic_windows"] = total_windows
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "SYNTHETIC_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    result = augment_dataset(args.source, args.output, args.variants, args.seed)
    print(json.dumps({
        "source_trials": result["source_trials"],
        "synthetic_trials": result["synthetic_trials"],
        "synthetic_windows": result["synthetic_windows"],
        "output_root": str(Path(args.output).resolve()),
    }, ensure_ascii=False, indent=2))
