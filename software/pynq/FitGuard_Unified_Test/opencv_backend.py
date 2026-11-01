"""OpenCV backend: denoise, contour extraction, and target position output."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Target:
    contour: np.ndarray
    area: float
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]


def _find_contours(binary: np.ndarray):
    result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result[0] if len(result) == 2 else result[1]


def clean_binary(
    binary: np.ndarray,
    median_ksize: int = 5,
    open_ksize: int = 3,
    close_ksize: int = 7,
) -> np.ndarray:
    """Remove isolated noise, then fill small holes and breaks."""
    if binary.ndim != 2 or binary.dtype != np.uint8:
        raise ValueError("binary must be a 2-D uint8 image")
    if median_ksize < 3 or median_ksize % 2 == 0:
        raise ValueError("median_ksize must be an odd integer >= 3")

    normalized = np.where(binary != 0, 255, 0).astype(np.uint8)
    denoised = cv2.medianBlur(normalized, median_ksize)

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
    )
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, open_kernel)
    denoised = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, close_kernel)
    return denoised


def extract_targets(
    clean: np.ndarray,
    min_area: float = 200.0,
    max_area_ratio: float = 0.80,
) -> List[Target]:
    """Return external targets sorted from largest to smallest."""
    image_area = float(clean.shape[0] * clean.shape[1])
    targets: List[Target] = []

    for contour in _find_contours(clean):
        area = float(cv2.contourArea(contour))
        if area < min_area or area > image_area * max_area_ratio:
            continue

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-9:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
        else:
            cx = int(round(moments["m10"] / moments["m00"]))
            cy = int(round(moments["m01"] / moments["m00"]))
            x, y, w, h = cv2.boundingRect(contour)

        targets.append(
            Target(
                contour=contour,
                area=area,
                center=(cx, cy),
                bbox=(x, y, w, h),
            )
        )

    targets.sort(key=lambda item: item.area, reverse=True)
    return targets


def annotate(image_bgr: np.ndarray, targets: List[Target]) -> np.ndarray:
    """Draw contour, bounding box, center, and pixel coordinates."""
    result = image_bgr.copy()
    for index, target in enumerate(targets, start=1):
        x, y, w, h = target.bbox
        cx, cy = target.center
        cv2.drawContours(result, [target.contour], -1, (0, 255, 0), 2)
        cv2.rectangle(result, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 2)
        cv2.drawMarker(
            result,
            (cx, cy),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            14,
            2,
        )
        cv2.putText(
            result,
            f"#{index} ({cx},{cy}) A={target.area:.0f}",
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return result


def process_binary(
    binary: np.ndarray,
    image_bgr: Optional[np.ndarray] = None,
    min_area: float = 200.0,
):
    """Complete backend. Returns clean mask, targets, and annotated BGR image."""
    clean = clean_binary(binary)
    targets = extract_targets(clean, min_area=min_area)
    if image_bgr is None:
        image_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    annotated = annotate(image_bgr, targets)
    return clean, targets, annotated
