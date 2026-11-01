"""Motion-aware person tracker for FitGuard V2.

The PL threshold/morphology image remains part of the candidate path, while
temporal luma change prevents a large static bright region from becoming the
tracked subject.  This module is deliberately independent from the V1 runtime.
"""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


WIDTH = 320
HEIGHT = 240


@dataclass
class MotionVisionFeature:
    timestamp: float
    valid: int
    cx: float
    cy: float
    area: float
    width: float
    height: float
    vx: float
    vy: float
    quality: float
    motion_score: float
    source: str


def _iou(a, b):
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _center_distance(a, b):
    if a is None or b is None:
        return 1.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = (ax + aw * 0.5) - (bx + bw * 0.5)
    dy = (ay + ah * 0.5) - (by + bh * 0.5)
    return min(1.0, float(np.hypot(dx / WIDTH, dy / HEIGHT)))


def _boxes(binary, min_area=120.0):
    result = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0] if len(result) == 2 else result[1]
    found = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        found.append((x, y, w, h, area))
    return found


class MotionPersonTracker:
    """Select one moving human-like target and retain it through short stops."""

    def __init__(self, hold_seconds=1.5, ema=0.25, acquire_frames=3):
        self.hold_seconds = float(hold_seconds)
        self.ema = float(ema)
        self.acquire_frames = int(acquire_frames)
        self.previous_luma = None
        self.bbox: Optional[Tuple[int, int, int, int]] = None
        self.last_seen = 0.0
        self.previous_feature = None
        self.pending_bbox = None
        self.pending_count = 0
        self.track_dx = 0.0
        self.track_dy = 0.0
        # A fast display box is derived from the exact candidate selected by
        # the validated stable tracker. It never enters CNN/state features.
        self.display_bbox = None

    @staticmethod
    def luma_from_yuyv(raw):
        flat = np.asarray(raw, dtype=np.uint8).reshape(-1)
        if flat.size != WIDTH * HEIGHT * 2:
            raise ValueError(f"unexpected YUYV byte count: {flat.size}")
        return flat[0::2].reshape(HEIGHT, WIDTH).copy()

    def _motion_mask(self, luma):
        if self.previous_luma is None:
            self.previous_luma = luma
            return np.zeros_like(luma)
        current = cv2.GaussianBlur(luma, (5, 5), 0)
        previous = cv2.GaussianBlur(self.previous_luma, (5, 5), 0)
        delta = current.astype(np.int16) - previous.astype(np.int16)
        delta -= int(np.median(delta))  # reject global brightness changes
        magnitude = np.abs(delta).astype(np.uint8)
        threshold = max(10, int(np.percentile(magnitude, 88)))
        motion = np.where(magnitude >= threshold, 255, 0).astype(np.uint8)
        motion = cv2.morphologyEx(
            motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        motion = cv2.morphologyEx(
            motion, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
        )
        motion = cv2.dilate(motion, np.ones((9, 5), np.uint8), iterations=1)
        self.previous_luma = luma
        return motion

    def _candidate_score(self, bbox, motion, pl_mask, luma):
        x, y, w, h = bbox
        area_ratio = (w * h) / float(WIDTH * HEIGHT)
        if area_ratio < 0.012 or area_ratio > 0.68 or h < 28:
            return None
        aspect = w / float(max(h, 1))
        if aspect > 1.30:
            return None
        # Full-body and waist-mounted views may vary, but broad static windows
        # should receive little geometric credit.
        geometry = max(0.0, 1.0 - abs(aspect - 0.48) / 0.85)
        roi = motion[y:y + h, x:x + w]
        motion_ratio = float(np.count_nonzero(roi)) / float(max(w * h, 1))
        motion_score = min(1.0, motion_ratio / 0.18)
        predicted = None
        if self.bbox is not None:
            bx, by, bw, bh = self.bbox
            predicted = (
                int(round(bx + self.track_dx)),
                int(round(by + self.track_dy)),
                bw,
                bh,
            )
        continuity = max(
            _iou(bbox, self.bbox),
            _iou(bbox, predicted),
            1.0 - 2.2 * _center_distance(bbox, predicted or self.bbox),
        )
        center_x = (x + w * 0.5) / WIDTH
        center_prior = max(0.0, 1.0 - abs(center_x - 0.5) / 0.65)
        ground_prior = float(np.clip(((y + h) / HEIGHT - 0.38) / 0.62, 0.0, 1.0))
        border_penalty = 0.20 if x <= 1 or y <= 1 or x + w >= WIDTH - 1 else 0.0
        pl_roi = pl_mask[y:y + h, x:x + w]
        bright_fraction = float(np.count_nonzero(pl_roi)) / float(max(w * h, 1))
        # PL is now supporting evidence only. A box containing both foreground
        # and background receives a small bonus; it can never create a target.
        pl_contrast = 2.0 * min(bright_fraction, 1.0 - bright_fraction)
        luma_roi = luma[y:y + h, x:x + w]
        highlight_fraction = float(np.count_nonzero(luma_roi >= 245)) / float(max(w * h, 1))
        texture = min(1.0, float(np.std(luma_roi)) / 38.0)
        # A saturated, almost textureless region is normally a lamp, window or
        # white wall rather than the person.  Reject it before it can steal a
        # fresh lock when auto-exposure makes it flicker.
        if highlight_fraction > 0.48 and texture < 0.42:
            return None
        size_penalty = 0.0
        if self.bbox is not None:
            _, _, old_w, old_h = self.bbox
            size_penalty = min(
                1.0,
                abs(np.log(max(w, 1) / max(old_w, 1)))
                + abs(np.log(max(h, 1) / max(old_h, 1))),
            )
        score = (
            0.44 * motion_score
            + 0.34 * continuity
            + 0.14 * geometry
            + 0.03 * center_prior
            + 0.05 * ground_prior
            + 0.02 * pl_contrast
            + 0.03 * texture
            - 0.42 * highlight_fraction
            - 0.16 * size_penalty
            - border_penalty
        )
        return score, motion_score

    def _smooth_bbox(self, old_box, new_box):
        ox, oy, ow, oh = old_box
        nx, ny, nw, nh = new_box
        ocx, ocy = ox + ow * 0.5, oy + oh * 0.5
        ncx, ncy = nx + nw * 0.5, ny + nh * 0.5
        # Original validated feature path retained exactly for CNN/state input.
        raw_dx, raw_dy = ncx - ocx, ncy - ocy
        ncx = float(np.clip(ncx, ocx - 32.0, ocx + 32.0))
        ncy = float(np.clip(ncy, ocy - 24.0, ocy + 24.0))
        displacement = float(np.hypot(raw_dx, raw_dy))
        center_alpha = 0.58 if displacement >= 8.0 else 0.40
        cx = center_alpha * ncx + (1.0 - center_alpha) * ocx
        cy = center_alpha * ncy + (1.0 - center_alpha) * ocy
        width = 0.20 * nw + 0.80 * ow
        height = 0.20 * nh + 0.80 * oh
        width = int(np.clip(round(width), 1, WIDTH))
        height = int(np.clip(round(height), 1, HEIGHT))
        x = int(np.clip(round(cx - width * 0.5), 0, WIDTH - width))
        y = int(np.clip(round(cy - height * 0.5), 0, HEIGHT - height))
        return x, y, width, height

    @staticmethod
    def _smooth_display_bbox(old_box, new_box):
        """Responsive display-only box for the same selected target/frame."""
        if old_box is None:
            return new_box
        ox, oy, ow, oh = old_box
        nx, ny, nw, nh = new_box
        ocx, ocy = ox + ow * 0.5, oy + oh * 0.5
        ncx, ncy = nx + nw * 0.5, ny + nh * 0.5
        raw_dx, raw_dy = ncx - ocx, ncy - ocy
        ncx = float(np.clip(ncx, ocx - 48.0, ocx + 48.0))
        ncy = float(np.clip(ncy, ocy - 34.0, ocy + 34.0))
        displacement = float(np.hypot(raw_dx, raw_dy))
        center_alpha = 0.80 if displacement >= 18.0 else (0.66 if displacement >= 8.0 else 0.44)
        size_alpha = 0.30 if displacement >= 8.0 else 0.24
        cx = center_alpha * ncx + (1.0 - center_alpha) * ocx
        cy = center_alpha * ncy + (1.0 - center_alpha) * ocy
        width = int(np.clip(round(size_alpha * nw + (1.0 - size_alpha) * ow), 1, WIDTH))
        height = int(np.clip(round(size_alpha * nh + (1.0 - size_alpha) * oh), 1, HEIGHT))
        x = int(np.clip(round(cx - width * 0.5), 0, WIDTH - width))
        y = int(np.clip(round(cy - height * 0.5), 0, HEIGHT - height))
        return x, y, width, height

    def update(self, raw, pl_mask, timestamp):
        luma = self.luma_from_yuyv(raw)
        motion = self._motion_mask(luma)
        candidates = []

        for x, y, w, h, _ in _boxes(motion, min_area=110):
            # Motion tends to represent body edges; modest expansion recovers
            # a useful person box without making the entire frame a target.
            px = int(w * 0.25)
            py_top, py_bottom = int(h * 0.18), int(h * 0.55)
            left = max(0, x - px)
            top = max(0, y - py_top)
            box = (
                left, top,
                min(WIDTH - left, w + 2 * px),
                min(HEIGHT - top, h + py_top + py_bottom),
            )
            candidates.append(box)

        scored = []
        for bbox in candidates:
            # While a track is fresh, reject distant one-frame disturbances.
            # A genuinely new target may be acquired only after the old track
            # expires and passes the consecutive-frame acquisition gate below.
            if (
                self.bbox is not None
                and timestamp - self.last_seen <= self.hold_seconds
                and _iou(bbox, self.bbox) < 0.03
                and _center_distance(bbox, self.bbox) > 0.22
            ):
                continue
            result = self._candidate_score(bbox, motion, pl_mask, luma)
            if result is not None:
                score, motion_score = result
                scored.append((score, motion_score, bbox))
        scored.sort(key=lambda item: item[0], reverse=True)

        selected = None
        display_measurement = None
        if scored and scored[0][0] >= 0.36:
            quality, motion_score, new_box = scored[0]
            if self.bbox is None:
                if (
                    self.pending_bbox is not None
                    and _center_distance(new_box, self.pending_bbox) <= 0.12
                ):
                    self.pending_count += 1
                else:
                    self.pending_bbox = new_box
                    self.pending_count = 1
                if self.pending_count >= self.acquire_frames:
                    self.bbox = new_box
                    self.last_seen = timestamp
                    selected = (new_box, float(np.clip(quality, 0.0, 1.0)), motion_score, "motion+pl")
                    display_measurement = new_box
                    self.pending_bbox = None
                    self.pending_count = 0
            else:
                old_x, old_y, old_w, old_h = self.bbox
                old_cx, old_cy = old_x + old_w * 0.5, old_y + old_h * 0.5
                new_x, new_y, new_w, new_h = new_box
                new_cx, new_cy = new_x + new_w * 0.5, new_y + new_h * 0.5
                self.track_dx = 0.65 * self.track_dx + 0.35 * (new_cx - old_cx)
                self.track_dy = 0.65 * self.track_dy + 0.35 * (new_cy - old_cy)
                smooth = self._smooth_bbox(self.bbox, new_box)
                self.bbox = smooth
                self.last_seen = timestamp
                selected = (smooth, float(np.clip(quality, 0.0, 1.0)), motion_score, "motion+pl")
                display_measurement = new_box
        elif self.bbox is not None and timestamp - self.last_seen <= self.hold_seconds:
            age = timestamp - self.last_seen
            selected = (
                self.bbox,
                max(0.18, 0.55 * (1.0 - age / self.hold_seconds)),
                0.0,
                "hold",
            )
        else:
            self.bbox = None
            self.track_dx = 0.0
            self.track_dy = 0.0
            self.display_bbox = None
            if not scored:
                self.pending_bbox = None
                self.pending_count = 0

        # Display uses the current-frame measurement of this same track only.
        # No extrapolated/stale position can become a different visual target.
        if selected is not None:
            if display_measurement is not None:
                self.display_bbox = self._smooth_display_bbox(self.display_bbox, display_measurement)
            elif self.display_bbox is None:
                self.display_bbox = self.bbox

        valid = int(selected is not None)
        if valid:
            (x, y, w, h), quality, motion_score, source = selected
            cx, cy = x + w * 0.5, y + h * 0.5
            area = float(w * h)
        else:
            x = y = w = h = 0
            cx = cy = area = quality = motion_score = 0.0
            source = "lost"

        vx = vy = 0.0
        if valid and self.previous_feature is not None and self.previous_feature.valid:
            dt = timestamp - self.previous_feature.timestamp
            if dt > 0:
                vx = (cx / WIDTH - self.previous_feature.cx) / dt
                vy = (cy / HEIGHT - self.previous_feature.cy) / dt

        feature = MotionVisionFeature(
            timestamp=float(timestamp), valid=valid,
            cx=float(cx / WIDTH), cy=float(cy / HEIGHT),
            area=float(area / (WIDTH * HEIGHT)),
            width=float(w / WIDTH), height=float(h / HEIGHT),
            vx=float(vx), vy=float(vy), quality=float(quality),
            motion_score=float(motion_score), source=source,
        )
        self.previous_feature = feature
        return feature, motion, self.annotate(raw, feature, self.display_bbox)

    @staticmethod
    def annotate(raw, feature, display_bbox=None):
        bgr = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
        color = (40, 220, 40) if feature.valid else (40, 40, 230)
        if feature.valid:
            if display_bbox is None:
                w = int(round(feature.width * WIDTH))
                h = int(round(feature.height * HEIGHT))
                x = int(round(feature.cx * WIDTH - w * 0.5))
                y = int(round(feature.cy * HEIGHT - h * 0.5))
            else:
                x, y, w, h = display_bbox
            display_cx, display_cy = x + w * 0.5, y + h * 0.5
            cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
            cv2.drawMarker(
                bgr, (int(display_cx), int(display_cy)),
                (0, 0, 255), cv2.MARKER_CROSS, 13, 2,
            )
        cv2.putText(
            bgr,
            f"Target {'LOCKED' if feature.valid else 'LOST'}  "
            f"Q={feature.quality:.2f}  {feature.source}",
            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA,
        )
        return bgr


def feature_dict(feature):
    return asdict(feature)
