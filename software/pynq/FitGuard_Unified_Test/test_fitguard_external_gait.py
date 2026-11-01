import numpy as np
import cv2

from fitguard_external_gait import ExternalGaitTracker


def main():
    tracker = ExternalGaitTracker(min_area=200)
    blank = np.zeros((240, 320), dtype=np.uint8)
    record, selected, _, _ = tracker.update(blank, timestamp=0.0)
    assert not record.valid and selected is None

    image = blank.copy()
    cv2.rectangle(image, (8, 170), (155, 205), 255, -1)      # wide distractor
    cv2.rectangle(image, (205, 45), (250, 220), 255, -1)     # full-body candidate
    record, selected, _, _ = tracker.update(image, timestamp=0.05)
    assert record.valid and selected.bbox[0] == 205

    for index in range(1, 90):
        frame = blank.copy()
        x = 40 + index
        bob = int(round(3 * np.sin(index * 0.45)))
        cv2.rectangle(frame, (x, 45 + bob), (x + 40, 220 + bob), 255, -1)
        record, selected, _, indicators = tracker.update(frame, timestamp=0.05 * (index + 1))
        assert record.valid and selected is not None

    assert indicators["forward_speed_body_widths_s"] > 0
    assert indicators["vertical_bob_percent_height"] > 0
    assert indicators["estimated_cadence_steps_min"] is not None
    print("FITGUARD_EXTERNAL_GAIT_TEST_PASS", indicators)


if __name__ == "__main__":
    main()
