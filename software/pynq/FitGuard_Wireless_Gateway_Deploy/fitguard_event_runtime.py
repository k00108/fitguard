"""Marked-abnormality event recording.

The recorder is driven only by the final debounced state. A single raw CNN
window can never trigger an event.
"""

from collections import deque
import json
from pathlib import Path
import time


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


class MarkedEventRecorder:
    def __init__(self, root="fitguard_marked_events", pre_context=12):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.history = deque(maxlen=int(pre_context))
        self.active = None
        self.event_count = 0
        for path in self.root.glob("marked_event_*"):
            try:
                self.event_count = max(
                    self.event_count, int(path.name.rsplit("_", 1)[-1])
                )
            except ValueError:
                continue

    def _write(self):
        if self.active is not None:
            _atomic_json(self.active["directory"] / "event.json", self.active["payload"])

    def _start(self, record, snapshot_jpeg):
        self.event_count += 1
        event_id = f"marked_event_{self.event_count:04d}"
        directory = self.root / event_id
        directory.mkdir(parents=True, exist_ok=False)
        snapshot_name = None
        if snapshot_jpeg:
            snapshot_name = "trigger.jpg"
            (directory / snapshot_name).write_bytes(snapshot_jpeg)
        payload = {
            "event_id": event_id,
            "status": "active",
            "trigger_board_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "board_time_is_authoritative": False,
            "trigger_timestamp": record.get("timestamp"),
            "trigger_rule": "debounced final state equals Marked Abnormality",
            "snapshot": snapshot_name,
            "pre_context": list(self.history),
            "samples": [record],
        }
        self.active = {"directory": directory, "payload": payload}
        self._write()
        return event_id

    def _finish(self, reason, timestamp=None):
        if self.active is None:
            return None
        payload = self.active["payload"]
        payload["status"] = "completed"
        payload["end_reason"] = reason
        payload["end_timestamp"] = timestamp
        payload["ended_board_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._write()
        event_id = payload["event_id"]
        self.active = None
        return event_id

    def update(self, record, snapshot_jpeg=None):
        healthy = record.get("sensor_state") == "OK"
        marked = record.get("label") == "Marked Abnormality" and healthy
        started = None
        ended = None
        if marked and self.active is None:
            started = self._start(record, snapshot_jpeg)
        elif marked:
            self.active["payload"]["samples"].append(record)
            self._write()
        elif self.active is not None:
            reason = "sensor_not_healthy" if not healthy else "state_recovered"
            ended = self._finish(reason, record.get("timestamp"))
        self.history.append(record)
        return {
            "active": self.active is not None,
            "event_id": (
                self.active["payload"]["event_id"] if self.active is not None else None
            ),
            "started": started,
            "ended": ended,
        }

    def close(self, reason="run_stopped"):
        return self._finish(reason)

