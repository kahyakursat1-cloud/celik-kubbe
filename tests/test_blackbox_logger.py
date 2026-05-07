import csv
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _drain_queue_sync(logger) -> None:
    """Run the worker loop body once for each queued item, no QThread needed."""
    import queue as _q
    while True:
        try:
            data = logger._queue.get_nowait()
        except _q.Empty:
            return
        if data["type"] == "track":
            with open(logger._track_file, mode="a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([data["timestamp"], data["id"], data["sinif"], data["level"],
                            data["range"], data["bearing"], data["velocity"],
                            data["altitude"], data["source"], data["status"]])
        elif data["type"] == "event":
            with open(logger._event_file, mode="a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([data["timestamp"], data["event_type"],
                            data["target_id"], data["description"]])


class _FakeThreat:
    def __init__(self, tid="THR-007"):
        self.id = tid
        self.sinif = "Drone"
        self.threat_level = "YÜKSEK"
        self.engaged = False
        self.kaynak = "fuzyon"
        self.altitude = 250
        self.velocity_ms = -45.0

    def range_km(self):
        return 1.234

    def bearing(self):
        return 42.5


class BlackboxLoggerTests(unittest.TestCase):
    def setUp(self):
        from src.blackbox_logger import BlackboxLogger
        self.tmp = tempfile.TemporaryDirectory()
        self.logger = BlackboxLogger(log_dir=self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_csv_headers_written_on_init(self):
        with open(self.logger._track_file, encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertIn("Timestamp", header)
        self.assertIn("Threat_ID", header)
        self.assertIn("Range_km", header)
        self.assertIn("Source", header)

        with open(self.logger._event_file, encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertEqual(header, ["Timestamp", "Event_Type", "Target_ID", "Description"])

    def test_log_tehdit_writes_track_row(self):
        self.logger.log_tehdit(_FakeThreat("THR-001"))
        _drain_queue_sync(self.logger)

        with open(self.logger._track_file, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 2)  # header + 1 satır
        body = rows[1]
        self.assertEqual(body[1], "THR-001")
        self.assertEqual(body[2], "Drone")
        self.assertEqual(body[8], "fuzyon")
        self.assertEqual(body[9], "Active")

    def test_log_olay_writes_event_row(self):
        self.logger.log_olay("ENGAGEMENT", "THR-099", "Pil-ALFA atış emri")
        _drain_queue_sync(self.logger)

        with open(self.logger._event_file, encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][1], "ENGAGEMENT")
        self.assertEqual(rows[1][2], "THR-099")


if __name__ == "__main__":
    unittest.main(verbosity=2)
