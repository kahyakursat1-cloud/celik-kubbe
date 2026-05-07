import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QCoreApplication


def _ensure_app():
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    return app


class GimbalMockTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()

    def test_baslat_in_mock_sets_connected(self):
        from src.gimbal_controller import GimbalController

        g = GimbalController(mock=True)
        g.baslat()
        self.assertTrue(g._bagli)
        g.durdur()

    def test_hedefe_yonel_emits_status_signal(self):
        from src.gimbal_controller import GimbalController

        g = GimbalController(mock=True)
        g.baslat()

        captured = []
        g.gimbal_durum_sinyal.connect(captured.append)

        g.hedefe_yonel(tid="THR-X", bearing_deg=45.0, distance_km=2.0, altitude_m=400.0)
        self.assertTrue(captured, "gimbal_durum_sinyal yayılmadı")
        last = captured[-1]
        self.assertAlmostEqual(last["pan"], 45.0, delta=0.01)
        self.assertGreater(last["tilt"], 0.0)  # 400 m yükseklikte tilt > 0
        self.assertEqual(last["hedef_id"], "THR-X")

    def test_pan_normalization_wraps_around_180(self):
        from src.gimbal_controller import GimbalController

        g = GimbalController(mock=True)
        g.baslat()

        captured = []
        g.gimbal_durum_sinyal.connect(captured.append)

        g.hedefe_yonel(tid="X", bearing_deg=370.0, distance_km=1.0, altitude_m=100.0)
        self.assertAlmostEqual(captured[-1]["pan"], 10.0, delta=0.01)

    def test_tilt_clamped_to_bounds(self):
        from src.gimbal_controller import GimbalController

        g = GimbalController(mock=True)
        g.baslat()

        captured = []
        g.gimbal_durum_sinyal.connect(captured.append)

        # Çok yakın + yüksek hedef → tilt hesapça > 90 olur, clamp 90'a düşmeli
        g.hedefe_yonel(tid="X", bearing_deg=0.0, distance_km=0.001, altitude_m=10000.0)
        self.assertLessEqual(captured[-1]["tilt"], 90.0)
        self.assertGreaterEqual(captured[-1]["tilt"], -30.0)

    def test_serbest_mod_resets_target(self):
        from src.gimbal_controller import GimbalController

        g = GimbalController(mock=True)
        g.baslat()

        captured = []
        g.gimbal_durum_sinyal.connect(captured.append)

        g.hedefe_yonel(tid="X", bearing_deg=30.0, distance_km=1.0, altitude_m=200.0)
        g.serbest_mod()
        self.assertEqual(captured[-1]["pan"], 0.0)
        self.assertEqual(captured[-1]["tilt"], 0.0)
        self.assertEqual(captured[-1]["hedef_id"], "YOK")


if __name__ == "__main__":
    unittest.main(verbosity=2)
