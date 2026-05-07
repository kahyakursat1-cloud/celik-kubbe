import sys
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QCoreApplication


def _ensure_app():
    app = QCoreApplication.instance() or QCoreApplication(sys.argv)
    return app


def _radar_det(bearing=0.0, range_km=1.0, vel=-50.0, snr=20.0):
    return {"bearing_deg": bearing, "range_km": range_km,
            "velocity_ms": vel, "snr_db": snr}


def _camera_det(cx=0.5, cy=0.5, w=0.3, h=0.3, sinif="Drone", guven=0.9, tid=1):
    return {"track_id": tid, "sinif": sinif, "guven": guven,
            "cx": cx, "cy": cy, "w": w, "h": h}


class SensorFusionEdgeTests(unittest.TestCase):
    def setUp(self):
        _ensure_app()
        from src.sensor_fusion import SensorFusion
        self.fusion = SensorFusion()

    def test_camera_only_then_radar_promotes_to_fusion(self):
        # 1. Sadece kamera → yalniz_kamera kaynak
        self.fusion.kamera_girdisi([_camera_det(cx=0.5, w=0.3, h=0.3)])
        tracks = list(self.fusion.aktif_izler.values())
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].kaynak, "yalniz_kamera")

        # 2. Aynı bearing'te radar tespiti gel → fuzyon
        # cx=0.5 → bearing=0; aynı bearing'i ver
        self.fusion.radar_girdisi([_radar_det(bearing=0.0, range_km=1.0)])
        self.fusion.kamera_girdisi([_camera_det(cx=0.5, w=0.3, h=0.3)])
        tracks = list(self.fusion.aktif_izler.values())
        kaynaklar = {t.kaynak for t in tracks}
        self.assertIn("fuzyon", kaynaklar)

    def test_stale_track_dropped_after_timeout(self):
        self.fusion.IZ_ZAMAN_ASIMI_S = 0.05
        self.fusion.radar_girdisi([_radar_det(bearing=10.0, range_km=2.0)])
        self.assertEqual(len(self.fusion.aktif_izler), 1)

        time.sleep(0.1)
        # Boş girdi tetikle → süresi dolan izler temizlenmeli
        self.fusion.radar_girdisi([])
        self.assertEqual(len(self.fusion.aktif_izler), 0)

    def test_threat_score_higher_for_close_fast_missile(self):
        # Yakın + yüksek hızla yaklaşan + Missile sınıfı → KRİTİK
        self.fusion.radar_girdisi([_radar_det(bearing=0.0, range_km=0.6, vel=-150.0)])
        self.fusion.kamera_girdisi([
            _camera_det(cx=0.5, w=0.3, h=0.3, sinif="BalisticMissile", guven=0.95)
        ])
        track = next(iter(self.fusion.aktif_izler.values()))
        self.assertGreaterEqual(track.tehdit_skoru, 80.0)
        self.assertEqual(track.tehdit_seviyesi, "KRİTİK")

    def test_distant_slow_target_is_low_threat(self):
        self.fusion.radar_girdisi([_radar_det(bearing=0.0, range_km=4.5, vel=5.0)])
        track = next(iter(self.fusion.aktif_izler.values()))
        self.assertLess(track.tehdit_skoru, 30.0)
        self.assertEqual(track.tehdit_seviyesi, "DÜŞÜK")

    def test_radar_only_track_without_class(self):
        self.fusion.radar_girdisi([_radar_det(bearing=20.0, range_km=2.0)])
        track = next(iter(self.fusion.aktif_izler.values()))
        self.assertEqual(track.kaynak, "yalniz_radar")
        self.assertEqual(track.sinif, "Bilinmeyen")

    def test_temizle_clears_all_tracks(self):
        self.fusion.radar_girdisi([_radar_det(bearing=0.0), _radar_det(bearing=30.0)])
        self.assertGreaterEqual(len(self.fusion.aktif_izler), 1)
        self.fusion.temizle()
        self.assertEqual(len(self.fusion.aktif_izler), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
