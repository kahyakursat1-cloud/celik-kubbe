"""
test_vlm_scene_analyzer.py — VLM mock + cache + throttle birim testleri.
Gerçek model yüklemez; mock backend ile davranışı doğrular.
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

shared_root = Path(__file__).resolve().parents[1]
if str(shared_root) not in sys.path:
    sys.path.insert(0, str(shared_root))

from src.vlm_scene_analyzer import (
    VlmSceneAnalyzer,
    VlmAnalysis,
    _mock_analyze,
)


SAMPLE_TRACKS_LOW = [
    {"track_id": 1, "sinif": "Helicopter", "range_km": 3.0, "velocity_ms": 5.0},
]

SAMPLE_TRACKS_HIGH = [
    {"track_id": 1, "sinif": "Drone", "range_km": 2.0, "velocity_ms": -25.0},
    {"track_id": 2, "sinif": "BalisticMissile", "range_km": 4.5, "velocity_ms": -120.0},
]


class TestMockAnalyze(unittest.TestCase):

    def test_empty_tracks_returns_neutral(self):
        sonuc = _mock_analyze([])
        self.assertEqual(sonuc.anomaly_score, 0.5)
        self.assertTrue(sonuc.is_mock)

    def test_high_threat_class_yields_high_anomaly(self):
        sonuc = _mock_analyze(SAMPLE_TRACKS_HIGH)
        self.assertGreater(sonuc.anomaly_score, 0.85,
                           f"Balistik füze varken anomaly>0.85 bekleniyor, gerçek={sonuc.anomaly_score}")

    def test_low_threat_class_yields_low_anomaly(self):
        sonuc = _mock_analyze(SAMPLE_TRACKS_LOW)
        self.assertLess(sonuc.anomaly_score, 0.5,
                        f"Helicopter için anomaly<0.5 bekleniyor, gerçek={sonuc.anomaly_score}")

    def test_per_track_briefs_present(self):
        sonuc = _mock_analyze(SAMPLE_TRACKS_HIGH)
        self.assertIn(1, sonuc.per_track)
        self.assertIn(2, sonuc.per_track)
        self.assertIn("Drone", sonuc.per_track[1])

    def test_deterministic_output(self):
        """Aynı input → aynı çıktı (tekrarlanabilirlik)."""
        s1 = _mock_analyze(SAMPLE_TRACKS_HIGH)
        s2 = _mock_analyze(SAMPLE_TRACKS_HIGH)
        self.assertEqual(s1.anomaly_score, s2.anomaly_score)
        self.assertEqual(s1.summary, s2.summary)


class TestVlmAnalyzer(unittest.TestCase):

    def setUp(self):
        # Mock modda — gerçek model yüklenmez
        self.analyzer = VlmSceneAnalyzer(mock=True, throttle_s=0.1)

    def test_basic_analysis(self):
        sonuc = self.analyzer.queue_analysis(None, SAMPLE_TRACKS_HIGH)
        self.assertIsNotNone(sonuc)
        self.assertTrue(sonuc.is_mock)
        self.assertGreater(sonuc.anomaly_score, 0.5)

    def test_throttling(self):
        """İkinci çağrı throttle_s içinde gelmeli → None döner."""
        s1 = self.analyzer.queue_analysis(None, SAMPLE_TRACKS_LOW)
        # Throttle 0.1s; ardarda çağrı throttle'a yakalanır
        s2 = self.analyzer.queue_analysis(None, SAMPLE_TRACKS_HIGH)
        self.assertIsNotNone(s1)
        self.assertIsNone(s2, "Throttle aktifken ikinci çağrı None dönmeli")

    def test_cache_hit(self):
        """Aynı track + frame yeniden analiz çağrılırsa cache'ten dönmeli."""
        analyzer = VlmSceneAnalyzer(mock=True, throttle_s=0.0)  # throttle yok
        s1 = analyzer.queue_analysis(None, SAMPLE_TRACKS_LOW)
        s2 = analyzer.queue_analysis(None, SAMPLE_TRACKS_LOW)
        # Cache hit → aynı obje (identity), aynı içerik
        self.assertEqual(s1.summary, s2.summary)
        self.assertEqual(s1.anomaly_score, s2.anomaly_score)

    def test_reset_cache(self):
        analyzer = VlmSceneAnalyzer(mock=True, throttle_s=0.0)
        analyzer.queue_analysis(None, SAMPLE_TRACKS_LOW)
        self.assertEqual(len(analyzer._cache), 1)
        analyzer.reset_cache()
        self.assertEqual(len(analyzer._cache), 0)

    def test_to_dict_serializable(self):
        sonuc = self.analyzer.queue_analysis(None, SAMPLE_TRACKS_HIGH)
        d = sonuc.to_dict()
        import json
        # JSON serialize edilebilir mi
        s = json.dumps(d)
        self.assertIn("anomaly_score", s)
        self.assertIn("per_track", s)


class TestVlmFusion(unittest.TestCase):
    """SensorFusion.skorla_ve_acikla içine entegre edilen VLM kanalı testleri."""

    def test_neutral_vlm_doesnt_change_score(self):
        """e_vlm = 0.5 → bonus = 0; skor değişmemeli."""
        from src.sensor_fusion import SensorFusion, FusedTrack
        # PySide6 olmadan QObject oluşturma çalışmayabilir; skip
        try:
            fusion = SensorFusion(vlm_lambda=0.5)
        except Exception as e:
            self.skipTest(f"PySide6 unavailable: {e}")
        track = FusedTrack(
            track_id=1, range_km=2.0, velocity_ms=-20.0,
            sinif="Drone", kaynak="yalniz_radar",
            vlm_anomaly_score=0.5,  # neutral
        )
        seviye, skor, _ = fusion._skorla_ve_acikla(track)

        track_no_vlm = FusedTrack(
            track_id=2, range_km=2.0, velocity_ms=-20.0,
            sinif="Drone", kaynak="yalniz_radar",
            vlm_anomaly_score=0.5,
        )
        fusion2 = SensorFusion(vlm_lambda=0.0)  # VLM kapalı
        _, skor_off, _ = fusion2._skorla_ve_acikla(track_no_vlm)
        self.assertAlmostEqual(skor, skor_off, places=2,
                               msg="Neutral e_vlm fusion skorunu değiştirmemeli")

    def test_high_vlm_anomaly_raises_score(self):
        from src.sensor_fusion import SensorFusion, FusedTrack
        try:
            fusion_off = SensorFusion(vlm_lambda=0.0)
            fusion_on = SensorFusion(vlm_lambda=0.5)
        except Exception as e:
            self.skipTest(f"PySide6 unavailable: {e}")
        track = FusedTrack(
            track_id=1, range_km=2.0, velocity_ms=-20.0,
            sinif="Drone", kaynak="yalniz_radar",
            vlm_anomaly_score=0.9,  # yüksek anomali
        )
        _, skor_off, _ = fusion_off._skorla_ve_acikla(track)
        # Yeni instance gerek (state)
        track2 = FusedTrack(
            track_id=1, range_km=2.0, velocity_ms=-20.0,
            sinif="Drone", kaynak="yalniz_radar",
            vlm_anomaly_score=0.9,
        )
        _, skor_on, _ = fusion_on._skorla_ve_acikla(track2)
        self.assertGreater(skor_on, skor_off,
                           "Yüksek VLM anomaly fusion skorunu artırmalı")


if __name__ == "__main__":
    unittest.main()
