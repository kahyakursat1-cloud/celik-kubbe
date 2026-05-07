"""
Celik Kubbe current-module integration checks.

Run:
    python src/test_celikkubbe_pipeline.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ConfigTests(unittest.TestCase):
    def test_config_contains_runtime_sections(self):
        from src.konfig import cfg

        for section in ("model", "kamera", "radar", "fuzyon", "gimbal", "loglama"):
            self.assertIn(section, cfg)

        self.assertGreater(cfg["model"]["guven_esik"], 0.0)
        self.assertGreater(cfg["radar"]["maks_menzil_km"], 0.0)


class ModelFileTests(unittest.TestCase):
    def test_configured_model_files_exist(self):
        from src.konfig import cfg

        model_path = PROJECT_ROOT / cfg["model"]["yolu"]
        onnx_path = PROJECT_ROOT / cfg["model"]["onnx_yolu"]

        self.assertTrue(model_path.is_file(), f"Model file missing: {model_path}")
        self.assertTrue(onnx_path.is_file(), f"ONNX file missing: {onnx_path}")


class RadarAndCoordinateTests(unittest.TestCase):
    def test_radar_params_produce_physical_resolution(self):
        from src.radar_bridge import RadarParams

        params = RadarParams(max_range_km=3.0, bandwidth_hz=200e6, prf_hz=5000)

        self.assertGreater(params.range_resolution_m, 0.0)
        self.assertGreater(params.range_bin_size_m, 0.0)
        self.assertGreater(params.velocity_resolution_ms, 0.0)

    def test_coordinate_helpers_preserve_physical_range(self):
        from src.coordinate_utils import display_radius_to_km, km_to_display_radius

        radius = km_to_display_radius(2.25, max_range_km=3.0)

        self.assertLessEqual(radius, 0.95)
        self.assertAlmostEqual(display_radius_to_km(radius, max_range_km=3.0), 2.25)


class WTATests(unittest.TestCase):
    def test_profiled_battery_can_engage_in_range_threat(self):
        from src.battery_profiles import profile_for_battery
        from src.wta_optimizer import BatteryState, ThreatState, WTAOptimizer

        profile = profile_for_battery("PİL-ALFA")
        assignments = WTAOptimizer.optimize(
            [
                BatteryState(
                    "BAT-ALFA",
                    ammo=1,
                    max_range_km=profile.max_range_km,
                    prob_kill=profile.prob_kill,
                )
            ],
            [ThreatState("THR-001", range_km=10.0, threat_score=90.0)],
        )

        self.assertEqual(assignments, [("BAT-ALFA", "THR-001")])


class SensorFusionTests(unittest.TestCase):
    def test_radar_and_camera_inputs_create_fused_track(self):
        from src.sensor_fusion import SensorFusion

        fusion = SensorFusion()
        fusion.radar_girdisi([
            {
                "bearing_deg": 0.0,
                "range_km": 1.0,
                "velocity_ms": -60.0,
                "snr_db": 22.0,
            }
        ])
        fusion.kamera_girdisi([
            {
                "track_id": 7,
                "sinif": "Missile",
                "guven": 0.92,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.35,
                "h": 0.35,
            }
        ])

        tracks = list(fusion.aktif_izler.values())
        self.assertTrue(tracks)
        self.assertEqual(tracks[0].kaynak, "fuzyon")
        self.assertGreaterEqual(tracks[0].tehdit_skoru, 80.0)


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    unittest.main(verbosity=2)
