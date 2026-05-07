import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class RadarParamsTests(unittest.TestCase):
    def test_aeris10_default_resolution_matches_physics(self):
        from src.radar_bridge import RadarParams

        p = RadarParams()
        # c / (2 * 200 MHz) = 0.75 m
        self.assertAlmostEqual(p.range_resolution_m, 0.75, places=2)
        # 3 km / 64 bin ≈ 46.875 m
        self.assertAlmostEqual(p.range_bin_size_m, 46.875, places=2)
        self.assertGreater(p.velocity_resolution_ms, 0.0)
        self.assertGreater(p.max_unambiguous_velocity_ms, 0.0)

    def test_aeris10e_max_range_scales_bin_size(self):
        from src.radar_bridge import RadarParams

        p = RadarParams(max_range_km=20.0)
        # 20 km / 64 bin = 312.5 m
        self.assertAlmostEqual(p.range_bin_size_m, 312.5, places=2)
        # Range resolution bandwidth'a bağlı, max_range'den bağımsız
        self.assertAlmostEqual(p.range_resolution_m, 0.75, places=2)

    def test_velocity_resolution_inversely_scales_with_doppler_bins(self):
        from src.radar_bridge import RadarParams

        p_low = RadarParams(num_doppler_bins=16)
        p_high = RadarParams(num_doppler_bins=64)
        self.assertGreater(p_low.velocity_resolution_ms, p_high.velocity_resolution_ms)


class PLFMAvailabilityTests(unittest.TestCase):
    def test_module_imports_without_plfm_repo(self):
        # Module-level import shouldn't blow up even if PLFM_AVAILABLE is False
        from src import radar_bridge

        self.assertIn(radar_bridge.PLFM_AVAILABLE, (True, False))
        # Shim constants must always be defined
        self.assertEqual(radar_bridge.NUM_RANGE_BINS, 64)
        self.assertEqual(radar_bridge.NUM_DOPPLER_BINS, 32)


class RadarDetectionTests(unittest.TestCase):
    def test_detection_to_dict_roundtrip(self):
        from src.radar_bridge import RadarDetection

        d = RadarDetection(
            range_bin=12, doppler_bin=18, range_km=1.5,
            velocity_ms=-50.0, bearing_deg=42.0,
            magnitude=1234.5, snr_db=18.0, timestamp=100.0,
        )
        out = d.to_dict()
        self.assertEqual(out["range_bin"], 12)
        self.assertAlmostEqual(out["range_km"], 1.5)
        self.assertAlmostEqual(out["velocity_ms"], -50.0)
        self.assertEqual(set(out.keys()),
                         {"range_bin", "doppler_bin", "range_km", "velocity_ms",
                          "bearing_deg", "magnitude", "snr_db", "timestamp"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
