import unittest


class CoordinateUtilsTests(unittest.TestCase):
    def test_physical_range_round_trips_through_display_radius(self):
        from src.coordinate_utils import km_to_display_radius, display_radius_to_km

        radius = km_to_display_radius(1.5, max_range_km=3.0)

        self.assertAlmostEqual(radius, 0.475)
        self.assertAlmostEqual(display_radius_to_km(radius, max_range_km=3.0), 1.5)

    def test_display_radius_is_clamped_to_scope_bounds(self):
        from src.coordinate_utils import km_to_display_radius

        self.assertEqual(km_to_display_radius(-2.0, max_range_km=3.0), 0.0)
        self.assertEqual(km_to_display_radius(99.0, max_range_km=3.0), 0.95)


class BatteryProfileTests(unittest.TestCase):
    def test_known_battery_names_have_distinct_profiles(self):
        from src.battery_profiles import profile_for_battery

        alfa = profile_for_battery("PIL-ALFA")
        gamma = profile_for_battery("PIL-GAMMA")

        self.assertGreater(alfa.max_range_km, gamma.max_range_km)
        self.assertGreater(alfa.prob_kill, 0.8)
        self.assertGreater(gamma.prob_kill, 0.7)

    def test_profile_lookup_handles_turkish_display_names(self):
        from src.battery_profiles import profile_for_battery

        beta = profile_for_battery("PİL-BETA")

        self.assertEqual(beta.profile_id, "PIL-BETA")
        self.assertGreaterEqual(beta.max_range_km, 10.0)


class WTAOptimizerTests(unittest.TestCase):
    def test_optimizer_does_not_assign_out_of_range_targets(self):
        from src.wta_optimizer import BatteryState, ThreatState, WTAOptimizer

        assignments = WTAOptimizer.optimize(
            [BatteryState("BAT-0", ammo=1, max_range_km=3.0, prob_kill=0.9)],
            [ThreatState("THR-001", range_km=8.0, threat_score=100.0)],
        )

        self.assertEqual(assignments, [])


class SharedPanelDataTests(unittest.TestCase):
    def test_celik_kubbe_panel_mentions_current_runtime_modules(self):
        import sys
        from pathlib import Path

        shared_root = Path(__file__).resolve().parents[2] / "shared"
        if str(shared_root) not in sys.path:
            sys.path.insert(0, str(shared_root))

        from yarismalar_verisi import PROJELER

        veri = PROJELER["celik_kubbe"]
        donanim_text = " ".join(item["bilesen"] for item in veri["donanim"])
        yazilim_text = " ".join(item["ad"] for item in veri["yazilim"])
        ister_text = " ".join(item["baslik"] + " " + item["aciklama"] for item in veri["isterler"])

        self.assertIn("AERIS-10", donanim_text)
        self.assertIn("Gimbal", donanim_text)
        self.assertIn("SensorFusion", yazilim_text)
        self.assertIn("WTA Optimizer", yazilim_text)
        self.assertIn("Gerçek Menzil", ister_text)
        self.assertIn("Batarya Profilleri", ister_text)

    def test_celik_kubbe_hardware_bom_is_detailed_and_priced(self):
        import sys
        from pathlib import Path

        shared_root = Path(__file__).resolve().parents[2] / "shared"
        if str(shared_root) not in sys.path:
            sys.path.insert(0, str(shared_root))

        from yarismalar_verisi import PROJELER

        donanim = PROJELER["celik_kubbe"]["donanim"]
        toplam_tl = sum(item["adet"] * item["birim_fiyat_tl"] for item in donanim)

        self.assertGreaterEqual(len(donanim), 16)
        self.assertGreaterEqual(toplam_tl, 500_000)
        self.assertTrue(all(item["birim_fiyat_tl"] > 0 for item in donanim))
        self.assertTrue(any("FT2232H" in item["model"] for item in donanim))
        self.assertTrue(any("30x" in item["model"] for item in donanim))

    def test_celik_kubbe_hardware_bom_marks_need_and_reason(self):
        import sys
        from pathlib import Path

        shared_root = Path(__file__).resolve().parents[2] / "shared"
        if str(shared_root) not in sys.path:
            sys.path.insert(0, str(shared_root))

        from yarismalar_verisi import PROJELER

        donanim = PROJELER["celik_kubbe"]["donanim"]
        gereklilikler = {item.get("gereklilik") for item in donanim}

        self.assertIn("Zorunlu", gereklilikler)
        self.assertIn("Opsiyonel", gereklilikler)
        self.assertIn("Yedek", gereklilikler)
        self.assertTrue(all(item.get("gerekce") for item in donanim))

    def test_shared_panel_displays_need_and_reason_columns(self):
        from pathlib import Path

        panel_source = (Path(__file__).resolve().parents[2] / "shared" / "bilgi_paneli.py").read_text(encoding="utf-8")

        self.assertIn("Gereklilik", panel_source)
        self.assertIn("Neden Gerekli", panel_source)


class MainUiLayoutTests(unittest.TestCase):
    def test_right_status_panel_is_not_locked_to_narrow_width(self):
        from pathlib import Path

        main_source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")

        self.assertNotIn("setFixedWidth(260)", main_source)
        self.assertIn("setMinimumWidth(560)", main_source)
        self.assertIn("columns = QHBoxLayout()", main_source)


if __name__ == "__main__":
    unittest.main()
