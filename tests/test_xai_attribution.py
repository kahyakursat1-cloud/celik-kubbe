"""
test_xai_attribution.py — AlphaPolicy + ThreatExplainer birim testleri.
Mevcut kural-tabanlı _tehdit_hesapla davranışını koruduğunu doğrular
(baseline α'larla) + dinamik α güncellemelerinin makale Tablo 4 ile
tutarlı olduğunu kontrol eder.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from dataclasses import dataclass

# src import için path
shared_root = Path(__file__).resolve().parents[1]
if str(shared_root) not in sys.path:
    sys.path.insert(0, str(shared_root))

from src.xai_attribution import (
    AlphaPolicy,
    FusionContext,
    ThreatExplainer,
    hesapla_ham_bilesenler,
)


@dataclass
class _MockTrack:
    """Minimal track stub (FusedTrack subset)."""
    range_km: float = 2.0
    velocity_ms: float = -30.0
    sinif: str = "Drone"
    kaynak: str = "yalniz_radar"


class TestAlphaPolicy(unittest.TestCase):

    def test_baseline_alpha_normalizes_to_one(self):
        policy = AlphaPolicy()
        a_r, a_v, a_c = policy.compute(FusionContext())
        self.assertAlmostEqual(a_r + a_v + a_c, 1.0, places=5)

    def test_baseline_alpha_values(self):
        policy = AlphaPolicy()
        a_r, a_v, a_c = policy.compute(FusionContext())
        # Baseline 0.5, 0.3, 0.2 — herhangi bir kural tetiklenmediği için normalize sonrası aynı
        self.assertAlmostEqual(a_r, 0.5, places=2)
        self.assertAlmostEqual(a_v, 0.3, places=2)
        self.assertAlmostEqual(a_c, 0.2, places=2)

    def test_multi_threat_increases_velocity_weight(self):
        policy = AlphaPolicy()
        ctx_low = FusionContext(aktif_track_sayisi=1)
        ctx_high = FusionContext(aktif_track_sayisi=5)  # multi_threat tetikler
        _, av_low, _ = policy.compute(ctx_low)
        _, av_high, _ = policy.compute(ctx_high)
        self.assertGreater(av_high, av_low, "Multi-threat α_v artmalı")

    def test_radar_dropout_increases_camera_weight(self):
        policy = AlphaPolicy()
        ctx_normal = FusionContext()
        ctx_dropout = FusionContext(radar_dropout=True)
        _, _, ac_normal = policy.compute(ctx_normal)
        _, _, ac_dropout = policy.compute(ctx_dropout)
        self.assertGreater(ac_dropout, ac_normal, "Radar dropout α_c artmalı")

    def test_alpha_clipping(self):
        """α toplamı 1, hiçbiri [0.1, 0.8] dışına çıkmamalı (normalize öncesi)."""
        policy = AlphaPolicy(clip_min=0.1, clip_max=0.8)
        ctx_extreme = FusionContext(
            aktif_track_sayisi=10,
            radar_dropout=True,
            kamera_dropout=True,
            ortalama_snr_db=2.0,
        )
        a_r, a_v, a_c = policy.compute(ctx_extreme)
        for a in (a_r, a_v, a_c):
            self.assertGreaterEqual(a, 0.0)
            self.assertLessEqual(a, 1.0)
        self.assertAlmostEqual(a_r + a_v + a_c, 1.0, places=5)


class TestThreatExplainer(unittest.TestCase):

    def test_low_threat_drone_far(self):
        track = _MockTrack(range_km=5.0, velocity_ms=0.0, sinif="Drone")
        explainer = ThreatExplainer()
        attr = explainer.explain(track)
        self.assertLess(attr.score, 30.0, "Uzak ve durağan drone DÜŞÜK olmalı")

    def test_critical_ballistic_missile(self):
        track = _MockTrack(range_km=1.0, velocity_ms=-150.0, sinif="BalisticMissile",
                           kaynak="fuzyon")
        explainer = ThreatExplainer()
        attr = explainer.explain(track)
        # Yakın + hızlı yaklaşan + füze + fusion bonus → yüksek skor
        self.assertGreater(attr.score, 50.0,
                           f"Balistik füze YÜKSEK olmalı, score={attr.score}")

    def test_dominant_factor_for_close_target(self):
        """Yakın bir hedef için range faktörü dominant olmalı."""
        track = _MockTrack(range_km=0.5, velocity_ms=0.0, sinif="Drone")
        attr = ThreatExplainer().explain(track)
        self.assertEqual(attr.dominant_factor, "range")

    def test_dominant_factor_for_high_class_threat(self):
        """Düşük hız + uzak ama balistik füze → class dominant."""
        track = _MockTrack(range_km=4.5, velocity_ms=0.0, sinif="BalisticMissile")
        attr = ThreatExplainer().explain(track)
        self.assertEqual(attr.dominant_factor, "class")

    def test_shap_components_sum_to_score(self):
        """Linear additive: φ_r + φ_v + φ_c + φ_fusion = score (clip içinde)."""
        track = _MockTrack(range_km=2.0, velocity_ms=-20.0, sinif="Drone",
                           kaynak="yalniz_radar")
        attr = ThreatExplainer().explain(track)
        toplam = attr.phi_r + attr.phi_v + attr.phi_c + attr.phi_fusion
        self.assertAlmostEqual(toplam, attr.score, places=2)

    def test_dynamic_alpha_changes_score(self):
        """Aynı track, farklı bağlamda farklı skor üretmeli."""
        track = _MockTrack(range_km=3.0, velocity_ms=-20.0, sinif="Drone")
        explainer = ThreatExplainer()
        ctx_normal = FusionContext(aktif_track_sayisi=1)
        ctx_multi = FusionContext(aktif_track_sayisi=5)
        skor_normal = explainer.explain(track, ctx_normal).score
        skor_multi = explainer.explain(track, ctx_multi).score
        # Multi-threat'te α_v artar → velocity bileşeni daha ağır
        # Hareket eden drone için skor değişmeli
        self.assertNotAlmostEqual(skor_normal, skor_multi, places=1)

    def test_label_thresholds(self):
        track_kritik = _MockTrack(range_km=0.5, velocity_ms=-150.0,
                                  sinif="BalisticMissile", kaynak="fuzyon")
        track_dusuk = _MockTrack(range_km=4.9, velocity_ms=0.0, sinif="Drone")
        explainer = ThreatExplainer()
        seviye_k, skor_k, _ = explainer.explain_with_label(track_kritik)
        seviye_d, skor_d, _ = explainer.explain_with_label(track_dusuk)
        self.assertIn(seviye_k, {"YÜKSEK", "KRİTİK"})
        self.assertEqual(seviye_d, "DÜŞÜK")


class TestHamBilesenler(unittest.TestCase):

    def test_velocity_below_threshold_zero(self):
        """v_raw threshold (-10 m/s) altında 0 olmalı."""
        track = _MockTrack(velocity_ms=-5.0)
        bilesen = hesapla_ham_bilesenler(track)
        self.assertEqual(bilesen.v_raw, 0.0)

    def test_fusion_bonus_only_when_fused(self):
        track_solo = _MockTrack(kaynak="yalniz_radar")
        track_fused = _MockTrack(kaynak="fuzyon")
        b_solo = hesapla_ham_bilesenler(track_solo)
        b_fused = hesapla_ham_bilesenler(track_fused)
        self.assertEqual(b_solo.fusion_bonus, 0.0)
        self.assertGreater(b_fused.fusion_bonus, 0.0)


if __name__ == "__main__":
    unittest.main()
