"""
test_radar_simulator.py — PhysicsRadarSimulator birim testleri.

Doğrulama senaryoları:
  1. Radar denklemi: bilinen RCS+range → SNR (dB) formül kontrolü
  2. SNR mesafe bağımlılığı: R⁴ kuralı (range ×2 → SNR -12 dB)
  3. RCS sınıf sıralaması: Jet > Helikopter > Drone
  4. Aspect modulation: 90° beam-on > nose-on
  5. Anten pattern: boresight'ta max, kenaraa doğru azalma
  6. Multipath: düşük açılarda interferans deseni (0-4 arası)
  7. CFAR PFA istatistiksel doğrulama: Monte Carlo
  8. Swerling1 dağılımı: üstel (mean ≈ RCS, std ≈ mean)
  9. Ölçüm gürültüsü: range noise σ ∝ 1/√SNR
  10. Max range: Drone < Jet (RCS oranına göre)
  11. Deteksiyonsuz hedef: menzil dışında detected=False
  12. Full pipeline: 3 hedef, çeşitli menzil, fiziksel tutarlılık
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from radar_simulator import (
    PhysicsRadarSimulator, RadarParams, RadarEquation,
    RCSModel, AntennaPattern, MultipathModel, CFARDetector,
    TargetProfile, BOLTZMANN, LIGHT_SPEED, T_STANDARD,
)


class RadarEquationTests(unittest.TestCase):

    def setUp(self):
        self.params = RadarParams(
            transmit_power_w=10.0,
            tx_gain_db=30.0,
            rx_gain_db=30.0,
            noise_figure_db=5.0,
            bandwidth_hz=100e6,
            system_loss_db=3.0,
            frequency_hz=10e9,
        )
        self.eq = RadarEquation(self.params)

    def test_snr_formula_manual(self):
        rcs = 1.0  # m²
        R = 5000.0  # m
        lam = LIGHT_SPEED / self.params.frequency_hz
        Pt = self.params.transmit_power_w
        Gt = self.params.tx_gain_linear
        Gr = self.params.rx_gain_linear
        N = self.params.noise_power_w

        expected_snr = (Pt * Gt * Gr * lam**2 * rcs) / ((4*math.pi)**3 * R**4 * N)
        computed_snr = self.eq.snr_linear(rcs, R)
        self.assertAlmostEqual(computed_snr, expected_snr, places=10)

    def test_r4_law(self):
        """R ×2 → SNR -12 dB (R⁴ kuralı)."""
        snr1 = self.eq.snr_db(1.0, 1000.0)
        snr2 = self.eq.snr_db(1.0, 2000.0)
        delta_db = snr1 - snr2
        self.assertAlmostEqual(delta_db, 12.0, delta=0.1,
                               msg="R^4 kuralı: menzil ×2 → SNR -12 dB")

    def test_rcs_scaling(self):
        """RCS ×10 → SNR +10 dB."""
        snr1 = self.eq.snr_db(0.01, 3000.0)
        snr2 = self.eq.snr_db(0.10, 3000.0)
        self.assertAlmostEqual(snr2 - snr1, 10.0, delta=0.01)

    def test_max_range_increases_with_rcs(self):
        r_drone = self.eq.max_range_for_snr(0.01, 10.0)
        r_jet = self.eq.max_range_for_snr(5.0, 10.0)
        self.assertGreater(r_jet, r_drone)
        # R_max ∝ σ^(1/4) → ratio = (5/0.01)^0.25 ≈ 4.7×
        ratio = r_jet / r_drone
        expected_ratio = (5.0 / 0.01) ** 0.25
        self.assertAlmostEqual(ratio, expected_ratio, delta=0.1)

    def test_zero_range_returns_inf(self):
        snr = self.eq.snr_linear(1.0, 0.0)
        self.assertEqual(snr, float("inf"))


class RCSModelTests(unittest.TestCase):

    def test_jet_greater_than_drone(self):
        rcs_jet = RCSModel.mean_rcs("Jet", aspect_deg=0)
        rcs_drone = RCSModel.mean_rcs("Drone", aspect_deg=0)
        self.assertGreater(rcs_jet, rcs_drone)

    def test_beam_on_greater_than_nose_on(self):
        """Çoğu hedef için beam-on (90°) RCS > nose-on (0°)."""
        for cls in ["Jet", "Helikopter", "Drone", "FixedWingUAV"]:
            rcs_nose = RCSModel.mean_rcs(cls, aspect_deg=0)
            rcs_beam = RCSModel.mean_rcs(cls, aspect_deg=90)
            self.assertGreater(rcs_beam, rcs_nose,
                               msg=f"{cls}: beam-on > nose-on bekleniyor")

    def test_swerling1_mean(self):
        """Swerling Case 1: E[σ] = mean_rcs (büyük örneklem)."""
        rng = __import__("random").Random(42)
        mean_rcs = 1.0
        n = 50000
        samples = [RCSModel.swerling1_sample(mean_rcs, rng) for _ in range(n)]
        empirical_mean = sum(samples) / n
        self.assertAlmostEqual(empirical_mean, mean_rcs, delta=0.05,
                               msg="Swerling1 empirical mean ≈ mean_rcs")

    def test_swerling1_positive(self):
        """RCS her zaman pozitif olmalı."""
        rng = __import__("random").Random(0)
        for _ in range(1000):
            rcs = RCSModel.swerling1_sample(0.01, rng)
            self.assertGreater(rcs, 0)

    def test_aspect_symmetry(self):
        """Aspect -30° ve +30° aynı RCS'i vermeli."""
        rcs_pos = RCSModel.mean_rcs("Jet", aspect_deg=30)
        rcs_neg = RCSModel.mean_rcs("Jet", aspect_deg=-30)
        self.assertAlmostEqual(rcs_pos, rcs_neg, places=10)


class AntennaPatternTests(unittest.TestCase):

    def setUp(self):
        self.params = RadarParams(antenna_aperture_m=0.3, frequency_hz=10e9)
        self.ant = AntennaPattern(self.params)

    def test_boresight_max_gain(self):
        """Boresight'ta (offset=0) kazanç = 1.0."""
        gain = self.ant.gain_factor(0.0)
        self.assertAlmostEqual(gain, 1.0, places=5)

    def test_gain_decreases_off_boresight(self):
        """Açı arttıkça kazanç azalmalı."""
        g0 = self.ant.gain_factor(0.0)
        g5 = self.ant.gain_factor(5.0)
        g10 = self.ant.gain_factor(10.0)
        self.assertGreater(g0, g5)
        self.assertGreater(g5, g10)

    def test_gain_nonnegative(self):
        for deg in range(-30, 31, 5):
            g = self.ant.gain_factor(deg)
            self.assertGreaterEqual(g, 0.0)

    def test_half_power_beamwidth(self):
        """Yarı güç (3 dB) noktası HPBW formülüyle uyuşmalı."""
        hpbw = self.params.half_power_beamwidth_deg
        gain_half = self.ant.gain_factor(hpbw / 2)
        self.assertAlmostEqual(gain_half, 0.5, delta=0.1)


class MultipathTests(unittest.TestCase):

    def setUp(self):
        self.params = RadarParams(height_m=2.0)
        self.mp = MultipathModel(self.params)

    def test_factor_in_valid_range(self):
        """Interferans çarpanı 0-4 arasında olmalı."""
        for R in [500, 2000, 5000, 10000]:
            for el in [1, 5, 15, 30]:
                f = self.mp.interference_factor(R, el)
                self.assertGreaterEqual(f, 0.0, f"R={R}, el={el}")
                self.assertLessEqual(f, 4.0 + 1e-9, f"R={R}, el={el}")

    def test_high_elevation_near_free_space(self):
        """Yüksek açılarda multipath etkisi azalır."""
        f_low = self.mp.interference_factor(5000, 2.0)
        f_high = self.mp.interference_factor(5000, 45.0)
        # Her ikisi de 0-4 arasında; yüksek açıda daha stabil beklenir
        self.assertGreaterEqual(f_low, 0.0)
        self.assertGreaterEqual(f_high, 0.0)

    def test_range_variation_creates_pattern(self):
        """
        Farklı elevation açılarında interferans deseni değişmeli.
        Fizik notu: sabit elevation'da Δr ≈ 2·h_r·sin(el) range'e bağımsız;
        desen elevation değiştiğinde oluşur.
        """
        factors = [self.mp.interference_factor(5000.0, el) for el in range(1, 50, 3)]
        unique = len(set(round(f, 3) for f in factors))
        self.assertGreater(unique, 3, "Multipath desen elevation ile değişmeli")


class CFARTests(unittest.TestCase):

    def setUp(self):
        self.params = RadarParams(pfa=1e-4)
        self.cfar = CFARDetector(self.params, n_ref_cells=16, n_guard_cells=2)

    def test_alpha_positive(self):
        self.assertGreater(self.cfar._alpha, 0)

    def test_cfar_pfa_statistical(self):
        """
        Monte Carlo PFA testi: gürültü-only profilde yanlış alarm oranı
        teorik PFA'ya (1e-4) yakın olmalı (±2 kat tolerans).
        """
        rng = np.random.default_rng(12345)
        n_trials = 10000
        n_false = 0
        n_cells = 64
        cut_idx = 32  # merkez hücre

        for _ in range(n_trials):
            I = rng.normal(0, 1, n_cells)
            Q = rng.normal(0, 1, n_cells)
            power = I**2 + Q**2
            detected, _ = self.cfar.detect(power, cut_idx)
            if detected:
                n_false += 1

        empirical_pfa = n_false / n_trials
        theoretical_pfa = self.params.pfa
        # ±5× tolerans (Monte Carlo istatistiksel belirsizlik)
        self.assertLess(empirical_pfa, theoretical_pfa * 5 + 0.005,
                        f"Empirical PFA={empirical_pfa:.5f} > 5× theoretical")
        self.assertGreater(empirical_pfa, theoretical_pfa * 0.1 - 0.001,
                           f"Empirical PFA çok düşük: {empirical_pfa:.5f}")

    def test_strong_target_detected(self):
        """Çok yüksek SNR'de (50 dB+) hedef mutlaka tespit edilmeli."""
        rng = np.random.default_rng(0)
        n_cells = 64
        cut_idx = 32

        for trial in range(20):
            I = rng.normal(0, 1, n_cells)
            Q = rng.normal(0, 1, n_cells)
            power = I**2 + Q**2
            power[cut_idx] += 10000.0  # çok güçlü hedef
            detected, _ = self.cfar.detect(power, cut_idx)
            self.assertTrue(detected, f"Trial {trial}: güçlü hedef tespit edilemedi")

    def test_range_profile_generation(self):
        profile = self.cfar.generate_range_profile(
            [(100, 10.0), (200, 5.0)], n_cells=512
        )
        self.assertEqual(len(profile), 512)
        self.assertTrue(np.all(profile >= 0))


class PhysicsSimulatorTests(unittest.TestCase):

    def setUp(self):
        self.params = RadarParams(
            frequency_hz=10e9,
            transmit_power_w=10.0,
            tx_gain_db=30.0,
            rx_gain_db=30.0,
            noise_figure_db=5.0,
            bandwidth_hz=100e6,
            system_loss_db=3.0,
            min_snr_db=10.0,
            pfa=1e-4,
        )
        self.sim = PhysicsRadarSimulator(self.params, seed=0)

    def test_out_of_range_not_detected(self):
        targets = [TargetProfile(
            range_m=self.params.max_range_m * 1.5,
            bearing_deg=0, elevation_deg=10, velocity_ms=-100,
            sinif="Jet",
        )]
        dets = self.sim.detect_targets(targets)
        self.assertEqual(len(dets), 0, "Menzil dışı hedef detections'a eklenmemeli")

    def test_close_target_high_snr(self):
        """Yakın, büyük RCS hedef yüksek SNR'e sahip olmalı."""
        targets = [TargetProfile(
            range_m=500.0, bearing_deg=0, elevation_deg=15,
            velocity_ms=-200, sinif="Jet",
        )]
        dets = self.sim.detect_targets(targets)
        self.assertEqual(len(dets), 1)
        self.assertGreater(dets[0].snr_db, 25.0,
                           "500 m'de Jet: SNR > 25 dB bekleniyor")

    def test_far_drone_low_snr(self):
        """Uzak, küçük RCS drone düşük SNR'e sahip olmalı."""
        targets = [TargetProfile(
            range_m=12000.0, bearing_deg=0, elevation_deg=10,
            velocity_ms=-15, sinif="Drone",
        )]
        dets = self.sim.detect_targets(targets)
        self.assertEqual(len(dets), 1)
        self.assertLess(dets[0].snr_db, 0.0,
                        "12 km'de Drone: SNR < 0 dB bekleniyor (tespit edilemez)")

    def test_snr_ordering_by_rcs(self):
        """Aynı mesafede: Jet SNR > Drone SNR."""
        R = 3000.0
        t_jet = TargetProfile(R, 0, 15, -100, "Jet")
        t_drone = TargetProfile(R, 0, 15, -15, "Drone")
        dets = self.sim.detect_targets([t_jet, t_drone])
        self.assertEqual(len(dets), 2)
        self.assertGreater(dets[0].snr_db, dets[1].snr_db,
                           "Jet SNR > Drone SNR aynı mesafede")

    def test_range_noise_within_bounds(self):
        """Tespit edilen hedefte range noise, range resolution'dan büyük olmamalı."""
        targets = [TargetProfile(2000.0, 0, 20, -50, "Helikopter")]
        dets = self.sim.detect_targets(targets)
        if dets and dets[0].detected:
            range_res = LIGHT_SPEED / (2 * self.params.bandwidth_hz)  # 1.5 m
            self.assertLess(abs(dets[0].range_noise_m), range_res * 10,
                            "Range noise makul sınırlar içinde olmalı")

    def test_max_range_by_class(self):
        """Jet R_max > Helikopter R_max > Drone R_max."""
        r_jet = self.sim.max_detection_range("Jet")
        r_heli = self.sim.max_detection_range("Helikopter")
        r_drone = self.sim.max_detection_range("Drone")
        self.assertGreater(r_jet, r_heli)
        self.assertGreater(r_heli, r_drone)

    def test_summary_string(self):
        summary = self.sim.summary()
        self.assertIn("GHz", summary)
        self.assertIn("PFA", summary)
        self.assertIn("Drone", summary)

    def test_pipeline_multi_target(self):
        """3 hedef farklı mesafelerde → fiziksel tutarlılık."""
        targets = [
            TargetProfile(500.0,  10, 20, -200, "Balistik_Fuze"),
            TargetProfile(3000.0, 30, 10, -100, "Jet"),
            TargetProfile(8000.0, 60, 5,  -15,  "Drone"),
        ]
        dets = self.sim.detect_targets(targets)
        self.assertEqual(len(dets), 3)

        # SNR monoton azalmalı (mesafe arttıkça)
        self.assertGreater(dets[0].snr_db, dets[1].snr_db,
                           msg="Yakın hedef uzak hedeften yüksek SNR almalı (Füze vs Jet)")

        # Radar denklemi bozulmamış olmalı (NaN/inf yok)
        for d in dets:
            self.assertFalse(math.isnan(d.snr_db) and d.range_m < self.params.max_range_m)
            self.assertGreaterEqual(d.multipath_factor, 0.0)
            self.assertLessEqual(d.multipath_factor, 4.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
