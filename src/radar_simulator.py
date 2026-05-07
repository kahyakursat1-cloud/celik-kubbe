"""
src/radar_simulator.py — Fizik tabanlı radar simülatörü.

PLFM mock'unun üzerine gerçekçi bir layer:
  - RCS modeli (hedef sınıfı + aspect angle)
  - Radar denklemi → SNR (dB)
  - Range/Doppler AWGN gürültüsü
  - Multipath (yer yansıması, deterministic interference)
  - Sinc² anten pattern (yön bağımlı kazanç)
  - CA-CFAR dedektörü (gerçek PFA kontrolü)

Referanslar:
  Skolnik, "Introduction to Radar Systems", 3rd ed., McGraw-Hill 2001.
  Richards, "Fundamentals of Radar Signal Processing", 2nd ed., McGraw-Hill 2014.

Kullanım:
    from src.radar_simulator import PhysicsRadarSimulator, RadarParams, TargetProfile
    params = RadarParams()
    sim = PhysicsRadarSimulator(params, seed=0)
    detections = sim.detect_targets([
        TargetProfile(range_m=5000, bearing_deg=30, elevation_deg=10,
                      velocity_ms=-120, sinif="Balistik_Fuze"),
    ])
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ─── Fiziksel sabitler ───────────────────────────────────────────────────────
BOLTZMANN = 1.380649e-23   # J/K
LIGHT_SPEED = 3.0e8         # m/s
T_STANDARD = 290.0          # K (standart referans sıcaklık)


# ─── Hedef sınıfı RCS tablosu ─────────────────────────────────────────────────
# Ortalama RCS (m²), nose-on aspect. Skolnik Tablo 2-1'den uyarlandı.
RCS_TABLE: dict[str, float] = {
    "Drone":          0.01,
    "FixedWingUAV":   0.03,
    "Balistik_Fuze":  0.05,
    "Helikopter":     1.5,
    "Artillery":      0.08,
    "Jet":            5.0,
    "Bilinmeyen":     0.1,
}

# Aspect-angle RCS modulation: [nose=0°, beam=90°, tail=180°]
# Normalize faktörler (gerçek sistemlerde genellikle 3-10 dB değişim)
_ASPECT_MOD = {
    "Drone":         [1.0, 2.5, 0.8],
    "FixedWingUAV":  [1.0, 3.0, 0.9],
    "Balistik_Fuze": [1.0, 1.5, 0.7],
    "Helikopter":    [2.0, 4.0, 2.0],   # blade modulation yüksek
    "Artillery":     [1.0, 2.0, 0.5],
    "Jet":           [1.0, 3.5, 1.2],
    "Bilinmeyen":    [1.0, 2.0, 1.0],
}


@dataclass
class RadarParams:
    """AERIS-10 tipi X-band FMCW radar parametreleri."""
    frequency_hz: float = 10.0e9       # 10 GHz (X-band)
    transmit_power_w: float = 10.0     # İletim gücü
    tx_gain_db: float = 30.0          # Verici anten kazancı
    rx_gain_db: float = 30.0          # Alıcı anten kazancı
    noise_figure_db: float = 5.0      # Gürültü figürü
    bandwidth_hz: float = 100.0e6     # 100 MHz → 1.5 m range resolution
    system_loss_db: float = 3.0       # Sistem kayıpları
    antenna_aperture_m: float = 0.3   # Anten açıklığı (beam width için)
    pfa: float = 1e-6                 # İstenilen yanlış alarm olasılığı
    max_range_m: float = 20_000.0     # Maksimum menzil
    min_snr_db: float = 10.0         # Min SNR for detection (CFAR eşiği altı için)
    height_m: float = 2.0            # Radar yüksekliği (multipath için)

    @property
    def wavelength_m(self) -> float:
        return LIGHT_SPEED / self.frequency_hz

    @property
    def tx_gain_linear(self) -> float:
        return 10 ** (self.tx_gain_db / 10)

    @property
    def rx_gain_linear(self) -> float:
        return 10 ** (self.rx_gain_db / 10)

    @property
    def noise_figure_linear(self) -> float:
        return 10 ** (self.noise_figure_db / 10)

    @property
    def system_loss_linear(self) -> float:
        return 10 ** (self.system_loss_db / 10)

    @property
    def noise_power_w(self) -> float:
        return (BOLTZMANN * T_STANDARD * self.bandwidth_hz
                * self.noise_figure_linear * self.system_loss_linear)

    @property
    def half_power_beamwidth_deg(self) -> float:
        return math.degrees(0.886 * self.wavelength_m / self.antenna_aperture_m)


@dataclass
class TargetProfile:
    range_m: float
    bearing_deg: float
    elevation_deg: float
    velocity_ms: float          # Radial hız (negatif = yaklaşıyor)
    sinif: str = "Bilinmeyen"
    aspect_deg: float = 0.0     # Aspect angle (0=nose-on)

    @property
    def range_km(self) -> float:
        return self.range_m / 1000.0


# ─── Alt modüller ─────────────────────────────────────────────────────────────

class RCSModel:
    """
    Hedef sınıfı + aspect angle → RCS (m²).

    Aspect interpolation: Swerling Case 1 (slow fluctuation, exponential pdf).
    """

    @staticmethod
    def mean_rcs(sinif: str, aspect_deg: float = 0.0) -> float:
        base = RCS_TABLE.get(sinif, 0.1)
        mods = _ASPECT_MOD.get(sinif, [1.0, 2.0, 1.0])
        a = abs(aspect_deg) % 180.0
        if a <= 90.0:
            t = a / 90.0
            mod = mods[0] * (1 - t) + mods[1] * t
        else:
            t = (a - 90.0) / 90.0
            mod = mods[1] * (1 - t) + mods[2] * t
        return base * mod

    @staticmethod
    def swerling1_sample(mean_rcs: float, rng: random.Random) -> float:
        """
        Swerling Case 1: Exponential dağılım (chi-squared, 2 serbestlik derecesi).
        E[σ] = mean_rcs, Var büyük → gerçek hedef dalgalanması.
        """
        return rng.expovariate(1.0 / mean_rcs)


class AntennaPattern:
    """
    Sinc² anten pattern → bearing offset'e göre kazanç.
    Tek boyutlu uniform linear aperture modeli.
    """

    def __init__(self, params: RadarParams):
        self._params = params

    def gain_factor(self, bearing_offset_deg: float) -> float:
        """
        Anten boresight'ından sapma → kazanç faktörü (0-1, lineer).
        sinc²(π × D × sin(θ) / λ) formülü.
        """
        D = self._params.antenna_aperture_m
        lam = self._params.wavelength_m
        theta = math.radians(bearing_offset_deg)
        x = math.pi * D * math.sin(theta) / lam
        sinc_val = math.sin(x) / x if abs(x) > 1e-9 else 1.0
        return sinc_val ** 2


class MultipathModel:
    """
    Basit yer yansıması multipath modeli (flat earth, specular reflection).

    Toplam alan = direkt yol + image hedef yansıması.
    Interferans çarpanı F = |1 + Γ × exp(-j × Δφ)|²
    Γ = yer yansıma katsayısı (yaklaşık -1 düz yüzey için)

    Referans: Skolnik §12-2.
    """

    def __init__(self, params: RadarParams, ground_reflection_coeff: float = -0.9):
        self._h_r = params.height_m  # Radar yüksekliği
        self._gamma = ground_reflection_coeff
        self._lam = params.wavelength_m

    def interference_factor(self, range_m: float, elevation_deg: float) -> float:
        """Multipath interferans çarpanı (power, 0-4 arası)."""
        h_t_approx = range_m * math.tan(math.radians(max(elevation_deg, 0.1)))
        delta_r = 2.0 * self._h_r * h_t_approx / max(range_m, 1.0)
        delta_phi = 2 * math.pi * delta_r / self._lam
        real_part = 1.0 + self._gamma * math.cos(delta_phi)
        imag_part = self._gamma * math.sin(delta_phi)
        F_squared = real_part**2 + imag_part**2
        return float(np.clip(F_squared, 0.0, 4.0))


class RadarEquation:
    """
    Radar denklemi: SNR_linear = (P_t × G_t × G_r × λ² × σ) / ((4π)³ × R⁴ × N)
    N = k_B × T₀ × B × F × L
    """

    def __init__(self, params: RadarParams):
        self._p = params

    def snr_linear(
        self,
        rcs_m2: float,
        range_m: float,
        antenna_gain_factor: float = 1.0,
        multipath_factor: float = 1.0,
    ) -> float:
        if range_m <= 0:
            return float("inf")
        lam = self._p.wavelength_m
        numerator = (
            self._p.transmit_power_w
            * self._p.tx_gain_linear
            * self._p.rx_gain_linear
            * lam**2
            * rcs_m2
            * antenna_gain_factor
            * multipath_factor
        )
        denominator = (
            (4 * math.pi)**3
            * range_m**4
            * self._p.noise_power_w
        )
        return numerator / denominator if denominator > 0 else float("inf")

    def snr_db(
        self,
        rcs_m2: float,
        range_m: float,
        antenna_gain_factor: float = 1.0,
        multipath_factor: float = 1.0,
    ) -> float:
        snr = self.snr_linear(rcs_m2, range_m, antenna_gain_factor, multipath_factor)
        if snr <= 0:
            return -float("inf")
        return 10 * math.log10(snr)

    def max_range_for_snr(self, rcs_m2: float, snr_db_min: float) -> float:
        """Verilen minimum SNR için maksimum menzil."""
        snr_lin = 10 ** (snr_db_min / 10)
        lam = self._p.wavelength_m
        numerator = (
            self._p.transmit_power_w
            * self._p.tx_gain_linear
            * self._p.rx_gain_linear
            * lam**2
            * rcs_m2
        )
        denominator = (4 * math.pi)**3 * self._p.noise_power_w * snr_lin
        if denominator <= 0:
            return 0.0
        return (numerator / denominator) ** 0.25


class CFARDetector:
    """
    CA-CFAR (Cell Averaging Constant False Alarm Rate) dedektörü.

    N_ref referans hücresinin ortalamasından threshold belirler:
    T = α × Z  (Z = referans hücrelerinin ortalaması)
    α = N_ref × (PFA^(-1/N_ref) - 1)   [Skolnik Eq. 8.32]

    Verilen PFA için α katsayısını hesaplar.
    """

    def __init__(self, params: RadarParams, n_ref_cells: int = 16, n_guard_cells: int = 2):
        self._pfa = params.pfa
        self._n_ref = n_ref_cells
        self._n_guard = n_guard_cells
        self._alpha = self._compute_alpha()

    def _compute_alpha(self) -> float:
        """CA-CFAR threshold multiplier."""
        N = self._n_ref
        pfa = self._pfa
        return N * (pfa ** (-1.0 / N) - 1.0)

    def detect(
        self,
        range_cells: np.ndarray,
        cut_idx: int,
    ) -> tuple[bool, float]:
        """
        Tek CUT (Cell Under Test) için CFAR kararı.

        range_cells: güç değerleri array (rasgele gürültü + sinyal)
        cut_idx: test edilecek hücre indeksi

        Döndürür: (detected: bool, threshold: float)
        """
        N = self._n_ref
        G = self._n_guard
        total = len(range_cells)
        left_start = max(0, cut_idx - G - N // 2)
        left_end = max(0, cut_idx - G)
        right_start = min(total, cut_idx + G + 1)
        right_end = min(total, cut_idx + G + 1 + N // 2)

        ref = np.concatenate([
            range_cells[left_start:left_end],
            range_cells[right_start:right_end],
        ])
        if len(ref) == 0:
            return False, float("inf")

        Z = float(ref.mean())
        threshold = self._alpha * Z
        detected = float(range_cells[cut_idx]) > threshold
        return bool(detected), threshold

    def generate_range_profile(
        self,
        targets: list[tuple[int, float]],
        n_cells: int = 512,
        noise_std: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Sentetik range profile üretir.
        targets: [(cell_idx, amplitude), ...]
        Çıktı: güç değerleri (eksponansyel dağılım, Rayleigh envelope)
        """
        if rng is None:
            rng = np.random.default_rng(0)
        # Rayleigh gürültü: her I ve Q Gaussian → güç = I²+Q²
        I = rng.normal(0, noise_std, n_cells)
        Q = rng.normal(0, noise_std, n_cells)
        power = I**2 + Q**2  # Exponential distribution

        for cell_idx, amplitude in targets:
            if 0 <= cell_idx < n_cells:
                I_sig = rng.normal(amplitude, noise_std * 0.1)
                Q_sig = rng.normal(amplitude, noise_std * 0.1)
                power[cell_idx] += I_sig**2 + Q_sig**2
        return power


# ─── Ana simülatör ────────────────────────────────────────────────────────────

@dataclass
class RadarDetection:
    target_id: int
    range_m: float
    bearing_deg: float
    velocity_ms: float
    snr_db: float
    detected: bool
    cfar_threshold: float
    rcs_used_m2: float
    multipath_factor: float
    antenna_gain_factor: float
    range_noise_m: float = 0.0
    velocity_noise_ms: float = 0.0

    @property
    def range_km(self) -> float:
        return self.range_m / 1000.0


class PhysicsRadarSimulator:
    """
    Fizik tabanlı radar simülatörü.

    PLFM mock'unun yerini alacak şekilde tasarlandı:
    - RadarBridge.mock=True yerine bu simülatör kullanılabilir
    - eval/ altyapısıyla entegre: TargetProfile listesi → RadarDetection listesi

    Gürültü kaynakları:
      1. Sinyal dalgalanması: Swerling Case 1 (yavaş dalgalanma)
      2. Range gürültüsü: AWGN, σ = c / (2B) ≈ range resolution / SNR^0.5
      3. Velocity gürültüsü: Doppler resolution'dan kaynaklı
      4. CA-CFAR: false alarm kontrolü
    """

    def __init__(
        self,
        params: Optional[RadarParams] = None,
        boresight_bearing_deg: float = 0.0,
        seed: int = 0,
    ):
        self.params = params or RadarParams()
        self.boresight = boresight_bearing_deg
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._radar_eq = RadarEquation(self.params)
        self._rcs_model = RCSModel()
        self._antenna = AntennaPattern(self.params)
        self._multipath = MultipathModel(self.params)
        self._cfar = CFARDetector(self.params)

    def _range_resolution(self) -> float:
        return LIGHT_SPEED / (2 * self.params.bandwidth_hz)

    def _velocity_resolution(self, coherent_time_s: float = 0.01) -> float:
        return self.params.wavelength_m / (2 * coherent_time_s)

    def detect_targets(
        self,
        targets: list[TargetProfile],
        use_cfar: bool = True,
    ) -> list[RadarDetection]:
        """
        Hedef listesini radar fiziğiyle işleyip detections döndürür.

        Parametreler
        -----------
        targets: TargetProfile listesi
        use_cfar: True → CFAR kararı; False → sadece SNR eşiği

        Döndürür
        --------
        list[RadarDetection]: her hedef için sonuç (detected=True/False dahil)
        """
        detections = []
        range_res = self._range_resolution()
        vel_res = self._velocity_resolution()

        # Range profile oluştur (CFAR için)
        n_cells = 512
        range_profile = self._cfar.generate_range_profile(
            [], n_cells=n_cells, noise_std=1.0, rng=self._np_rng
        )

        for i, t in enumerate(targets):
            if t.range_m <= 0 or t.range_m > self.params.max_range_m:
                continue

            # 1. RCS (Swerling Case 1 dalgalanma)
            mean_rcs = self._rcs_model.mean_rcs(t.sinif, t.aspect_deg)
            rcs = self._rcs_model.swerling1_sample(mean_rcs, self._rng)

            # 2. Anten pattern
            bearing_offset = (t.bearing_deg - self.boresight + 180) % 360 - 180
            ant_gain = self._antenna.gain_factor(bearing_offset)

            # 3. Multipath
            mp_factor = self._multipath.interference_factor(
                t.range_m, t.elevation_deg
            )

            # 4. SNR hesapla
            snr_lin = self._radar_eq.snr_linear(rcs, t.range_m, ant_gain, mp_factor)
            snr_db_val = (10 * math.log10(snr_lin)
                          if snr_lin > 0 else -float("inf"))

            # 5. CFAR kararı
            cell_idx = int((t.range_m / self.params.max_range_m) * n_cells)
            cell_idx = max(0, min(cell_idx, n_cells - 1))

            if use_cfar:
                signal_amplitude = math.sqrt(max(snr_lin, 0)) * 2.0
                range_profile_copy = range_profile.copy()
                I_s = self._np_rng.normal(signal_amplitude, 0.1)
                Q_s = self._np_rng.normal(signal_amplitude, 0.1)
                range_profile_copy[cell_idx] += I_s**2 + Q_s**2
                detected, threshold = self._cfar.detect(range_profile_copy, cell_idx)
            else:
                detected = snr_db_val >= self.params.min_snr_db
                threshold = float("nan")

            # 6. Ölçüm gürültüsü (sadece detected ise)
            if detected:
                range_std = range_res / max(math.sqrt(snr_lin), 0.1)
                vel_std = vel_res / max(math.sqrt(snr_lin), 0.1) * 0.5
                range_noise = self._rng.gauss(0, range_std)
                vel_noise = self._rng.gauss(0, vel_std)
            else:
                range_noise = 0.0
                vel_noise = 0.0

            detections.append(RadarDetection(
                target_id=i,
                range_m=t.range_m + range_noise,
                bearing_deg=t.bearing_deg,
                velocity_ms=t.velocity_ms + vel_noise,
                snr_db=snr_db_val,
                detected=detected,
                cfar_threshold=threshold,
                rcs_used_m2=rcs,
                multipath_factor=mp_factor,
                antenna_gain_factor=ant_gain,
                range_noise_m=range_noise,
                velocity_noise_ms=vel_noise,
            ))

        return detections

    def max_detection_range(self, sinif: str) -> float:
        """Verilen hedef sınıfı için teorik maksimum menzil (m)."""
        rcs = RCS_TABLE.get(sinif, 0.1)
        return self._radar_eq.max_range_for_snr(rcs, self.params.min_snr_db)

    def summary(self) -> str:
        lines = [
            "PhysicsRadarSimulator",
            f"  Frekans    : {self.params.frequency_hz/1e9:.1f} GHz",
            f"  Dalga boyu : {self.params.wavelength_m*100:.1f} cm",
            f"  P_t        : {self.params.transmit_power_w:.1f} W",
            f"  G_tx/rx    : {self.params.tx_gain_db:.0f}/{self.params.rx_gain_db:.0f} dBi",
            f"  Bant gen.  : {self.params.bandwidth_hz/1e6:.0f} MHz",
            f"  Range res. : {self._range_resolution():.2f} m",
            f"  Gürültü g. : {self.params.noise_figure_db:.1f} dB",
            f"  PFA        : {self.params.pfa:.0e}",
            f"  CFAR alpha : {self._cfar._alpha:.2f}",
        ]
        for cls in ["Drone", "Balistik_Fuze", "Jet", "Helikopter"]:
            rmax = self.max_detection_range(cls) / 1000
            lines.append(f"  R_max({cls:12s}) = {rmax:.1f} km")
        return "\n".join(lines)
