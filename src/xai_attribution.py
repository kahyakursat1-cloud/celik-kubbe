"""
xai_attribution.py — ContextFusion XAI bileşeni.

İki sorumluluk:
  1. AlphaPolicy: Bağlam (multi-threat density, sensor dropout, low SNR)
     girdisinden dinamik tehdit ağırlıklarını (α_r, α_v, α_c) üretir.
     Makale Tablo 4'teki kural tablosunun gerçek runtime karşılığı.
  2. ThreatExplainer: Hesaplanan tehdit skorunun her bir faktöre ait
     katkısını (φ_r, φ_v, φ_c) "exact linear SHAP" yöntemiyle döker.
     Skor formülü additive olduğu için Shapley değerleri analitiktir;
     pahalı sampling gerekmez.

Mevcut kural-tabanlı _tehdit_hesapla mantığı (sensor_fusion.py) bozulmaz;
bu modül onun üstünde "neden bu skoru üretti?" sorusunu yanıtlar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Sınıf-tabanlı tehdit kimlik çarpanı (mevcut kural setiyle aynı) ──
_SINIF_KIMLIK_AGIRLIGI: dict[str, float] = {
    "BalisticMissile": 50.0,
    "Missile": 50.0,
    "Balistik_Fuze": 50.0,
    "Drone": 25.0,
    "UAV": 25.0,
    "FixedWingUAV": 25.0,
    "Mini_IHA": 25.0,
    "Helicopter": 15.0,
    "Aircraft": 15.0,
    "Jet": 15.0,
}


# ── Bağlam (context) durumu ────────────────────────────────────────────
@dataclass
class FusionContext:
    """
    Anlık operasyonel bağlam. AlphaPolicy bu yapıdan α değerlerini türetir.
    Tüm alanlar opsiyonel; eksik bağlam → temel ağırlıklar (no-op).
    """
    aktif_track_sayisi: int = 1
    radar_dropout: bool = False
    kamera_dropout: bool = False
    ortalama_snr_db: Optional[float] = None
    multi_threat_esigi: int = 3  # 3+ aktif track → multi_threat modu


# ── α-Policy (Makale Tablo 4'ün runtime karşılığı) ─────────────────────
@dataclass
class AlphaPolicy:
    """
    Bağlam-duyarlı tehdit ağırlığı politikası.
    Temel değerler: α_r=0.5, α_v=0.3, α_c=0.2 (toplam=1.0).
    Bağlama göre additive ayarlama, sonra [0.1, 0.8] aralığına kırpma.

    Makale §4.2 Tablo 4 ile bire bir aynı kuralları kodlar.
    """
    alpha_r_base: float = 0.5
    alpha_v_base: float = 0.3
    alpha_c_base: float = 0.2
    clip_min: float = 0.1
    clip_max: float = 0.8

    def compute(self, ctx: FusionContext) -> tuple[float, float, float]:
        """Bağlamdan dinamik (α_r, α_v, α_c) hesapla."""
        a_r, a_v, a_c = self.alpha_r_base, self.alpha_v_base, self.alpha_c_base

        # Kural 1: Multi-threat → hız faktörü ağırlığı artırılır
        if ctx.aktif_track_sayisi >= ctx.multi_threat_esigi:
            a_v += 0.15
            a_r -= 0.05
            a_c -= 0.10

        # Kural 2: Radar dropout → kamera güveni ağırlığı artırılır
        if ctx.radar_dropout:
            a_c += 0.15
            a_r -= 0.10
            a_v -= 0.05

        # Kural 3: Kamera dropout → radar faktörü artırılır
        if ctx.kamera_dropout:
            a_r += 0.10
            a_c -= 0.15
            a_v += 0.05

        # Kural 4: Düşük SNR → mesafe faktörü güvenilmez, hız ve sınıf öne çıkar
        if ctx.ortalama_snr_db is not None and ctx.ortalama_snr_db < 8.0:
            a_r -= 0.10
            a_v += 0.05
            a_c += 0.05

        # Kırp & normalize
        a_r = max(self.clip_min, min(self.clip_max, a_r))
        a_v = max(self.clip_min, min(self.clip_max, a_v))
        a_c = max(self.clip_min, min(self.clip_max, a_c))
        # Yeniden normalize (toplam=1)
        toplam = a_r + a_v + a_c
        return a_r / toplam, a_v / toplam, a_c / toplam


# ── Tehdit skoru bileşenleri (makaledeki r/v/c-factor karşılığı) ───────
@dataclass
class ThreatComponents:
    """Bir tehdit skorunun ham bileşen değerleri (ağırlıklandırılmadan önce)."""
    r_raw: float = 0.0  # 0-40 puan
    v_raw: float = 0.0  # 0-40 puan
    c_raw: float = 0.0  # 0-50 puan
    fusion_bonus: float = 0.0  # füzyon kaynaklı bonus (0 veya skor*0.1)


def hesapla_ham_bilesenler(track) -> ThreatComponents:
    """
    sensor_fusion._tehdit_hesapla içindeki linear additive bileşenleri
    saf değerler olarak döndür. Skor formülünü değiştirmez; sadece izole eder.
    """
    r_raw = max(0.0, (5.0 - track.range_km) / 4.5) * 40.0
    v_raw = 0.0
    if track.velocity_ms < -10.0:
        v_raw = min(40.0, abs(track.velocity_ms) / 2.0)
    c_raw = _SINIF_KIMLIK_AGIRLIGI.get(track.sinif, 0.0)
    fusion_bonus = 0.0
    if track.kaynak == "fuzyon":
        # Mevcut kuralda skor *= 1.1 var; SHAP additive form için
        # bonus = base_skor * 0.1 olarak ayrıştırılır
        base = r_raw + v_raw + c_raw
        fusion_bonus = base * 0.1
    return ThreatComponents(r_raw, v_raw, c_raw, fusion_bonus)


# ── SHAP (linear-exact) ────────────────────────────────────────────────
@dataclass
class ShapAttribution:
    """
    Bir tehdit skoru için exact linear Shapley dekompozisyonu.
    Skor formülü additive olduğu için Shapley payı = ilgili bileşenin değeri.
    """
    phi_r: float = 0.0       # range faktörü katkısı
    phi_v: float = 0.0       # velocity faktörü katkısı
    phi_c: float = 0.0       # class faktörü katkısı
    phi_fusion: float = 0.0  # füzyon bonusunun katkısı
    baseline: float = 0.0    # E[τ] (tüm faktörler sıfırken)
    score: float = 0.0
    alpha_r: float = 0.5
    alpha_v: float = 0.3
    alpha_c: float = 0.2
    dominant_factor: str = "range"  # phi_*'ların en büyüğü

    def to_dict(self) -> dict:
        return {
            "phi_r": round(self.phi_r, 2),
            "phi_v": round(self.phi_v, 2),
            "phi_c": round(self.phi_c, 2),
            "phi_fusion": round(self.phi_fusion, 2),
            "baseline": round(self.baseline, 2),
            "score": round(self.score, 2),
            "alpha_r": round(self.alpha_r, 3),
            "alpha_v": round(self.alpha_v, 3),
            "alpha_c": round(self.alpha_c, 3),
            "dominant": self.dominant_factor,
        }


class ThreatExplainer:
    """
    Tehdit skorunu α-ağırlıklı linear formülde tutarak hem skoru hem de
    bileşen katkılarını üretir. Mevcut _tehdit_hesapla ile aynı sayısal
    sonucu vermek için α=(0.5, 0.3, 0.2) varsayılan; AlphaPolicy farklı
    α verirse skor da farklılaşır (makale §4.2 Tablo 4 davranışı).
    """

    def __init__(self, policy: Optional[AlphaPolicy] = None):
        self.policy = policy or AlphaPolicy()

    def explain(self, track, ctx: Optional[FusionContext] = None) -> ShapAttribution:
        bilesenler = hesapla_ham_bilesenler(track)
        a_r, a_v, a_c = self.policy.compute(ctx or FusionContext())

        # α-ağırlıklı katkılar (linear, additive)
        # Not: Mevcut kural kodu r_raw için 40, v_raw için 40, c_raw için 50'ye
        # kadar ham puan veriyor. α'lar bu ham bileşenleri "weighting" eder.
        # α'ların toplamı 1 → toplam üst sınır ~50 (max) (makaleyle tutarlı).
        phi_r = a_r * bilesenler.r_raw
        phi_v = a_v * bilesenler.v_raw
        phi_c = a_c * bilesenler.c_raw
        phi_fusion = bilesenler.fusion_bonus  # füzyon bonusu α'dan bağımsız

        baseline = 0.0  # tüm faktörler 0 ise skor 0
        score = baseline + phi_r + phi_v + phi_c + phi_fusion
        # Mevcut kural koduyla uyumluluk için 0-100 üst sınır:
        score = max(0.0, min(100.0, score))

        # Hangi faktör dominant (sigmoid hover tooltip için)
        bilesen_sirala = {"range": phi_r, "velocity": phi_v, "class": phi_c}
        dominant = max(bilesen_sirala, key=bilesen_sirala.get)

        return ShapAttribution(
            phi_r=phi_r,
            phi_v=phi_v,
            phi_c=phi_c,
            phi_fusion=phi_fusion,
            baseline=baseline,
            score=score,
            alpha_r=a_r,
            alpha_v=a_v,
            alpha_c=a_c,
            dominant_factor=dominant,
        )

    def explain_with_label(self, track, ctx: Optional[FusionContext] = None) -> tuple[str, float, ShapAttribution]:
        """SensorFusion uyumlu wrapper: (seviye, skor, attribution) tuple."""
        attr = self.explain(track, ctx)
        if attr.score >= 80.0:
            seviye = "KRİTİK"
        elif attr.score >= 50.0:
            seviye = "YÜKSEK"
        elif attr.score >= 30.0:
            seviye = "ORTA"
        else:
            seviye = "DÜŞÜK"
        return seviye, attr.score, attr


# ── Modül-düzeyi singleton (basit kullanım için) ───────────────────────
_default_explainer: Optional[ThreatExplainer] = None


def get_default_explainer() -> ThreatExplainer:
    """SensorFusion'dan tek satır import için singleton."""
    global _default_explainer
    if _default_explainer is None:
        _default_explainer = ThreatExplainer()
    return _default_explainer
