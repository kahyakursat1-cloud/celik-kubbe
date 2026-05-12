"""
SensorFusion — Radar + Kamera (YOLOv11) veri birleştirme modülü.

İki sensör kaynağından gelen tespit verilerini birleştirir:
  • Radar  → mesafe (km), hız (m/s), azimut (°), SNR
  • Kamera → sınıf (Drone, Füze vb.), güven skoru, görsel takip ID

Eşleştirme stratejisi:
  1. Açısal yakınlık (bearing vs kamera cx → bearing)
  2. Mesafe tutarlılığı (radar range vs kamera bbox büyüklüğü)
  3. Zamansal korelasyon (timestamp farkı < eşik)

Çıktı: Birleştirilmiş "FusedTrack" nesneleri.
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

from PySide6.QtCore import QObject, Signal

from src.kalman_filter import KalmanFilter2D
from src.xai_attribution import (
    AlphaPolicy,
    FusionContext,
    ThreatExplainer,
    ShapAttribution,
)

logger = logging.getLogger("sensor_fusion")


# ── Birleştirilmiş İz (Fused Track) ─────────────────────────────────────────
@dataclass
class FusedTrack:
    """
    Radar + Kamera verilerinin birleştirildiği tek bir hedef izi.
    İçerisinde durum tahmini (State Estimation) için Kalman Filtresi barındırır.
    """
    track_id: int = 0
    # Konum (Filtrelenmiş Polar)
    range_km: float = 0.0
    velocity_ms: float = 0.0
    bearing_deg: float = 0.0
    altitude_m: float = 0.0
    # Sınıflandırma (kamera kaynaklı)
    sinif: str = "Bilinmeyen"
    guven: float = 0.0
    # Füzyon meta verileri
    kaynak: str = "yalniz_radar"
    radar_snr_db: float = 0.0
    son_guncelleme: float = 0.0
    yasam_suresi: float = 0.0
    olusturma_zamani: float = 0.0
    # Tehdit seviyesi (XAI Tabanlı Skorlama)
    tehdit_seviyesi: str = "DÜŞÜK"
    tehdit_skoru: float = 0.0
    # SHAP attribution (factor contributions to threat score)
    shap_attribution: Optional["ShapAttribution"] = None
    # VLM scene analysis (optional 4th evidence channel)
    vlm_anomaly_score: float = 0.5  # neutral prior
    vlm_brief: str = ""
    # Kalman Filtresi
    kalman: object = field(default=None)

    def to_dict(self) -> dict:
        d = {
            "track_id": self.track_id,
            "range_km": self.range_km,
            "velocity_ms": self.velocity_ms,
            "bearing_deg": self.bearing_deg,
            "altitude_m": self.altitude_m,
            "sinif": self.sinif,
            "guven": self.guven,
            "kaynak": self.kaynak,
            "radar_snr_db": self.radar_snr_db,
            "tehdit_seviyesi": self.tehdit_seviyesi,
            "tehdit_skoru": round(self.tehdit_skoru, 2),
            "son_guncelleme": self.son_guncelleme,
        }
        if self.shap_attribution is not None:
            d["shap"] = self.shap_attribution.to_dict()
        if self.vlm_brief:
            d["vlm_brief"] = self.vlm_brief
            d["vlm_anomaly_score"] = round(self.vlm_anomaly_score, 3)
        return d

# ── XAI Tabanlı Tehdit Algoritması (Fuzzy/Ağırlıklı Skor) ───────────────────
def _tehdit_hesapla(track: FusedTrack) -> tuple[str, float]:
    """
    Hedefin kinetik (hız, mesafe) ve kimlik (sınıf) verilerine dayanarak
    Açıklanabilir (Explainable) bir Tehdit Skoru (0-100) hesaplar.
    """
    skor = 0.0
    
    # 1. Kinetik Çarpan (Mesafe ve Hız)
    # Mesafe azaldıkça tehdit üstel olarak artar. (0.5 km = max, 5 km = min)
    r_factor = max(0, (5.0 - track.range_km) / 4.5) * 40.0
    
    # Hız (Bize yaklaşan hız negatiftir)
    v_factor = 0.0
    if track.velocity_ms < -10.0:
        v_factor = min(40.0, abs(track.velocity_ms) / 2.0)  # max 40 puan

    skor += r_factor + v_factor

    # 2. Sınıflandırma Çarpanı (Kimlik)
    sinif = track.sinif
    if sinif in {"BalisticMissile", "Missile", "Balistik_Fuze"}:
        skor += 50.0
    elif sinif in {"Drone", "UAV", "FixedWingUAV", "Mini_IHA"}:
        skor += 25.0
    elif sinif in {"Helicopter", "Aircraft", "Jet"}:
        skor += 15.0

    # 3. Güvenilirlik Çarpanı (Sensör Füzyon Kalitesi)
    if track.kaynak == "fuzyon":
        skor *= 1.1  # Her iki sensör onaylıyorsa tehdit kesindir

    skor = min(100.0, skor)

    # Etiketleme
    if skor >= 80.0:
        seviye = "KRİTİK"
    elif skor >= 50.0:
        seviye = "YÜKSEK"
    elif skor >= 30.0:
        seviye = "ORTA"
    else:
        seviye = "DÜŞÜK"

    return seviye, skor


# ── Ana Füzyon Sınıfı ────────────────────────────────────────────────────────
class SensorFusion(QObject):
    """
    Radar ve kamera tespitlerini birleştiren ana füzyon motoru.

    Kullanım:
        fusion = SensorFusion()
        fusion.fuzyon_sinyal.connect(gui_callback)
        radar_bridge.radar_tespit_sinyal.connect(fusion.radar_girdisi)
        yolo_pipeline.tespit_sinyal.connect(fusion.kamera_girdisi)
    """

    # Birleştirilmiş tespit sinyali
    fuzyon_sinyal = Signal(list)       # list[dict] — FusedTrack.to_dict()
    log_sinyal = Signal(str)

    # Füzyon parametreleri
    ACI_ESIGI_DEG: float = 15.0          # Açısal eşleşme eşiği (°)
    MESAFE_ESIGI_KM: float = 1.0         # Mesafe eşleşme eşiği (km)
    ZAMAN_ESIGI_S: float = 2.0           # Zamansal eşleşme penceresi (s)
    IZ_ZAMAN_ASIMI_S: float = 5.0        # İz düşme süresi (s)
    KAMERA_ACI_GENISLIGI_DEG: float = 120.0  # Kamera görüş açısı

    def __init__(self, parent=None, alpha_policy: Optional[AlphaPolicy] = None,
                 vlm_lambda: float = 0.3):
        super().__init__(parent)
        self._radar_tespitler: list[dict] = []
        self._kamera_tespitler: list[dict] = []
        self._aktif_izler: dict[int, FusedTrack] = {}
        self._sonraki_id: int = 1
        self._son_radar_zaman: float = 0.0
        self._son_kamera_zaman: float = 0.0
        # XAI: dinamik α policy + SHAP explainer (makale Tablo 4 runtime karşılığı)
        self._explainer = ThreatExplainer(alpha_policy or AlphaPolicy())
        # VLM evidence channel ağırlığı (MAP fusion'da log-likelihood ratio katsayısı)
        self._vlm_lambda = vlm_lambda

    @property
    def aktif_izler(self) -> dict[int, FusedTrack]:
        return self._aktif_izler

    def _current_context(self) -> FusionContext:
        """
        Anlık operasyonel bağlamı türet — α policy bunu kullanır.
        Multi-threat density, sensor dropout, average SNR.
        """
        simdi = time.time()
        # Sensor dropout: son güncellemeden >2s geçtiyse o sensör 'dropped' say
        radar_dropout = (simdi - self._son_radar_zaman) > 2.0 if self._son_radar_zaman else False
        kamera_dropout = (simdi - self._son_kamera_zaman) > 2.0 if self._son_kamera_zaman else False
        # Ortalama SNR (aktif izlerden)
        snr_list = [t.radar_snr_db for t in self._aktif_izler.values() if t.radar_snr_db > 0]
        ortalama_snr = sum(snr_list) / len(snr_list) if snr_list else None
        return FusionContext(
            aktif_track_sayisi=len(self._aktif_izler),
            radar_dropout=radar_dropout,
            kamera_dropout=kamera_dropout,
            ortalama_snr_db=ortalama_snr,
        )

    def _skorla_ve_acikla(self, track: FusedTrack) -> tuple[str, float, ShapAttribution]:
        """
        Mevcut _tehdit_hesapla'nın yerine geçen, dinamik α + SHAP attribution
        üreten yeni hesaplama. VLM anomaly_score 4. evidence kanalı olarak
        log-linear MAP fusion mantığıyla eklenir.

        VLM katkısı: e_vlm = 0.5 neutral, λ=0 davranışı identical to baseline.
        """
        ctx = self._current_context()
        seviye, skor, attr = self._explainer.explain_with_label(track, ctx)

        # ── VLM 4. evidence kanalı (log-likelihood ratio additive contribution) ──
        # e_vlm ∈ [0,1]; 0.5 = neutral, >0.5 → tehdit lehine, <0.5 → tehdit aleyhine
        # log[P(e|T=1)/P(e|T=0)] ≈ logit(e_vlm) (uniform Beta priors için)
        if self._vlm_lambda > 0 and track.vlm_anomaly_score != 0.5:
            e = max(0.01, min(0.99, track.vlm_anomaly_score))
            import math as _math
            log_lr = _math.log(e / (1.0 - e))  # logit
            vlm_bonus = self._vlm_lambda * log_lr * 5.0  # ölçek: 5 puan/unit logit
            skor = max(0.0, min(100.0, skor + vlm_bonus))
            # Etiketi tekrar belirle
            if skor >= 80.0:
                seviye = "KRİTİK"
            elif skor >= 50.0:
                seviye = "YÜKSEK"
            elif skor >= 30.0:
                seviye = "ORTA"
            else:
                seviye = "DÜŞÜK"

        return seviye, skor, attr

    def radar_girdisi(self, tespitler: list):
        """Radar tespitlerini al ve füzyonu tetikle."""
        self._radar_tespitler = tespitler
        self._son_radar_zaman = time.time()
        self._fuzyonu_calistir()

    def kamera_girdisi(self, tespitler: list):
        """Kamera (YOLO) tespitlerini al ve füzyonu tetikle."""
        self._kamera_tespitler = tespitler
        self._son_kamera_zaman = time.time()
        self._fuzyonu_calistir()

    def _fuzyonu_calistir(self):
        """
        Ana füzyon algoritması:
          1. Radar tespitlerini kontrol et
          2. Kamera tespitlerini kontrol et
          3. Eşleşenleri birleştir (radar konum + kamera sınıf)
          4. Eşleşmeyenleri tek-sensör izi olarak ekle
          5. Süresi dolan izleri kaldır
          6. Sonuçları yayınla
        """
        simdi = time.time()
        eslesmis_radar = set()
        eslesmis_kamera = set()

        # ── Adım 0: Tüm aktif izler için Kalman Predict ──
        for track in self._aktif_izler.values():
            if track.kalman:
                track.kalman.predict()

        # ── Adım 1: Radar-Kamera eşleştirmesi ──
        for ri, r_det in enumerate(self._radar_tespitler):
            r_bearing = r_det.get("bearing_deg", 0.0)
            r_range = r_det.get("range_km", 0.0)

            en_iyi_skor = float("inf")
            en_iyi_ki = -1

            for ki, k_det in enumerate(self._kamera_tespitler):
                if ki in eslesmis_kamera:
                    continue

                # Kamera cx → bearing dönüşümü
                k_bearing = (k_det.get("cx", 0.5) - 0.5) * self.KAMERA_ACI_GENISLIGI_DEG

                # Açısal fark
                aci_fark = abs(r_bearing - k_bearing)
                if aci_fark > 180:
                    aci_fark = 360 - aci_fark

                if aci_fark > self.ACI_ESIGI_DEG:
                    continue

                # Mesafe tutarlılık skoru (kamera bbox büyüklüğü ↔ radar mesafe)
                k_diag = math.sqrt(k_det.get("w", 0) ** 2 + k_det.get("h", 0) ** 2)
                k_tahmin_range = max(0.1, (1.0 - k_diag * 2) * 5.0)  # kaba tahmin
                mesafe_fark = abs(r_range - k_tahmin_range)

                # Toplam skor (düşük = daha iyi eşleşme)
                skor = aci_fark / self.ACI_ESIGI_DEG + mesafe_fark / self.MESAFE_ESIGI_KM
                if skor < en_iyi_skor:
                    en_iyi_skor = skor
                    en_iyi_ki = ki

            if en_iyi_ki >= 0 and en_iyi_skor < 2.0:
                # Eşleşme bulundu → füzyon izi oluştur/güncelle
                k_det = self._kamera_tespitler[en_iyi_ki]
                eslesmis_radar.add(ri)
                eslesmis_kamera.add(en_iyi_ki)

                track = self._iz_bul_veya_olustur(r_bearing, r_range)
                
                # Kalman Güncellemesi
                if track.kalman:
                    x = r_range * math.cos(math.radians(r_bearing))
                    y = r_range * math.sin(math.radians(r_bearing))
                    track.kalman.update([x, y])
                    fx, fy, fvx, fvy = track.kalman.get_state()
                    track.range_km = math.hypot(fx, fy)
                    track.bearing_deg = math.degrees(math.atan2(fy, fx))
                else:
                    track.range_km = r_range
                    track.bearing_deg = r_bearing
                
                track.velocity_ms = r_det.get("velocity_ms", 0.0)
                track.radar_snr_db = r_det.get("snr_db", 0.0)
                track.sinif = k_det.get("sinif", "Bilinmeyen")
                track.guven = k_det.get("guven", 0.0)
                track.kaynak = "fuzyon"
                track.son_guncelleme = simdi
                track.tehdit_seviyesi, track.tehdit_skoru, track.shap_attribution = self._skorla_ve_acikla(track)

        # ── Adım 2: Eşleşmeyen radar tespitleri → yalnız-radar izi ──
        for ri, r_det in enumerate(self._radar_tespitler):
            if ri in eslesmis_radar:
                continue
            r_bearing = r_det.get("bearing_deg", 0.0)
            r_range = r_det.get("range_km", 0.0)

            track = self._iz_bul_veya_olustur(r_bearing, r_range)
            
            if track.kalman:
                x = r_range * math.cos(math.radians(r_bearing))
                y = r_range * math.sin(math.radians(r_bearing))
                track.kalman.update([x, y])
                fx, fy, fvx, fvy = track.kalman.get_state()
                track.range_km = math.hypot(fx, fy)
                track.bearing_deg = math.degrees(math.atan2(fy, fx))
            else:
                track.range_km = r_range
                track.bearing_deg = r_bearing

            track.velocity_ms = r_det.get("velocity_ms", 0.0)
            track.radar_snr_db = r_det.get("snr_db", 0.0)
            if track.kaynak != "fuzyon":
                track.kaynak = "yalniz_radar"
            track.son_guncelleme = simdi
            track.tehdit_seviyesi, track.tehdit_skoru, track.shap_attribution = self._skorla_ve_acikla(track)

        # ── Adım 3: Eşleşmeyen kamera tespitleri → yalnız-kamera izi ──
        for ki, k_det in enumerate(self._kamera_tespitler):
            if ki in eslesmis_kamera:
                continue
            k_bearing = (k_det.get("cx", 0.5) - 0.5) * self.KAMERA_ACI_GENISLIGI_DEG
            k_diag = math.sqrt(k_det.get("w", 0) ** 2 + k_det.get("h", 0) ** 2)
            k_range = max(0.1, (1.0 - k_diag * 2) * 5.0)

            track = self._iz_bul_veya_olustur(k_bearing, k_range)
            
            if track.kalman:
                x = k_range * math.cos(math.radians(k_bearing))
                y = k_range * math.sin(math.radians(k_bearing))
                track.kalman.update([x, y])
                fx, fy, fvx, fvy = track.kalman.get_state()
                track.range_km = math.hypot(fx, fy)
                track.bearing_deg = math.degrees(math.atan2(fy, fx))
            else:
                track.range_km = k_range
                track.bearing_deg = k_bearing

            track.sinif = k_det.get("sinif", "Bilinmeyen")
            track.guven = k_det.get("guven", 0.0)
            if track.kaynak != "fuzyon":
                track.kaynak = "yalniz_kamera"
            track.son_guncelleme = simdi
            track.tehdit_seviyesi, track.tehdit_skoru, track.shap_attribution = self._skorla_ve_acikla(track)

        # ── Adım 4: Süresi dolan izleri kaldır ──
        suresi_dolen = [
            tid for tid, t in self._aktif_izler.items()
            if (simdi - t.son_guncelleme) > self.IZ_ZAMAN_ASIMI_S
        ]
        for tid in suresi_dolen:
            del self._aktif_izler[tid]

        # ── Adım 5: Yaşam süresini güncelle, tahminleri uygula ve yayınla ──
        sonuclar = []
        for track in self._aktif_izler.values():
            # Eğer bu döngüde bir ölçümle güncellenmediyse, Kalman tahminini (predict) konuma yansıt
            # (Coasting / Blind Tracking)
            if track.son_guncelleme < simdi and track.kalman:
                fx, fy, fvx, fvy = track.kalman.get_state()
                track.range_km = math.hypot(fx, fy)
                track.bearing_deg = math.degrees(math.atan2(fy, fx))
                # Tehdit seviyesini tahmin üzerinden yeniden hesapla (mesafe değişmiş olabilir)
                track.tehdit_seviyesi, track.tehdit_skoru, track.shap_attribution = self._skorla_ve_acikla(track)

            track.yasam_suresi = simdi - track.olusturma_zamani
            sonuclar.append(track.to_dict())

        if sonuclar:
            self.fuzyon_sinyal.emit(sonuclar)

    def _iz_bul_veya_olustur(self, bearing: float, range_km: float) -> FusedTrack:
        """
        Mevcut izler arasında açısal ve mesafe yakınlığına göre eşleşme ara.
        Bulamazsa yeni iz oluştur.
        """
        en_yakin_id = -1
        en_yakin_skor = float("inf")

        for tid, track in self._aktif_izler.items():
            aci_fark = abs(track.bearing_deg - bearing)
            if aci_fark > 180:
                aci_fark = 360 - aci_fark
            mesafe_fark = abs(track.range_km - range_km)
            skor = aci_fark + mesafe_fark * 10.0

            if skor < en_yakin_skor and skor < 30.0:
                en_yakin_skor = skor
                en_yakin_id = tid

        if en_yakin_id >= 0:
            return self._aktif_izler[en_yakin_id]

        # Yeni iz oluştur
        kf = KalmanFilter2D(dt=0.5)
        # Başlangıç durumu
        x = range_km * math.cos(math.radians(bearing))
        y = range_km * math.sin(math.radians(bearing))
        kf.x[0,0] = x
        kf.x[1,0] = y
        
        yeni = FusedTrack(
            track_id=self._sonraki_id,
            olusturma_zamani=time.time(),
            kalman=kf
        )
        self._aktif_izler[self._sonraki_id] = yeni
        self._sonraki_id += 1
        return yeni

    def temizle(self):
        """Tüm izleri sıfırla."""
        self._aktif_izler.clear()
        self._radar_tespitler.clear()
        self._kamera_tespitler.clear()
