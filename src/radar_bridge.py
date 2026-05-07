"""
RadarBridge — AERIS-10 PLFM Radar ↔ Çelik Kubbe köprü modülü.

PLFM_RADAR reposundaki radar_protocol.py üzerinden USB (FT2232H/FT601)
ile bağlanır, ham radar çerçevelerini alır ve Çelik Kubbe'nin Threat
veri modeline dönüştürülebilecek "tespit listesi" formatında yayınlar.

Çıktı sinyali (radar_tespit_sinyal):
    list[dict]:
        range_bin   : int    — 0..63 menzil bini
        doppler_bin : int    — 0..31 Doppler bini
        range_km    : float  — kilometre cinsinden mesafe
        velocity_ms : float  — m/s cinsinden radyal hız
        bearing_deg : float  — azimut (°)
        magnitude   : float  — sinyal gücü (lineer)
        snr_db      : float  — sinyal-gürültü oranı (dB)
"""

from __future__ import annotations

import math
import sys
import os
import time
import queue
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal, QTimer

# ── PLFM_RADAR radar_protocol.py yolunu ekle ────────────────────────────────
_PLFM_GUI_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                 "radar", "PLFM_RADAR", "9_Firmware", "9_3_GUI")
)
if _PLFM_GUI_DIR not in sys.path:
    sys.path.insert(0, _PLFM_GUI_DIR)

try:
    from radar_protocol import (
        RadarProtocol, RadarFrame, RadarAcquisition,
        FT2232HConnection, FT601Connection, DataRecorder,
        NUM_RANGE_BINS, NUM_DOPPLER_BINS,
    )
    PLFM_AVAILABLE = True
except ImportError:
    PLFM_AVAILABLE = False
    NUM_RANGE_BINS = 64
    NUM_DOPPLER_BINS = 32

logger = logging.getLogger("radar_bridge")


# ── Radar parametreleri ──────────────────────────────────────────────────────
@dataclass
class RadarParams:
    """AERIS-10 radar fiziksel parametreleri."""
    center_freq_hz: float = 10.5e9          # 10.5 GHz
    bandwidth_hz: float = 200e6             # 200 MHz chirp BW
    chirp_duration_us: float = 50.0         # 50 µs
    sampling_rate_hz: float = 400e6         # 400 MSPS ADC
    prf_hz: float = 5000.0                  # Darbe Tekrar Frekansı
    num_range_bins: int = NUM_RANGE_BINS
    num_doppler_bins: int = NUM_DOPPLER_BINS
    max_range_km: float = 3.0               # AERIS-10N varsayılan
    antenna_beamwidth_deg: float = 10.0     # 3 dB hüzme genişliği
    scan_rate_rpm: float = 15.0             # Step motor tarama hızı

    @property
    def range_resolution_m(self) -> float:
        """Menzil çözünürlüğü: c / (2 * BW)"""
        c = 3e8
        return c / (2 * self.bandwidth_hz)

    @property
    def max_unambiguous_range_m(self) -> float:
        """Belirsizlik olmayan maksimum menzil."""
        return self.max_range_km * 1000.0

    @property
    def range_bin_size_m(self) -> float:
        """Her menzil bininin metre karşılığı."""
        return self.max_unambiguous_range_m / self.num_range_bins

    @property
    def velocity_resolution_ms(self) -> float:
        """Hız çözünürlüğü: λ / (2 * N * T_PRI)"""
        c = 3e8
        lam = c / self.center_freq_hz
        t_pri = 1.0 / self.prf_hz
        return lam / (2 * self.num_doppler_bins * t_pri)

    @property
    def max_unambiguous_velocity_ms(self) -> float:
        """Belirsizlik olmayan maksimum hız."""
        c = 3e8
        lam = c / self.center_freq_hz
        return lam * self.prf_hz / 4.0


# ── CFAR tespit sonucu ───────────────────────────────────────────────────────
@dataclass
class RadarDetection:
    """Tek bir CFAR tespiti."""
    range_bin: int = 0
    doppler_bin: int = 0
    range_km: float = 0.0
    velocity_ms: float = 0.0
    bearing_deg: float = 0.0
    magnitude: float = 0.0
    snr_db: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "range_bin": self.range_bin,
            "doppler_bin": self.doppler_bin,
            "range_km": self.range_km,
            "velocity_ms": self.velocity_ms,
            "bearing_deg": self.bearing_deg,
            "magnitude": self.magnitude,
            "snr_db": self.snr_db,
            "timestamp": self.timestamp,
        }


# ── Radar Bridge Ana Sınıf ───────────────────────────────────────────────────
class RadarBridge(QThread):
    """
    AERIS-10 Radar → Çelik Kubbe köprüsü.

    Radar donanımına (veya mock moduna) bağlanır, çerçeve verisi toplar,
    CFAR tespitlerini çıkarır ve `radar_tespit_sinyal` üzerinden yayınlar.
    """

    # Sinyaller
    radar_tespit_sinyal = Signal(list)       # list[dict] — CFAR tespitleri
    radar_frame_sinyal = Signal(object)      # RadarFrame — ham çerçeve (opsiyonel)
    radar_durum_sinyal = Signal(str)          # Durum mesajı
    radar_baglanti_sinyal = Signal(bool)     # Bağlantı durumu

    def __init__(
        self,
        mock: bool = True,
        usb_type: str = "ft2232h",
        kayit_aktif: bool = False,
        params: Optional[RadarParams] = None,
    ):
        super().__init__()
        self._mock = mock
        self._usb_type = usb_type.lower()
        self._kayit_aktif = kayit_aktif
        self.params = params or RadarParams()

        self._dur = False
        self._bagli = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=4)
        self._connection = None
        self._acquisition: Optional[RadarAcquisition] = None
        self._recorder: Optional[DataRecorder] = None
        self._current_bearing: float = 0.0   # step motor açısı (°)

        # Tarama simülasyonu (step motor)
        self._scan_step_deg = 360.0 * self.params.scan_rate_rpm / (60.0 * 20.0)  # ~20 Hz güncelleme

    @property
    def bagli(self) -> bool:
        return self._bagli

    def run(self):
        """Ana çalışma döngüsü — radar bağlantısı ve veri toplama."""
        if not PLFM_AVAILABLE:
            logger.error("radar_protocol modülü bulunamadı — PLFM_RADAR yolu kontrol edin.")
            self.radar_durum_sinyal.emit("HATA: PLFM_RADAR modülü bulunamadı")
            return

        # USB bağlantısı oluştur — pyftdi backend yoksa veya cihaz reddederse
        # ValueError/UsbError/OSError fırlatır; thread'i sessiz öldürmemek için yakala.
        try:
            if self._usb_type == "ft601":
                self._connection = FT601Connection(mock=self._mock)
            else:
                self._connection = FT2232HConnection(mock=self._mock)

            opened = self._connection.open()
        except Exception as e:
            logger.error(f"Radar USB bağlantısı oluşturulamadı: {type(e).__name__}: {e}")
            self.radar_durum_sinyal.emit(f"HATA: USB bağlantısı: {type(e).__name__}")
            self.radar_baglanti_sinyal.emit(False)
            return

        if not opened:
            logger.error("Radar USB bağlantısı kurulamadı")
            self.radar_durum_sinyal.emit("HATA: USB bağlantısı kurulamadı")
            self.radar_baglanti_sinyal.emit(False)
            return

        self._bagli = True
        self.radar_baglanti_sinyal.emit(True)
        self.radar_durum_sinyal.emit(
            f"BAĞLI — {'Mock' if self._mock else 'Donanım'} | {self._usb_type.upper()}"
        )

        # HDF5 kayıt
        if self._kayit_aktif:
            self._recorder = DataRecorder()
            kayit_dosya = f"radar_kayit_{int(time.time())}.h5"
            self._recorder.start(kayit_dosya)

        # Veri toplama iş parçacığı başlat
        self._acquisition = RadarAcquisition(
            connection=self._connection,
            frame_queue=self._frame_queue,
            recorder=self._recorder,
        )
        self._acquisition.start()

        logger.info("RadarBridge aktif — veri toplama başladı")

        # Ana döngü: çerçeveleri oku ve tespitleri yayınla
        while not self._dur:
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Tarama açısını güncelle (step motor simülasyonu)
            self._current_bearing = (self._current_bearing + self._scan_step_deg) % 360.0

            # CFAR tespitlerini çıkar
            tespitler = self._extract_detections(frame)

            if tespitler:
                self.radar_tespit_sinyal.emit([t.to_dict() for t in tespitler])

            # Ham çerçeveyi de yayınla (GUI'de Range-Doppler haritası için)
            self.radar_frame_sinyal.emit(frame)

        # Temizlik
        if self._acquisition is not None:
            self._acquisition.stop()
            self._acquisition.join(timeout=2.0)
        if self._recorder is not None:
            self._recorder.stop()
        if self._connection is not None:
            self._connection.close()

        self._bagli = False
        self.radar_baglanti_sinyal.emit(False)
        self.radar_durum_sinyal.emit("BAĞLANTI KESİLDİ")
        logger.info("RadarBridge durduruldu")

    def durdur(self):
        """Radar köprüsünü güvenli şekilde durdur."""
        self._dur = True
        self.wait(5000)

    def _extract_detections(self, frame: RadarFrame) -> list[RadarDetection]:
        """
        RadarFrame'deki detection matrisinden tespit listesi oluştur.

        FPGA zaten CFAR çıkışını detection bitinde işaretlemiş durumda.
        Burada biz bunu fiziksel birimlere dönüştürüyoruz.
        """
        detections = []
        noise_floor = np.median(frame.magnitude[frame.magnitude > 0]) if np.any(frame.magnitude > 0) else 1.0
        if noise_floor <= 0:
            noise_floor = 1.0

        for rbin in range(self.params.num_range_bins):
            for dbin in range(self.params.num_doppler_bins):
                if frame.detections[rbin, dbin]:
                    mag = frame.magnitude[rbin, dbin]
                    snr = 10.0 * math.log10(max(mag / noise_floor, 1e-10))

                    # Menzil: bin → km
                    range_km = (rbin + 0.5) * self.params.range_bin_size_m / 1000.0

                    # Hız: Doppler bin → m/s
                    # Orta bin = sıfır hız, alt binler → yaklaşan, üst → uzaklaşan
                    dop_centered = dbin - self.params.num_doppler_bins // 2
                    velocity_ms = dop_centered * self.params.velocity_resolution_ms

                    det = RadarDetection(
                        range_bin=rbin,
                        doppler_bin=dbin,
                        range_km=range_km,
                        velocity_ms=velocity_ms,
                        bearing_deg=self._current_bearing,
                        magnitude=float(mag),
                        snr_db=snr,
                        timestamp=frame.timestamp,
                    )
                    detections.append(det)

        return detections

    def komut_gonder(self, opcode: int, value: int, addr: int = 0) -> bool:
        """FPGA'ya komut gönder (örn. CFAR eşiği, mod değişikliği)."""
        if self._connection is None or not self._connection.is_open:
            return False

        cmd = RadarProtocol.build_command(opcode, value, addr)
        return self._connection.write(cmd)
