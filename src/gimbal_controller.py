"""
GimbalController — Radar ve Füzyon tespitlerine göre kamerayı (Gimbal) yönlendirir.

MAVLink protokolü kullanılarak (veya özel seri protokol) hedef konumuna (pan/tilt)
otomatik odaklanmayı ve takibi (tracking) sağlar.
"""

from __future__ import annotations

import math
import time
import logging
from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger("gimbal_controller")

class GimbalController(QObject):
    """
    Gimbal (Kamera Yönlendirme) modülü.
    
    Belirlenen hedef koordinatlarına göre Pan (Azimut) ve Tilt (İrtifa) açılarını hesaplar,
    ilgili donanıma (seri port veya MAVLink) komut gönderir.
    """
    gimbal_durum_sinyal = Signal(dict)   # {"pan": float, "tilt": float, "hedef_id": str}
    log_sinyal = Signal(str)

    def __init__(self, mock: bool = True, port: str = "COM3", baud: int = 115200):
        super().__init__()
        self._mock = mock
        self._port = port
        self._baud = baud
        
        self._hedef_tid: str | None = None
        self._hedef_pan: float = 0.0
        self._hedef_tilt: float = 0.0
        
        self._aktif_pan: float = 0.0
        self._aktif_tilt: float = 0.0
        
        self._bagli = False
        self._mavlink_conn = None

        # PID parametreleri (Gelecekte daha pürüzsüz takip için eklenebilir)
        self._max_hiz_deg_s = 60.0

    def baslat(self):
        """Gimbal donanımı ile bağlantı kurar."""
        if self._mock:
            self._bagli = True
            logger.info("Gimbal MOCK modunda başlatıldı.")
            self.log_sinyal.emit("Gimbal MOCK bağlantısı aktif.")
            return

        try:
            from pymavlink import mavutil
            self._mavlink_conn = mavutil.mavlink_connection(self._port, baud=self._baud)
            self._bagli = True
            logger.info(f"Gimbal bağlantısı başarılı: {self._port} @ {self._baud}")
            self.log_sinyal.emit("Gimbal MAVLink bağlantısı başarılı.")
        except Exception as e:
            self._bagli = False
            logger.error(f"Gimbal bağlantı hatası: {e}")
            self.log_sinyal.emit(f"Gimbal Bağlantı Hatası: {e}")

    def durdur(self):
        """Gimbal bağlantısını kapatır."""
        self._bagli = False
        if self._mavlink_conn:
            self._mavlink_conn.close()
        logger.info("Gimbal durduruldu.")

    def hedefe_yonel(self, tid: str, bearing_deg: float, distance_km: float, altitude_m: float):
        """
        Belirtilen hedefe yönelmek için gerekli Pan ve Tilt açılarını hesaplar ve komut gönderir.
        """
        if not self._bagli:
            return

        self._hedef_tid = tid
        self._hedef_pan = bearing_deg

        # Tilt hesaplama (Düze göre)
        # distance_m = distance_km * 1000
        # tilt_rad = atan2(altitude_m, distance_m)
        dist_m = max(1.0, distance_km * 1000.0)
        tilt_rad = math.atan2(altitude_m, dist_m)
        self._hedef_tilt = math.degrees(tilt_rad)

        self._komut_gonder(self._hedef_pan, self._hedef_tilt)

    def serbest_mod(self):
        """Takibi bırakıp başlangıç pozisyonuna (0, 0) döner."""
        self._hedef_tid = None
        self._komut_gonder(0.0, 0.0)
        logger.info("Gimbal serbest moda geçti.")

    def _komut_gonder(self, pan: float, tilt: float):
        """Hesaplanan açıları donanıma gönderir."""
        # Pan açısını -180 ile 180 arasına normalize et
        pan = (pan + 180) % 360 - 180
        
        # Tilt sınırları
        tilt = max(-30.0, min(90.0, tilt))

        self._aktif_pan = pan
        self._aktif_tilt = tilt

        if self._mock:
            # Mock modda doğrudan ulaştığını varsay
            self.gimbal_durum_sinyal.emit({
                "pan": round(pan, 2),
                "tilt": round(tilt, 2),
                "hedef_id": self._hedef_tid or "YOK"
            })
            return

        if self._mavlink_conn:
            from pymavlink import mavutil
            # MAV_CMD_DO_MOUNT_CONTROL: pitch, roll, yaw
            self._mavlink_conn.mav.command_long_send(
                self._mavlink_conn.target_system,
                self._mavlink_conn.target_component,
                mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL,
                1, # confirmation
                tilt * 100.0,  # pitch (santiderece veya MAVLink versiyonuna göre derece)
                0.0,           # roll
                pan * 100.0,   # yaw
                0, 0, 0,
                mavutil.mavlink.MAV_MOUNT_MODE_MAVLINK_TARGETING
            )
            self.gimbal_durum_sinyal.emit({
                "pan": round(pan, 2),
                "tilt": round(tilt, 2),
                "hedef_id": self._hedef_tid or "YOK"
            })
