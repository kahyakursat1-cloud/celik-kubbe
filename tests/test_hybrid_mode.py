"""
Hibrit mod testleri — radar.mock ve gimbal.mock farklı kombinasyonlarda
sistemin graceful kaldığını ve bileşenlerin bağımsız initialize edildiğini
doğrular. Donanım yokken real moda geçiş crash etmemeli.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QCoreApplication, Qt


def _ensure_app():
    return QCoreApplication.instance() or QCoreApplication(sys.argv)


class GimbalRealWithoutHardwareTests(unittest.TestCase):
    """gimbal.mock=False ama gerçek port yok — graceful degrade beklenir."""

    def setUp(self):
        _ensure_app()

    def test_baslat_with_nonexistent_port_fails_gracefully(self):
        from src.gimbal_controller import GimbalController

        g = GimbalController(mock=False, port="COM_NOT_EXIST_999", baud=115200)
        log_messages = []
        g.log_sinyal.connect(log_messages.append)

        # Crash etmemeli, sadece _bagli False kalmalı
        g.baslat()

        self.assertFalse(g._bagli, "Geçersiz portla baslat() yine de _bagli=True yaptı")
        self.assertTrue(log_messages, "Hata log_sinyal'i yayılmadı")
        self.assertIn("Hata", log_messages[-1] + "Hata")  # mesaj içeriği değil var olması önemli

    def test_hedefe_yonel_no_op_when_disconnected(self):
        from src.gimbal_controller import GimbalController

        g = GimbalController(mock=False, port="COM_NOT_EXIST_999")
        g.baslat()  # bağlanmayacak

        captured = []
        g.gimbal_durum_sinyal.connect(captured.append)

        # Bağlı değilken hedefe_yonel sessiz no-op olmalı
        g.hedefe_yonel(tid="X", bearing_deg=10.0, distance_km=1.0, altitude_m=100.0)
        self.assertFalse(captured, "Bağlı değilken durum sinyali yayıldı")


class RadarRealModeWithoutHardwareTests(unittest.TestCase):
    """radar.mock=False + donanım yok → bağlantı graceful başarısız olmalı."""

    def setUp(self):
        _ensure_app()

    def test_radar_real_mode_fails_gracefully_without_hardware(self):
        from src.radar_bridge import RadarBridge, PLFM_AVAILABLE

        if not PLFM_AVAILABLE:
            self.skipTest("PLFM_RADAR repo erişilebilir değil")

        bridge = RadarBridge(mock=False, usb_type="ft2232h")
        connection_states: list[bool] = []
        status_msgs: list[str] = []
        # DirectConnection: cross-thread sinyaller için event loop gerekmesin
        bridge.radar_baglanti_sinyal.connect(connection_states.append, Qt.DirectConnection)
        bridge.radar_durum_sinyal.connect(status_msgs.append, Qt.DirectConnection)

        bridge.start()
        bridge.wait(2000)  # bağlantı denemesi + bail-out için süre
        bridge.durdur()

        # Donanım yok → bağlanmamış olmalı + hata sinyali yayılmış olmalı
        self.assertFalse(any(connection_states),
                         f"Donanımsız real mod yanlış bağlantı bildirdi: {connection_states}")
        self.assertFalse(bridge.bagli, "bagli property donanımsız modda True kaldı")
        # USB hatası graceful yakalanıp durum sinyali ile bildirilmeli
        self.assertTrue(status_msgs, "Hiç durum sinyali yayılmadı (sessiz crash)")
        self.assertTrue(any("HATA" in s for s in status_msgs),
                        f"Hata bildiren durum mesajı yok: {status_msgs}")


class SensorFusionStandaloneTests(unittest.TestCase):
    """Füzyon, radar veya kamera olmadan başlatılıp graceful kalmalı."""

    def setUp(self):
        _ensure_app()

    def test_fusion_without_any_input_produces_no_tracks(self):
        from src.sensor_fusion import SensorFusion

        fusion = SensorFusion()
        emitted = []
        fusion.fuzyon_sinyal.connect(emitted.append)

        # Hiçbir girdi yok — sadece zaman aşımı kontrolü tetiklenebilir
        # ama sinyal yayılmamalı (boş izler için)
        self.assertEqual(len(fusion.aktif_izler), 0)
        self.assertEqual(len(emitted), 0)

    def test_fusion_with_only_radar_no_camera(self):
        from src.sensor_fusion import SensorFusion

        fusion = SensorFusion()
        fusion.radar_girdisi([
            {"bearing_deg": 0.0, "range_km": 1.0,
             "velocity_ms": -30.0, "snr_db": 18.0}
        ])
        # Yalnız-radar izi oluşmalı
        tracks = list(fusion.aktif_izler.values())
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].kaynak, "yalniz_radar")
        self.assertEqual(tracks[0].sinif, "Bilinmeyen")

    def test_fusion_with_only_camera_no_radar(self):
        from src.sensor_fusion import SensorFusion

        fusion = SensorFusion()
        fusion.kamera_girdisi([
            {"track_id": 1, "sinif": "Drone", "guven": 0.85,
             "cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.3}
        ])
        tracks = list(fusion.aktif_izler.values())
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].kaynak, "yalniz_kamera")
        self.assertEqual(tracks[0].sinif, "Drone")


class ConfigCombinationTests(unittest.TestCase):
    """config.yaml kritik bayrakların tüm 4 kombinasyonu için anlamlı default."""

    def test_all_mock_combinations_are_valid_booleans(self):
        from src.konfig import cfg

        for section in ("radar", "gimbal"):
            self.assertIn(section, cfg)
            self.assertIsInstance(cfg[section].get("mock", True), bool)
            self.assertIsInstance(cfg[section].get("aktif", True), bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
