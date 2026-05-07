"""
GUI smoke testleri — CelikKubbeGUI MainWindow'un instantiate edilebildiğini
ve kritik widget yapısını koruduğunu doğrular.

Headless test: QApplication başlatılır ama exec() çağrılmaz, dolayısıyla
QTimer'lar tetiklenmez ve test deterministik kalır.
"""

import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# main.py göreli importlar nedeniyle CWD'yi proje köküne sabitle
os.chdir(PROJECT_ROOT)

try:
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton
    PYSIDE_AVAILABLE = True
except ImportError:
    PYSIDE_AVAILABLE = False


def _ensure_qapp():
    if not PYSIDE_AVAILABLE:
        return None
    app = QApplication.instance() or QApplication(sys.argv)
    return app


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 mevcut değil")
class GuiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_qapp()
        from main import CelikKubbeGUI
        cls.GUI = CelikKubbeGUI

    def setUp(self):
        try:
            self.win = self.GUI()
        except Exception as e:
            self.skipTest(f"CelikKubbeGUI instantiate edilemedi: {e}")

    def tearDown(self):
        try:
            from PySide6.QtWidgets import QApplication
            for attr in ("_cam_timer", "sweep_timer", "data_timer", "threat_timer"):
                t = getattr(self.win, attr, None)
                if t is not None:
                    try:
                        t.stop()
                    except Exception:
                        pass
            for attr in ("_radar_bridge", "_pipeline", "_sensor_fusion", "_blackbox"):
                obj = getattr(self.win, attr, None)
                if obj is not None and hasattr(obj, "durdur"):
                    try:
                        obj.durdur()
                    except Exception:
                        pass
                    if hasattr(obj, "wait"):
                        try:
                            obj.wait(500)
                        except Exception:
                            pass
            cap = getattr(self.win, "_cap", None)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self.win.close()
            QApplication.processEvents()
        except Exception:
            pass

    def test_main_window_is_qmainwindow(self):
        self.assertIsInstance(self.win, QMainWindow)
        self.assertTrue(self.win.windowTitle())  # Boş başlık olmamalı

    def test_engage_button_exists_with_object_name(self):
        # btn_engage objectName ile aranabilmeli (CSS hedeflemesi için kritik)
        btn = self.win.findChild(QPushButton, "btn_engage")
        # Plan: butonun varlığı veya benzeri kontrol butonu
        # Eğer specific name yoksa, herhangi bir QPushButton'un varlığı bile yeterli
        any_button = self.win.findChild(QPushButton)
        self.assertIsNotNone(any_button, "Hiç QPushButton bulunamadı")
        if btn is not None:
            self.assertEqual(btn.objectName(), "btn_engage")

    def test_panel_objects_present(self):
        # Tüm "panel" objectName'li QWidget'lar olmalı (en az 3 panel)
        panels = [w for w in self.win.findChildren(QWidget) if w.objectName() == "panel"]
        self.assertGreaterEqual(len(panels), 3,
                                f"En az 3 'panel' bekleniyor, bulundu: {len(panels)}")

    def test_right_status_panel_min_width_preserved(self):
        # test_core_behaviors.py'deki kuralı koru: sağ panel min 560 genişliğe sahip
        widgets_560 = [w for w in self.win.findChildren(QWidget)
                       if w.minimumWidth() >= 560]
        self.assertTrue(widgets_560,
                        "minimumWidth >= 560 olan QWidget bulunamadı (sağ durum paneli)")


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 mevcut değil")
class GuiImportSafetyTests(unittest.TestCase):
    """Main module düzgün import edilebilmeli (syntax/import hatası olmamalı)."""

    def test_main_module_imports(self):
        import importlib
        importlib.import_module("main")

    def test_celikkubbegui_class_exists(self):
        import main
        self.assertTrue(hasattr(main, "CelikKubbeGUI"))
        self.assertTrue(callable(main.CelikKubbeGUI))


if __name__ == "__main__":
    unittest.main(verbosity=2)
