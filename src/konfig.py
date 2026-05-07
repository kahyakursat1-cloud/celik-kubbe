"""Yapılandırma Yükleyici — Çelik Kubbe. config.yaml > varsayılan."""
from __future__ import annotations
import os, logging
from typing import Any

logger = logging.getLogger(__name__)

_VARSAYILAN: dict[str, Any] = {
    "model": {"yolu": "models/yolo11m_celikkubbe.pt",
              "onnx_yolu": "models/yolo11m_celikkubbe.onnx",
              "guven_esik": 0.40, "tracker": "bytetrack.yaml"},
    "kamera": {"index": 0, "fps": 30},
    "radar": {"aci_genisligi": 120.0, "mesafe_kalibrasyon": 1000,
              "aktif": True, "mock": True, "usb_tipi": "ft2232h",
              "kayit_aktif": False,
              "merkez_frekans_ghz": 10.5, "bant_genisligi_mhz": 200,
              "prf_hz": 5000, "maks_menzil_km": 3.0, "tarama_hizi_rpm": 15.0},
    "fuzyon": {"aktif": True, "aci_esigi_deg": 15.0,
               "mesafe_esigi_km": 1.0, "zaman_asimi_s": 5.0},
    "gimbal": {"aktif": True, "mock": True, "port": "COM3", "baud": 115200},
    "asama": {"aktif": 1},
    "simulasyon": {"rastgele_tehdit_arali": 2.0, "max_simule_tehdit": 8},
    "loglama": {"seviye": "INFO", "dosya": "logs/celik_kubbe.log",
                "max_bayt": 5_242_880, "yedek_sayisi": 3},
}

def _birlestir(t, u):
    s = t.copy()
    for k, v in u.items():
        s[k] = _birlestir(s[k], v) if k in s and isinstance(s[k], dict) and isinstance(v, dict) else v
    return s

def _yukle():
    p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    if not os.path.isfile(p): return _VARSAYILAN.copy()
    try:
        import yaml
        with open(p, encoding="utf-8") as f: return _birlestir(_VARSAYILAN, yaml.safe_load(f) or {})
    except ImportError:
        logger.warning("PyYAML kurulu değil — varsayılanlar kullanılıyor.")
        return _VARSAYILAN.copy()
    except Exception as e:
        logger.error(f"config.yaml okunamadı: {e}"); return _VARSAYILAN.copy()

cfg: dict[str, Any] = _yukle()
