"""
hw_smoke_test.py — Çelik Kubbe donanım smoke test harness.

Mock veya gerçek donanım modunda radar/gimbal/kamerayı ayrı ayrı veya
birlikte test eder. PySide6 olmadan minimal QCoreApplication event loop
ile çalışır.

Kullanım:
    python deployment/hw_smoke_test.py --all                  # Hepsi
    python deployment/hw_smoke_test.py --radar-only           # Sadece radar
    python deployment/hw_smoke_test.py --gimbal-only          # Sadece gimbal
    python deployment/hw_smoke_test.py --camera-only          # Sadece kamera
    python deployment/hw_smoke_test.py --all --real           # Gerçek donanım
    python deployment/hw_smoke_test.py --radar-only --real --usb ft601

Çıkış kodu:
    0  → tüm seçili testler PASS
    1  → en az bir test FAIL
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QCoreApplication, QTimer

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _ok(msg: str) -> None:
    print(f"{GREEN}[OK]{RESET}   {msg}")


def _fail(msg: str) -> None:
    print(f"{RED}[FAIL]{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def test_radar(mock: bool, usb_type: str, duration_s: float = 8.0) -> bool:
    from src.radar_bridge import RadarBridge, PLFM_AVAILABLE

    if not PLFM_AVAILABLE:
        _fail(f"PLFM_RADAR modülü import edilemedi (radar_protocol.py yolu: "
              f"{PROJECT_ROOT.parent / 'radar' / 'PLFM_RADAR' / '9_Firmware' / '9_3_GUI'})")
        return False

    app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    state = {
        "connected": False,
        "detection_count": 0,
        "frame_count": 0,
        "last_status": "",
    }

    bridge = RadarBridge(mock=mock, usb_type=usb_type, kayit_aktif=False)

    bridge.radar_baglanti_sinyal.connect(lambda b: state.update(connected=b))
    bridge.radar_durum_sinyal.connect(lambda s: state.update(last_status=s))
    bridge.radar_tespit_sinyal.connect(
        lambda lst: state.update(detection_count=state["detection_count"] + len(lst))
    )
    bridge.radar_frame_sinyal.connect(
        lambda _f: state.update(frame_count=state["frame_count"] + 1)
    )

    bridge.start()

    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(app.quit)
    timer.start(int(duration_s * 1000))
    app.exec()

    bridge.durdur()

    label = "MOCK" if mock else f"GERÇEK ({usb_type.upper()})"
    if not state["connected"]:
        _fail(f"Radar bağlantısı kurulamadı [{label}] — son durum: {state['last_status']!r}")
        return False

    _ok(f"Radar bağlantısı kuruldu [{label}]")
    print(f"       └─ {duration_s:.0f}s içinde {state['frame_count']} frame, "
          f"{state['detection_count']} CFAR tespiti")
    if state["frame_count"] == 0:
        _warn("Hiç frame alınmadı — donanım/mock veri akışı kontrol edilmeli")
    return True


def test_gimbal(mock: bool, port: str, baud: int) -> bool:
    from src.gimbal_controller import GimbalController

    app = QCoreApplication.instance() or QCoreApplication(sys.argv)

    state = {"signals": [], "logs": []}

    gimbal = GimbalController(mock=mock, port=port, baud=baud)
    gimbal.gimbal_durum_sinyal.connect(lambda d: state["signals"].append(d))
    gimbal.log_sinyal.connect(lambda s: state["logs"].append(s))

    gimbal.baslat()

    if not gimbal._bagli:
        label = "MOCK" if mock else f"GERÇEK ({port}@{baud})"
        last_log = state["logs"][-1] if state["logs"] else "—"
        _fail(f"Gimbal bağlantısı kurulamadı [{label}] — log: {last_log}")
        return False

    gimbal.hedefe_yonel(tid="SMOKE-001", bearing_deg=45.0, distance_km=1.5, altitude_m=300.0)
    gimbal.hedefe_yonel(tid="SMOKE-002", bearing_deg=-30.0, distance_km=0.8, altitude_m=150.0)
    gimbal.serbest_mod()

    QTimer.singleShot(200, app.quit)
    app.exec()

    gimbal.durdur()

    label = "MOCK" if mock else f"GERÇEK ({port}@{baud})"
    _ok(f"Gimbal bağlantısı kuruldu [{label}]")
    print(f"       └─ {len(state['signals'])} durum sinyali alındı, "
          f"{len(state['logs'])} log mesajı")
    if not state["signals"]:
        _warn("hedefe_yonel çağrılarına rağmen durum sinyali yok")
    return True


def test_camera(index: int, duration_s: float = 3.0) -> bool:
    try:
        import cv2
    except ImportError:
        _fail("opencv-python kurulu değil")
        return False

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        _fail(f"Kamera açılamadı (index={index}) — başka uygulama kullanıyor olabilir")
        return False

    t0 = time.monotonic()
    frame_count = 0
    last_shape = None
    while time.monotonic() - t0 < duration_s:
        ret, frame = cap.read()
        if ret and frame is not None:
            frame_count += 1
            last_shape = frame.shape

    cap.release()

    if frame_count == 0:
        _fail(f"Kamera açıldı ama {duration_s:.0f}s içinde frame alınamadı")
        return False

    fps = frame_count / duration_s
    _ok(f"Kamera frame okundu (index={index}) — {frame_count} frame "
        f"@ ~{fps:.1f} FPS, çözünürlük: {last_shape}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Çelik Kubbe donanım smoke test")
    parser.add_argument("--all", action="store_true", help="Tüm bileşenleri test et")
    parser.add_argument("--radar-only", action="store_true")
    parser.add_argument("--gimbal-only", action="store_true")
    parser.add_argument("--camera-only", action="store_true")
    parser.add_argument("--real", action="store_true", help="Gerçek donanım (mock=false)")
    parser.add_argument("--usb", choices=["ft2232h", "ft601"], default="ft2232h")
    parser.add_argument("--port", default="COM3", help="Gimbal seri port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--cam-index", type=int, default=0)
    parser.add_argument("--radar-duration", type=float, default=8.0)
    parser.add_argument("--cam-duration", type=float, default=3.0)
    args = parser.parse_args()

    if not any([args.all, args.radar_only, args.gimbal_only, args.camera_only]):
        parser.error("En az bir mod seçin: --all, --radar-only, --gimbal-only, --camera-only")

    mock = not args.real
    print(f"Mod: {'GERÇEK DONANIM' if args.real else 'MOCK'}")
    print("=" * 60)

    results: dict[str, bool] = {}

    if args.all or args.radar_only:
        print("\n[1/3] Radar (AERIS-10)")
        results["radar"] = test_radar(mock, args.usb, args.radar_duration)

    if args.all or args.gimbal_only:
        print("\n[2/3] Gimbal (MAVLink)")
        results["gimbal"] = test_gimbal(mock, args.port, args.baud)

    if args.all or args.camera_only:
        print("\n[3/3] Kamera")
        results["camera"] = test_camera(args.cam_index, args.cam_duration)

    print("\n" + "=" * 60)
    print("ÖZET")
    for name, ok in results.items():
        (_ok if ok else _fail)(f"{name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
