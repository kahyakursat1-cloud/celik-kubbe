"""
diagnose.py — Çelik Kubbe sistem sağlık kontrolü.

Demo/yarışma günü öncesi tek komutluk hazırlık taraması:
- Python bağımlılıkları
- GPU/CUDA durumu
- Model dosyalarının varlığı
- USB cihazları (radar/gimbal)
- config.yaml geçerliliği
- Log dizini disk alanı
- Şartname PDF varlığı

Çıkış: tüm kritikler OK ise [READY], aksi halde [NOT READY].
"""

from __future__ import annotations

import io
import os
import platform
import shutil
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

PROJECT_ROOT = Path(__file__).resolve().parent
critical_failures: list[str] = []
warnings: list[str] = []


def section(title: str) -> None:
    print(f"\n{CYAN}── {title} ─{'─' * (54 - len(title))}{RESET}")


def ok(msg: str) -> None:
    print(f"{GREEN}[OK]{RESET}    {msg}")


def fail(msg: str, critical: bool = True) -> None:
    print(f"{RED}[FAIL]{RESET}  {msg}")
    if critical:
        critical_failures.append(msg)


def warn(msg: str) -> None:
    print(f"{YELLOW}[WARN]{RESET}  {msg}")
    warnings.append(msg)


def check_import(module_name: str, critical: bool = True) -> bool:
    try:
        __import__(module_name)
        ok(f"{module_name}")
        return True
    except ImportError as e:
        fail(f"{module_name}: {e}", critical=critical)
        return False
    except Exception as e:
        fail(f"{module_name}: {type(e).__name__}: {e}", critical=critical)
        return False


def check_python_env() -> None:
    section("Python Ortamı")
    print(f"        Sürüm: {sys.version.split()[0]} ({platform.platform()})")
    print(f"        Yorumlayıcı: {sys.executable}")
    if sys.version_info < (3, 10):
        fail("Python 3.10+ önerilir")
    else:
        ok(f"Python sürümü uygun (>= 3.10)")


def check_dependencies() -> None:
    section("Çekirdek Bağımlılıklar")
    for mod in ("PySide6", "PySide6.QtWidgets", "PySide6.QtCore", "cv2",
                "numpy", "ultralytics", "yaml", "h5py", "openpyxl"):
        check_import(mod)

    section("Donanım Bağımlılıkları")
    check_import("pyftdi", critical=False)
    check_import("pymavlink", critical=False)
    check_import("serial", critical=False)


def check_gpu() -> None:
    section("GPU / CUDA")
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            cuda_v = torch.version.cuda
            ok(f"CUDA aktif — {count} cihaz, [{name}], CUDA {cuda_v}")
        else:
            warn("CUDA mevcut değil — inference CPU üzerinde yavaş çalışır")
    except ImportError:
        fail("torch kurulu değil")


def check_model_files() -> None:
    section("Model Dosyaları")
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from konfig import cfg
    except Exception as e:
        fail(f"konfig yüklenemedi: {e}")
        return

    pt_path = PROJECT_ROOT / cfg["model"]["yolu"]
    onnx_path = PROJECT_ROOT / cfg["model"]["onnx_yolu"]
    engine_path = pt_path.with_suffix(".engine")

    if engine_path.is_file():
        size_mb = engine_path.stat().st_size / 1024 / 1024
        ok(f"TensorRT engine ({size_mb:.1f} MB) — öncelikli, edge cihazda en hızlı")
    else:
        warn(f"TensorRT engine yok ({engine_path.name}) — Jetson'da export_trt.py çalıştırın")

    if pt_path.is_file():
        size_mb = pt_path.stat().st_size / 1024 / 1024
        ok(f"PyTorch model ({size_mb:.1f} MB): {pt_path.name}")
    else:
        fail(f"PyTorch model yok: {pt_path}")

    if onnx_path.is_file():
        ok(f"ONNX export mevcut: {onnx_path.name}")
    else:
        warn(f"ONNX yok: {onnx_path.name}")


def check_config() -> None:
    section("Yapılandırma (config.yaml)")
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from konfig import cfg
    except Exception as e:
        fail(f"config.yaml yüklenemedi: {e}")
        return

    required_sections = ("model", "kamera", "radar", "fuzyon", "gimbal",
                         "tehditler", "loglama")
    for s in required_sections:
        if s not in cfg:
            fail(f"config.yaml eksik bölüm: {s!r}")
        else:
            ok(f"bölüm: {s}")

    if cfg.get("radar", {}).get("mock") is True:
        warn("radar.mock=True — gerçek donanım için config.yaml güncelleyin")
    if cfg.get("gimbal", {}).get("mock") is True:
        warn("gimbal.mock=True — gerçek donanım için config.yaml güncelleyin")

    siniflar = cfg.get("tehditler", {}).get("siniflar", [])
    if len(siniflar) < 4:
        fail(f"tehdit sınıfı sayısı düşük: {len(siniflar)}")
    else:
        ok(f"tehdit sınıfı sayısı: {len(siniflar)}")


def check_usb_devices() -> None:
    section("USB Cihazları")
    try:
        from serial.tools import list_ports
    except ImportError:
        warn("pyserial kurulu değil — USB cihazları taranamadı")
        return

    ports = list(list_ports.comports())
    if not ports:
        warn("Hiç seri/USB port bulunamadı")
        return

    ftdi_found = False
    for p in ports:
        vid_pid = ""
        if p.vid is not None and p.pid is not None:
            vid_pid = f" [VID:PID={p.vid:04x}:{p.pid:04x}]"
        marker = ""
        if p.vid == 0x0403:
            ftdi_found = True
            marker = " ← FTDI (radar olabilir)"
        print(f"        {p.device}: {p.description}{vid_pid}{marker}")

    if ftdi_found:
        ok("FTDI cihaz tespit edildi (radar muhtemelen bağlı)")
    else:
        warn("FTDI cihaz yok — radar gerçek modunda çalışmayacak")


def check_disk_space() -> None:
    section("Disk Alanı (logs/)")
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    total, used, free = shutil.disk_usage(log_dir)
    free_mb = free / 1024 / 1024

    if free_mb < 500:
        fail(f"logs/ için yetersiz disk: {free_mb:.0f} MB (min 500 MB önerilir)")
    elif free_mb < 2000:
        warn(f"Disk düşük: {free_mb:.0f} MB serbest")
    else:
        ok(f"Disk yeterli: {free_mb:.0f} MB serbest")


def check_documentation() -> None:
    section("Şartname / Dokümantasyon")
    docs_dir = PROJECT_ROOT / "docs"
    pdfs = list(docs_dir.glob("*.pdf")) if docs_dir.exists() else []
    if not pdfs:
        warn("docs/ altında şartname PDF yok — jüri demo için referans")
    else:
        for p in pdfs:
            ok(f"PDF mevcut: {p.name}")


def check_local_modules() -> None:
    section("Yerel Modüller")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    for mod in ("konfig", "tespit_pipeline", "radar_bridge",
                "gimbal_controller", "sensor_fusion", "wta_optimizer",
                "kalman_filter", "blackbox_logger"):
        check_import(mod)

    shared_path = PROJECT_ROOT.parent / "shared"
    if shared_path.is_dir():
        sys.path.insert(0, str(shared_path))
        check_import("yarismalar_verisi", critical=False)
        check_import("bilgi_paneli", critical=False)
    else:
        warn(f"shared/ dizini yok: {shared_path}")


def main() -> int:
    print(f"{CYAN}=== Çelik Kubbe — Sistem Sağlık Kontrolü ==={RESET}")

    check_python_env()
    check_dependencies()
    check_gpu()
    check_model_files()
    check_config()
    check_local_modules()
    check_usb_devices()
    check_disk_space()
    check_documentation()

    print()
    print("=" * 60)
    if critical_failures:
        print(f"{RED}[NOT READY]{RESET} {len(critical_failures)} kritik sorun, "
              f"{len(warnings)} uyarı")
        for f in critical_failures:
            print(f"  - {f}")
        return 1

    if warnings:
        print(f"{YELLOW}[READY (with warnings)]{RESET} "
              f"{len(warnings)} uyarı — demo öncesi gözden geçirin")
        for w in warnings:
            print(f"  - {w}")
        return 0

    print(f"{GREEN}[READY]{RESET} Tüm kontroller geçti — sistem hazır")
    return 0


if __name__ == "__main__":
    sys.exit(main())
