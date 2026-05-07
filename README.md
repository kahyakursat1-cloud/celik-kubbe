# Çelik Kubbe — TEKNOFEST 2026 Hava Savunma Sistemi

YOLOv11m + AERIS-10 radar sensör füzyonu üzerine kurulmuş, gerçek zamanlı 6 sınıf hava tehdit (BalisticMissile, Drone, Helicopter, Jet, Artillery, FixedWingUAV) tespit ve takip sistemi. PySide6 GUI, MAVLink gimbal kontrolü, ByteTrack/Kalman izleme, XAI tabanlı tehdit skorlaması.

## Hızlı Başlangıç

```bash
# 1) Bağımlılıklar
pip install -r requirements.txt

# 2) Sistem hazırlık taraması (READY çıkmalı)
python diagnose.py

# 3) GUI'yi mock modunda başlat
python main.py
# Windows: baslat.bat
```

## Belgeler

| Belge | Konu |
|-------|------|
| [docs/MIMARI.md](docs/MIMARI.md) | Modül akış diyagramı, sinyal-slot topolojisi |
| [docs/SENSOR_FUSION.md](docs/SENSOR_FUSION.md) | Eşleştirme skoru, Kalman 4-state, XAI tehdit formülü |
| [docs/WINDOWS_KULLANIM.md](docs/WINDOWS_KULLANIM.md) | Windows geliştirme ortamı kurulumu |
| [docs/YENIDEN_EGITIM.md](docs/YENIDEN_EGITIM.md) | YOLOv11 dataset hazırlama ve fine-tune |
| [deployment/README.md](deployment/README.md) | Linux/Jetson dağıtımı |
| [deployment/hardware_validation.md](deployment/hardware_validation.md) | Mock'tan gerçek donanıma geçiş checklist |
| [CLAUDE.md](CLAUDE.md) | Proje teknik kuralları (Claude Code için) |

## Yapı

```
main.py             GUI ana uygulaması (CelikKubbeGUI, QMainWindow)
config.yaml         Merkezi yapılandırma
diagnose.py         Sistem sağlık kontrolü
src/                Çekirdek modüller (radar, fuzyon, gimbal, kalman, vb.)
models/             YOLOv11 ağırlıkları (.pt + .onnx, Jetson'da .engine)
deployment/         Edge cihaz dağıtımı, smoke test, TensorRT export, benchmark
docs/               Mimari ve algoritma belgeleri
tests/              Birim testler (unittest, ~38 test)
shared/ → ../shared avci_drone ile paylaşılan modüller
```

## Mod Bayrakları (`config.yaml`)

| Bayrak | Mock | Gerçek |
|--------|------|--------|
| `radar.mock` | sentetik veri | FTDI USB |
| `gimbal.mock` | sinyal-içi simulasyon | pymavlink bağlantı |

Hibrit mod desteklenir (örn. `radar.mock=True` + `gimbal.mock=False`); her bileşen bağımsız initialize edilir.

## Test

```bash
# Tüm testler
python -m unittest discover -s tests -v

# Donanım smoke (mock veya --real)
python deployment/hw_smoke_test.py --all
```

## Edge Cihaz (Jetson)

```bash
# TensorRT engine üret (Jetson üzerinde)
python deployment/export_trt.py --half

# Inference benchmark
python deployment/benchmark_jetson.py --model models/yolo11m_celikkubbe.engine
```

Detay: [deployment/README.md](deployment/README.md)

## Lisans

TEKNOFEST 2026 yarışma katılımı kapsamında. Yeniden kullanım için ekip izni gerekir.
