# Windows — Geliştirme ve Çalıştırma Rehberi

Linux/Jetson dağıtımı için [../deployment/README.md](../deployment/README.md) ve [../deployment/hardware_validation.md](../deployment/hardware_validation.md) belgelerine bakın. Bu rehber Windows 10/11 geliştirme ortamını kapsar.

## Hızlı Başlangıç

```cmd
:: 1) Sanal ortam (önerilen, ama miniconda da OK)
python -m venv venv
venv\Scripts\activate

:: 2) Bağımlılıklar
pip install -r requirements.txt

:: 3) Hazırlık taraması
python diagnose.py

:: 4) GUI'yi başlat (mock modunda)
python main.py
```

## requirements.txt İçeriği (Asgari)

Eğer dosya yoksa şu paketleri yükleyin:

```
PySide6
opencv-python
numpy
ultralytics       # YOLOv11
torch torchvision # CUDA wheel: pip install torch --index-url https://download.pytorch.org/whl/cu118
pyyaml
h5py
openpyxl
pyftdi            # Radar USB
pymavlink         # Gimbal MAVLink
pyserial
```

## CUDA Kurulumu (Önerilir)

GPU inference için PyTorch CUDA wheel:

```cmd
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

`python diagnose.py` → `[OK] CUDA aktif — ... CUDA 11.8` görmeli.

Kurulum doğrusu yoksa CPU fallback otomatik etkin (yaklaşık 5-10× yavaş).

## COM Port Atama (Gimbal)

1. Aygıt Yöneticisi → Ports (COM & LPT)
2. Gimbal cihazını "USB Serial Port (COMx)" altında bul
3. `config.yaml` içinde `gimbal.port` değerini güncelle (örn. `COM4`)
4. `python deployment/hw_smoke_test.py --gimbal-only --real --port COM4`

Birden fazla cihaz varsa Aygıt Yöneticisi → Properties → Details → "Hardware Ids" ile FTDI/PX4 ayrımı yapın.

## FTDI USB (Radar)

Windows'ta `pyftdi` çoğunlukla zbus driver gerektirir. Kurulum:

```cmd
pip install pyftdi
```

İlk çalıştırmada Windows uyarı verirse Zadig (https://zadig.akeo.ie/) ile FTDI cihazına `libusb-win32` veya `WinUSB` driver atayın. Detay: pyftdi installation docs.

## baslat.bat (Opsiyonel)

Hızlı başlatma için kök dizine basit bir batch:

```batch
@echo off
cd /d %~dp0
call venv\Scripts\activate
python main.py
pause
```

## Sık Sorunlar

| Sorun | Çözüm |
|-------|-------|
| `ImportError: PLFM_AVAILABLE=False` | `D:\bilsem_beyin\radar\PLFM_RADAR\9_Firmware\9_3_GUI\radar_protocol.py` mevcut değil — mock'a geri dön |
| Kamera frame yok | Başka uygulama (Skype, Teams, browser tab) kamerayı tutuyor — kapat |
| Konsol Türkçe karakter bozuk | `chcp 65001` veya `set PYTHONIOENCODING=utf-8` |
| `Could not load Qt platform plugin "windows"` | PySide6'yı yeniden kur: `pip install --force-reinstall PySide6` |
| GUI açılıyor ama radar paneli boş | `config.yaml` → `radar.aktif: true` ve `radar.mock: true` |
| `model.export(format="engine")` hata | TensorRT Windows wheel sınırlı; export Jetson'da yapılmalı |

## Test Çalıştırma

```cmd
:: Tüm testler
python -m unittest discover -s tests -v

:: Belirli test
python -m unittest tests.test_sensor_fusion_edges -v

:: Smoke (donanımsız)
python deployment/hw_smoke_test.py --gimbal-only --camera-only
```
