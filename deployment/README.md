# Çelik Kubbe — Dağıtım (Deployment) Belgeleri

Bu dizin, sistemi uç cihazlara (Jetson Nano, Raspberry Pi 5 veya Linux PC) kurmak ve performansını optimize etmek için gerekli araçları içerir.

## Dosyalar

1. `launch.sh`: Sistemi başlatan ana kabuk betiği. Python ortamını (venv) ve USB yetkilerini kontrol edip `main.py` dosyasını çalıştırır.
2. `setup_usb.sh`: AERIS-10 Radar donanımı (FT2232H/FT601) için root yetkisi olmadan erişim sağlayan `udev` kurallarını yükler.
3. `export_trt.py`: Görüntü işleme performansını (FPS) artırmak için YOLOv11 modelini TensorRT'ye dönüştürür (FP16/INT8).
4. `benchmark_jetson.py`: Edge cihazda inference FPS/latency ölçümü, JSON çıktı.
5. `hw_smoke_test.py`: Radar/gimbal/kamera için bağımsız donanım smoke testi (mock veya `--real`).
6. `hardware_validation.md`: Mock'tan gerçek donanıma geçiş 8-adımlı checklist + troubleshooting.
7. `analyze_mission.py`: Görev sonrası `tracks_*.csv`/`events_*.csv` analizi — dashboard PNG + summary.txt.
8. `celikkubbe.service`: Uygulamayı cihaz her açıldığında otomatik olarak başlatan (kiosk mod) systemd servisi.

## Kurulum Adımları (Jetson Nano Örneği)

### 1. USB Radar Yetkileri
AERIS-10 radarını `sudo` kullanmadan okuyabilmek için:
```bash
sudo ./deployment/setup_usb.sh
```
Ardından USB kablosunu çıkarıp tekrar takın.

### 2. Model Optimizasyonu (Önerilen)
Jetson Nano üzerinde PyTorch modeli (`.pt`) yavaş çalışır. TensorRT'ye (`.engine`) dönüştürerek yüksek FPS alın:
```bash
cd deployment
python export_trt.py --model ../models/yolo11m_celikkubbe.pt --half
```
*(Not: Bu işlem Jetson üzerinde 15-20 dakika sürebilir)*
İşlem bittikten sonra `config.yaml` dosyasındaki model yolunu güncelleyin:
`yolu: "models/yolo11m_celikkubbe.engine"`

### 3. Otomatik Başlatma Servisi (Systemd)
Sistemin güç verildiğinde otomatik açılması için:
```bash
sudo cp deployment/celikkubbe.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable celikkubbe.service
sudo systemctl start celikkubbe.service
```
*(Not: `celikkubbe.service` içerisindeki dizin yollarını `/home/kullanici_adiniz/...` şeklinde kendi cihazınıza göre düzenlemeyi unutmayın.)*

### 4. Görev Sonrası Analiz
Sistem çalıştıktan sonra `logs/blackbox/` altında oluşan kayıtları görselleştirmek için:
```bash
# En son görevi otomatik bul
python deployment/analyze_mission.py --latest

# Belirli bir kaydı analiz et
python deployment/analyze_mission.py logs/blackbox/tracks_20260505_110608.csv

# Birden fazla görevi birleşik analiz et
python deployment/analyze_mission.py logs/blackbox/tracks_2026050*.csv
```
Çıktı: `logs/analysis/<timestamp>/dashboard.png` (6 panel: polar iz haritası, tehdit seviyesi, sınıf pie, kaynak dağılımı, range zaman serisi, olay zaman çizgisi) + `summary.txt` (metrik özeti).

Jüri demosu için: `dashboard.png`'i sunum slaytına ekleyin, `summary.txt`'i sözlü açıklama dayanağı olarak kullanın.
