# CLAUDE.md — Çelik Kubbe (Hava Savunma)

## Parent
`bilsem_beyin/CLAUDE.md` kurallarını devralır. Mod: 🏆 Yarışma (TEKNOFEST 2026)

## Proje
YOLOv11m tabanlı hava tehdit tespit + AERIS-10 radar sensör füzyonu. PySide6 GUI üzerinden gerçek zamanlı 6 sınıf hedef takibi (BalisticMissile, Drone, Helicopter, Jet, Artillery, FixedWingUAV) ve gimbal yönlendirme.

## Yapı
```
main.py            → 1418 satır PySide6 GUI (CelikKubbeGUI sınıfı)
config.yaml        → Merkezi yapılandırma (model, radar, gimbal, füzyon, tehditler)
src/
  konfig.py            → YAML loader (cfg dict)
  tespit_pipeline.py   → YOLOv11 + ByteTrack inference
  radar_bridge.py      → AERIS-10 USB (FT2232H/FT601) köprüsü
  sensor_fusion.py     → Radar+Kamera füzyonu, FusedTrack, XAI tehdit skoru
  kalman_filter.py     → KalmanFilter2D ([x,y,vx,vy] state)
  gimbal_controller.py → MAVLink gimbal pan/tilt
  wta_optimizer.py     → Weapon-Target Assignment
  battery_profiles.py  → PIL-ALFA/BETA/GAMMA/DELTA pil profilleri
  coordinate_utils.py  → km↔display radius dönüşümü
  blackbox_logger.py   → CSV görev kayıtları (tracks_*.csv, events_*.csv)
models/             → yolo11m_celikkubbe.pt + .onnx (Jetson'da .engine)
deployment/         → hw_smoke_test, export_trt, benchmark_jetson, hardware_validation.md
tests/              → unittest: core_behaviors, radar_bridge, kalman, blackbox, gimbal, fusion_edges, gui_smoke
docs/               → MIMARI.md, SENSOR_FUSION.md, WINDOWS_KULLANIM.md, YENIDEN_EGITIM.md
```

## Teknik Bağlam
- **avci_drone ile paylaşım:** `../shared/yarismalar_verisi.py`, `../shared/bilgi_paneli.py` aktif olarak kullanılıyor; tek kaynak. Kod kopyalama YOK
- **PLFM_RADAR bağımlılığı:** `D:/bilsem_beyin/radar/PLFM_RADAR/9_Firmware/9_3_GUI/radar_protocol.py` — `radar_bridge.py:35-50`'de hardcoded yol
- **Mock/Real geçişi:** `config.yaml` içinde `radar.mock` ve `gimbal.mock` flag'leri; geçiş öncesi `deployment/hardware_validation.md` checklist
- **Edge dağıtım:** Jetson Nano/Orin için `deployment/export_trt.py --half` (FP16) veya `--int8 --calib-dir` (INT8)

## Kurallar (Bu Projeye Özel)
- avci_drone'dan kod kopyalama → `shared/` altında ortak modül oluştur veya genişlet
- Yarışma şartnamesini (`docs/2026_Çelikkubbe_..._Şartname_TR_v1.3.pdf`) referans al; varsayımla hedef sınıfı ekleme
- Birim test ekleme stilinde mevcut `tests/test_*.py` desenini kullan: unittest + path setup (parents[1] PROJECT_ROOT, parents[2] shared root)
- Demo/sahne öncesi: `python diagnose.py` → `[READY]`, `python -m unittest discover -s tests` → tüm PASS

## Komutlar
```bash
# Mock sistem GUI
python main.py

# Sistem hazırlık taraması
python diagnose.py

# Donanım smoke (mock veya --real)
python deployment/hw_smoke_test.py --all

# Tüm testler
python -m unittest discover -s tests -v

# TensorRT export (Jetson'da)
python deployment/export_trt.py --half

# Inference benchmark
python deployment/benchmark_jetson.py --model models/yolo11m_celikkubbe.engine --synthetic
```

---
**Güncelleme:** 2026-05-05 | **Versiyon:** 2.0
