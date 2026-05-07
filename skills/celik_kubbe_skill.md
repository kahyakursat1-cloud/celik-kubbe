# ROLE
Air defense systems engineer — TEKNOFEST 2026 Çelik Kubbe.

# SYSTEM ARCHITECTURE
- 3 aşamalı hava savunma sistemi
- Platform: Statik radar + ateş kontrol + görüntü işleme

# TECH STACK
- YOLOv11m (yolo11m_celikkubbe.pt + ONNX) + ByteTrack
- TespitPipeline (src/tespit_pipeline.py) — tespit_sinyal(tespitler)
- Threat sınıfları: BalisticMissile, Drone, Helicopter, Jet, Artillery, FixedWingUAV
- Radar widget (PySide6 custom) — polar koordinat görselleştirme
- _kamera_tespitler: dict[track_id, Threat] — kamera→radar koordinat dönüşümü
- Kamera bbox → bearing=(cx-0.5)*120°, mesafe=bbox_diag*1000

# ŞARTNAME §5 AŞAMALAR
## Aşama 1: Farklı Menzillerde Duran Hedef İmhası
- Statik hedefler farklı mesafelerde
- Tespit → Sınıflandırma → Ateş komutu

## Aşama 2: Sürü Saldırısı ve İmhası
- Çok hedef simultane takibi
- ByteTrack track_id bazlı sürü takibi
- Sürü dağılım analizi

## Aşama 3: Farklı Katlardaki Hareketli Hedefler
- 3D uzayda hareketli hedef takibi
- İrtifa katmanı ayrımı

# RADAR → GERÇEK DÜNYA
- bearing = (cx - 0.5) × 120°
- mesafe_km = sqrt(w²+h²) × 1000 (kalibrasyona göre)
- Renk: kırmızı (düşman) / sarı (belirsiz) / yeşil (dost)
- _cam_timer: 33ms (30fps)

# LAUNCHER
- main.py — PySide6 başlatıcı
- baslat.bat — Windows çift-tıkla

# RULES
- ONNX export tercih edilir (deployment hızı)
- Tüm tehditler track_id ile kalıcı takip
- Simülasyon: model yoksa rastgele tehdit üret
