# YOLOv11 Yeniden Eğitim Rehberi

Bu rehber, [src/train_celikkubbe.py](../src/train_celikkubbe.py) ile özel hava tehdit dataseti üzerinde modeli yeniden veya kademeli (fine-tune) eğitmek içindir.

## Dataset Yapısı

```
data/
  celikkubbe.yaml              # Ultralytics dataset config
  images/
    train/  *.png|*.jpg
    val/    *.png|*.jpg
  labels/
    train/  *.txt              # YOLO formatı: cls cx cy w h (normalize 0-1)
    val/    *.txt
```

Sınıf indeksleri **`config.yaml` `tehditler.siniflar` listesi ile aynı sırada** olmalı:

```yaml
# data/celikkubbe.yaml
path: D:/bilsem_beyin/celik_kubbe/data
train: images/train
val: images/val
names:
  0: BalisticMissile
  1: Drone
  2: Helicopter
  3: Jet
  4: Artillery
  5: FixedWingUAV
```

`config.yaml` ile `celikkubbe.yaml` sınıf adları **bire bir eşleşmeli**, aksi halde inference çıktısı yanlış etiketlenir.

## Sıfırdan Eğitim

```bash
python src/train_celikkubbe.py \
  --data data/celikkubbe.yaml \
  --weights yolo11m.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

`yolo11m.pt` ilk çağrıda Ultralytics tarafından otomatik indirilir.

## Fine-Tune (Mevcut Modeli Devam Ettirme)

Yeni veri eklediyseniz veya hyperparam tweak'i yapıyorsanız:

```bash
python src/train_celikkubbe.py \
  --data data/celikkubbe.yaml \
  --weights models/yolo11m_celikkubbe.pt \
  --epochs 30 \
  --lr0 0.001 \
  --resume
```

`--lr0 0.001` (varsayılanın 1/10'u) aşırı kayma riskini düşürür.

## Sentetik Veri Üretimi

Mevcut sentetik üretim araçları:

- [src/data/blender_generator.py](../src/data/blender_generator.py) — Blender ile 3D model render
- [src/data/rocket_factory.py](../src/data/rocket_factory.py) — füze modeli + varyasyon
- [src/data/thermal_shader.py](../src/data/thermal_shader.py) — termal görüntü stili
- [src/data/augment_realism.py](../src/data/augment_realism.py) — sentetik→gerçek domain adapt

Tipik akış: 3D model (.stl/.obj) → blender_generator → augment_realism → train/val ayrımı.

## Eğitim Sonrası

1. **Validation:** `runs/detect/train/weights/best.pt` ile mAP skorunu kontrol et
2. **ONNX export:** Inference cross-platform için
   ```bash
   yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=640
   ```
3. **Yeni modeli devreye al:**
   ```bash
   cp runs/detect/train/weights/best.pt models/yolo11m_celikkubbe.pt
   ```
   `config.yaml` zaten bu yola işaret ediyor; otomatik kullanılır.
4. **Smoke test:**
   ```bash
   python -m unittest tests.test_celikkubbe_pipeline.ModelFileTests
   python diagnose.py
   ```
5. **Edge dağıtım için TensorRT:** Jetson'da `python deployment/export_trt.py --half`

## Hyperparam İpuçları

| Parametre | Sentetik ağırlıklı dataset | Karma dataset |
|-----------|---------------------------|---------------|
| `epochs` | 150-200 (yavaş yakınsama) | 80-120 |
| `batch` | 16 (RTX 4060'da) / 8 (Jetson Orin) | aynı |
| `lr0` | 0.01 (sıfırdan) / 0.001 (fine-tune) | aynı |
| `mosaic` | 0.5 (sentetik için azalt) | 1.0 |
| `mixup` | 0.0 (sınıf karışımı bozar) | 0.1 |
| `hsv_v` | 0.4 (gece/gündüz) | 0.4 |

## Performans Beklentisi

| Setup | mAP@50 hedefi |
|-------|---------------|
| Yalnız sentetik (793 görüntü) | 0.65-0.75 |
| Sentetik + 200 gerçek | 0.80-0.85 |
| Sentetik + 1000+ gerçek | 0.90+ |

Düşük mAP nedenleri (sırasıyla kontrol):
1. Sınıf dengesizliği — `class_count.py` ile dağılımı kontrol et
2. Kötü etiketleme — birkaç düzine örnek manuel doğrula (Roboflow/CVAT)
3. Domain gap — `augment_realism.py` çıktılarını gerçek görüntülerle karşılaştır
4. Yetersiz epoch — overfit görmediyseniz daha uzun eğit
