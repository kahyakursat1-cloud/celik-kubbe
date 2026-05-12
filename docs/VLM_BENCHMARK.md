# VLM Model Seçimi — ContextFusion için Benchmark Planı

**Durum:** İskelet — gerçek ölçümler işlem yapıldıkça eklenecek.
**Hedef donanım:** Workstation (RTX 3060) + Jetson Orin NX 16GB (INT8).
**Karar:** [boş — Phase 1 sonunda doldurulacak]

---

## 1. Aday Modeller

| Model | Boyut (params) | INT8 RAM | FP16 RAM | HF Repo |
|---|---|---|---|---|
| Qwen2-VL-2B-Instruct | 2.2B | ~2.5 GB | ~4.5 GB | `Qwen/Qwen2-VL-2B-Instruct` |
| Qwen2-VL-7B-Instruct | 7.6B | ~7 GB | ~15 GB | `Qwen/Qwen2-VL-7B-Instruct` |
| MiniCPM-V 2.6 | 8B | ~8 GB | ~16 GB | `openbmb/MiniCPM-V-2_6` |
| PaliGemma-3B-mix-224 | 3B | ~3.5 GB | ~6 GB | `google/paligemma-3b-mix-224` |

---

## 2. Değerlendirme Sahneleri

5 temsili C-UAV sahnesinde her model çalıştırılır:

1. **Tek drone yakın (S1):** Tek hedef, 2 km, yaklaşan, açık hava.
2. **Çoklu tehdit (S2):** 2 drone + 1 helikopter aynı karede.
3. **Payload/anomaly (S3):** Drone alt tarafında görünür mount.
4. **Bird-vs-drone (S4):** Düşük conf YOLO, gerçek görüntü (Drone-vs-Bird benchmark).
5. **Düşük SNR / sis (S5):** Belirsiz silüet, kötü hava.

---

## 3. Ölçütler

| # | Ölçüt | Nasıl ölçülür |
|---|---|---|
| M1 | Sahne tasviri kalitesi | İnsan değerlendirici (1-5 skala): doğruluk, tamamlık, kısalık |
| M2 | Hallucination oranı | Bilinen-yok objelerin (örn. "missile" yokken "missile" denmesi) frekansı / 5 sahne |
| M3 | INT8 kalite kaybı | M1'in FP16 ve INT8 versiyonları arasındaki Δ |
| M4 | Workstation FP16 latency | Mean / p95 ms (RTX 3060) |
| M5 | Workstation INT8 latency | Mean / p95 ms (RTX 3060) |
| M6 | Jetson INT8 latency | Mean / p95 ms (Orin NX 16GB) — sığarsa |
| M7 | Bellek footprint | nvidia-smi VRAM kullanımı |
| M8 | JSON parse başarı oranı | Yapılandırılmış çıktının parse edilebilme oranı / 5 sahne |

---

## 4. Sonuçlar — [Doldurulacak]

```
Bu tablo Phase 1 (1 hafta) sonunda benchmark çalıştırıldıktan sonra doldurulur:

| Model            | M1 | M2 | M3 | M4(FP16) | M5(INT8) | M6(Jetson) | M7 | M8 |
|------------------|----|----|----|----------|----------|------------|----|----|
| Qwen2-VL-2B      |    |    |    |          |          |            |    |    |
| Qwen2-VL-7B      |    |    |    |          |          | DNF        |    |    |
| MiniCPM-V 2.6    |    |    |    |          |          | DNF        |    |    |
| PaliGemma-3B     |    |    |    |          |          |            |    |    |
```

---

## 5. Karar Kriterleri

- **Edge-first:** Jetson Orin NX 16GB'da INT8 inference < 500 ms ortalama.
- **Kalite eşiği:** M1 ≥ 3.5/5 ortalama.
- **Hallucination eşiği:** M2 ≤ 1/5 sahne.
- **Reproducibility:** HuggingFace repo + commit/revision sabit, INT8 calibration deterministik.

---

## 6. Tercih Edilen (varsayım)

**Birinci tercih:** Qwen2-VL-2B-Instruct (INT8)
**Yedek (workstation only):** Qwen2-VL-7B-Instruct (FP16)
**Karar tarihi:** [boş]
**Karar gerekçesi:** [boş — benchmark sonuçlarına göre]

---

## 7. Test Komutları (henüz çalıştırılmadı)

```bash
# Workstation benchmark
python docs/vlm_select.py --models qwen2vl-2b qwen2vl-7b paligemma-3b --eval-set tests/vlm_eval_scenes/
# Çıktı: docs/figures/vlm_benchmark_results.json

# Jetson sığma testi (gerçek cihazda)
python deployment/test_vlm_jetson_fit.py --model Qwen/Qwen2-VL-2B-Instruct --int8
```

---

## 8. Reproducibility

Her brief çıktısının altında footer:
```
model_id: Qwen/Qwen2-VL-2B-Instruct
revision: <git-sha>
prompt_template_hash: <md5>
temperature: 0
max_new_tokens: 200
```
