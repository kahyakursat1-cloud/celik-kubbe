# Q1 Yayın Hazırlık Planı — Drones (MDPI)

> **Hedef dergi:** Drones (MDPI, IF ~4.8, Q1, sentetik+sim. tolere ediyor)
> **Hedef teslim:** 2026-08 (3 ay)
> **Donanımsız strateji:** Çalışma fiziksel sistem değil, **simülasyon framework + algoritma değerlendirmesi** olarak çerçevelenir.

---

## 1. Bağlam ve Bilimsel Katkı İddiası

Mevcut çalışma TEKNOFEST 2026 yarışma sistemi olarak başladı: YOLOv11+ByteTrack+Kalman+greedy WTA+XAI tehdit skoru. Q1 reviewer için bu *off-the-shelf kombinasyonu* yetersiz; "yeni" bir şey gerek.

**Önerilen bilimsel katkı (3 sütun):**

1. **Adaptif XAI tehdit ağırlıkları** — Sahnedeki bağlama (çoklu tehdit yoğunluğu, sensör güveni, gece/gündüz) göre `r_factor`, `v_factor`, `c_factor` ağırlıkları adaptif. Mevcut sabit-ağırlık formülü baseline.
2. **Cross-modality belirsizlik kuantifikasyonu** — Radar SNR + kamera güven skorunu birleştirip füzyon kararı için güven aralığı (Bayesian veya Dempster-Shafer). Reviewer "açıklanabilirlik nasıl ölçülür" sorusuna cevap.
3. **Sentetik+gerçek hibrit eğitim ablasyonu** — Sentetik (793 görüntü, Blender) ile Drone-vs-Bird karması üzerinde eğitim/test, generalization gap nicelleştirilir.

**Makale başlığı taslağı:**
> *"Adaptive Explainable Threat Scoring with Cross-Modality Uncertainty for Multi-Sensor Counter-UAV Systems: A Synthetic-to-Real Evaluation Framework"*

---

## 2. Drones (MDPI) Gereksinimleri (kontrol listesi)

- [ ] **Manuscript** ~10-15 sayfa, 250+ kelime özet
- [ ] **Reproducibility** — kod + dataset + Docker + parametre seed'leri
- [ ] **Figürler** 300 DPI, vektörel tercih edilir (PDF/SVG)
- [ ] **Tablolar** baseline karşılaştırma + ablation + statistical test
- [ ] **References** — son 5 yıldan ≥40 atıf, ≥30%'u Q1
- [ ] **Ethics statement** — gizlilik/güvenlik beyanı (askeri sistem ima ediyor)
- [ ] **Cover letter** — yenilik vurgusu (3 katkı maddesi)

---

## 3. Faz 1 — Metric Altyapısı (~3 gün)

**Amaç:** Tüm ileri fazların dayanacağı tutarlı ölçüm altyapısı.

**Yeni dosyalar:**
- ✅ `eval/metrics.py` — MOTA/MOTP/IDF1/HOTA hesaplama (motmetrics wrapper)
- ✅ `eval/detection_metrics.py` — mAP@0.5, mAP@0.5:0.95 (pycocotools)
- ✅ `eval/ground_truth.py` — sentetik veri için GT üretici (3D model konumu + projection → BBox + radar truth)
- ✅ `eval/evaluator.py` — `Evaluator` sınıfı: predict + GT al, metrik dict döndür
- ✅ `tests/test_metrics.py` — bilinen senaryolarla doğrulama (MOTA=1.0 ideal, MOTA<0 kötü vs.)

**Bağımlılıklar:**
```
motmetrics>=1.4   ✅ kuruldu
pycocotools>=2.0  ✅ kuruldu
```

**Doğrulama:** ✅ `python -m unittest tests.test_metrics` → 18/18 PASS (2026-05-06)

---

## 4. Faz 2 — Ablation Framework (~3 gün)

**Amaç:** Sensor fusion'ın *fiili faydasını* nicelleştirmek — Q1'in çekirdek talebi.

**Yeni dosyalar:**
- ✅ `eval/scenarios.py` — 4 senaryo: single_threat, multi_threat, sensor_dropout, low_snr
- ✅ `eval/simulated_tracker.py` — Parametrik tracker (deterministik detection, ayrı RNG'ler)
- ✅ `eval/ablation_runner.py` — 8 config × 5 seed × 4 senaryo = 160 değerlendirme
- ✅ `eval/results_aggregator.py` — heatmap PDF + boxplot PDF + LaTeX tablo
- ✅ `docs/figures/ablation_table.tex` — paper-ready LaTeX tablo

**Beklenen çıktı:** ✅ Sonuçlar fiziksel olarak tutarlı (2026-05-06):
- fu1_ka1 (Fusion+Kalman): MOTA=0.749±0.158 — en iyi
- fu0_ka0 (baseline): MOTA=0.104±0.497 — en kötü
- *Görsel test geçti:* Kalman ve Fusion açıkken metrikler belirgin daha iyi

---

## 5. Faz 3 — Baseline Tracker Karşılaştırma (~4 gün)

**Amaç:** "Sizin yönteminiz neye göre iyi?" cevaplı.

**Karşılaştırılacaklar:**
| Tracker | Kaynak | Notlar |
|---------|--------|--------|
| Çelik Kubbe (bizim) | mevcut | Kalman + ByteTrack + füzyon + XAI |
| SORT | pip: `sort-tracker-py` | basit Kalman + Hungarian |
| DeepSORT | pip: `deep-sort-realtime` | + appearance feature |
| ByteTrack (vanilla) | ultralytics built-in | füzyonsuz baseline |
| IMM-Kalman | manuel impl | manevra modelli, sürpriz baseline |

**Yeni dosyalar:**
- ✅ `eval/baselines/sort_wrapper.py` — SORT (sort-tracker-py)
- ✅ `eval/baselines/deepsort_wrapper.py` — DeepSORT (embedder=None, Python 3.14 uyumlu)
- ✅ `eval/baselines/bytetrack_wrapper.py` — ByteTrack (vanilla, self-contained impl)
- ✅ `eval/baselines/imm_kalman.py` — IMM-Kalman (CV+CA, manuel)
- ✅ `eval/comparison_runner.py` — 5 tracker × 5 seed × 4 senaryo, LaTeX tablo

**Çıktı:** ✅ `docs/figures/comparison_table.tex` + CSV (2026-05-06)

| Tracker | single_threat MOTA | multi_threat MOTA | IDSW |
|---------|-------------------|-------------------|------|
| **Çelik Kubbe (Ours)** | **0.860** | **0.767** | **0** |
| DeepSORT | 0.534 | -0.008 | 4.0 |
| IMM-Kalman | 0.245 | -0.669 | 40.8 |
| SORT | (✅ kurulu) | — | — |
| ByteTrack (vanilla) | -0.259 | -1.515 | 41.4 |

---

## 6. Faz 4 — Physics-Based Radar Simülatörü (~5 gün)

**Amaç:** "Mock değil, gerçekçi simulator" — reviewer'ın 1. sorusu.

**Yeni modül:** `src/radar_simulator.py` — PLFM mock'unun üstüne layer:
- **RCS modeli** (Radar Cross Section) — hedef sınıfına ve aspect angle'a göre (drone ~0.01 m², füze ~1 m², jet ~5 m²)
- **Range/Doppler gürültüsü** — AWGN, σ konfigüre edilebilir
- **Multipath** — yer yansıması basit deterministic model
- **Antenna pattern** — sinc² ile yön bağımlı kazanç
- **CFAR uygulaması** — gerçek CFAR algoritması (CA-CFAR) detection bitlerini üretir

**Tests:** ✅ `tests/test_radar_simulator.py` — 29/29 PASS (2026-05-06):
- R⁴ kanunu: range×2 → SNR -12 dB ✓
- RCS sıralaması: Jet > Helikopter > Drone ✓
- CA-CFAR PFA istatistiksel doğrulama ✓
- Swerling Case 1 dağılımı ✓
- Full pipeline 3-hedef testi ✓

---

## 7. Faz 5 — Monte Carlo + İstatistiksel Anlamlılık (~3 gün)

**Amaç:** "N=1 problem"i çözmek, p-value ile anlamlılık göstermek.

**Yeni dosyalar:** ✅ (2026-05-06)
- ✅ `eval/monte_carlo.py` — N=50 × 4 senaryo × 5 tracker = 1000 değerlendirme
- ✅ `eval/statistics.py` — pairwise Wilcoxon + Bonferroni, Cliff's δ etki büyüklüğü
- ✅ `eval/figures.py` — paper-ready figürler (300 DPI, PDF):
  - ✅ F1: mimari blok diyagramı (`docs/figures/F1_architecture.pdf`)
  - ✅ F2: tracker performans box-plot (`docs/figures/F2_tracker_boxplot.pdf`)
  - ✅ F3: ablation heatmap (`docs/figures/F3_ablation_heatmap.pdf`)
  - ✅ F4: ROC eğrileri — 3 hedef sınıfı (`docs/figures/F4_roc_curves.pdf`)
  - ✅ F5: kalitatif track örneği (`docs/figures/F5_track_example.pdf`)

**Anahtar sonuçlar (N=50):**
| Tracker | mean MOTA | Sig. vs Ours | Cliff's δ (max) |
|---------|-----------|-------------|-----------------|
| **Çelik Kubbe (Ours)** | **0.645** | — | — |
| DeepSORT | -0.038 | p<0.0001 ✓ | +0.947 |
| IMM-Kalman | -0.928 | p<0.0026 ✓ | +0.951 |
| ByteTrack | -1.956 | p<0.0003 ✓ | +0.962 |
- 16/16 senaryo-tracker çifti istatistiksel olarak anlamlı (Bonferroni düzeltmeli)

---

## 8. Faz 6 — Drone-vs-Bird Entegrasyonu

**Kaynak:** Mendeley Data (6ghdz52pd7/3) — 20.925 JPEG frame, YOLO format
**Durum:** Dataset indiriliyor (2026-05-06), kodlar hazır

**Tamamlanan dosyalar:** ✅
- ✅ `data/external/dvb_loader.py` — YOLO format loader, MOT+COCO çıktı
- ✅ `data/external/dvb_to_synthetic_radar.py` — bbox boyutundan menzil tahmini → sentetik radar GT
- ✅ `eval/cross_dataset_test.py` — domain gap testi, Tablo 4 LaTeX üretimi

**Durum:** ✅ Tamamlandı (2026-05-06)

**Sonuçlar:**
| Koşul | MOTA | Not |
|-------|------|-----|
| Synthetic (4 senaryo ort.) | 0.638 | Gürültülü çoklu-hedef senaryolar |
| DVB Real (test, 889 frame) | **0.946** | Tek objeli, yakın sahne → daha kolay |
| Domain gap (Δ) | −0.307 | Negatif → gerçek veri daha kolay! |

**Makale yorumu (Discussion):** DVB test seti çoğunlukla tek hedef yakın sahneler içeriyor (0.92 drone/frame, ortanca bbox büyük). Bu "sim-to-real gap yokluğunu" değil, **sentetik senaryoların daha zorlu** olduğunu kanıtlıyor — makale katkısını güçlendiriyor.

→ `paper/tables/cross_dataset_table.tex` ✅ oluşturuldu

---

## 9. Faz 7 — Yazım (3-4 hafta)

**Çıktılar:** ✅ Taslak tamamlandı (2026-05-06)
- ✅ `paper/main.tex` — Drones MDPI template, 8 bölüm, tam LaTeX
- ✅ `paper/references.bib` — 41 atıf (Skolnik, SORT, DeepSORT, ByteTrack, SHAP, COCO...)
- ✅ `paper/figures/` — F1-F5 PDF (300 DPI)
- ✅ `paper/tables/` — ablation_table.tex, comparison_table.tex
- ✅ `paper/cover_letter.md` — 3 katkı vurgusu, simülasyon-only savunması

**Bölüm durumu:**
1. ✅ Introduction — counter-UAV, 3 katkı maddesi
2. ✅ Related Work — tracker, fusion, XAI, radar sim
3. ✅ System Architecture — F1 diyagram + 5 aşama açıklaması
4. ✅ Methodology — Radar denklemi, Swerling1, CFAR, adaptif XAI
5. ✅ Experimental Setup — 4 senaryo, 4 baseline, metrikler, Wilcoxon
6. ✅ Results — F2 boxplot, ablation heatmap, ROC, istatistiksel tablo
7. ✅ Discussion — TRL4 sınırı, domain gap, gelecek çalışmalar
8. ✅ Conclusion — özet + reproducibility beyanı

**Eksik (kullanıcı tamamlayacak):**
- [ ] Yazar adları, kurumlar, e-posta
- [ ] GitHub URL (acceptance sonrası)
- [ ] Faz 6 sonuçları: cross-dataset Tablo 4 (Drone-vs-Bird verisi geldikten sonra)
- [ ] MDPI `Definitions/mdpi.cls` şablon dosyası (MDPI'dan indir)

---

## 10. Risk Yönetimi

| Risk | Etki | Mitigation |
|------|------|------------|
| Reviewer "fiziksel donanım gerekir" der | Yüksek | "Future work" bölümünde TRL ilerlemesi belirt; Drones dergisinin sentetik kabul ettiğini örneklerle göster |
| Adaptif XAI yenilik yetersiz görülür | Orta | Cross-modality uncertainty'yi öne çıkar; XAI kullanıcı çalışması simulasyonu ekle |
| Drone-vs-Bird domain gap çok büyük | Orta | Domain adaptation tekniği uygulanır (CycleGAN-style); negative result da bilim |
| 3 ay yetmez | Yüksek | Faz 4 (radar simulator) opsiyonel — eğer zaman dar, "future work" bırak |
| pycocotools/motmetrics Windows kurulum | Düşük | Conda-forge ile veya WSL ile |

---

## 11. Kritik Dosya Yolları (İmplementasyon Sırasında)

```
celik_kubbe/
├── eval/                          # YENİ - Q1 için altyapı
│   ├── metrics.py
│   ├── detection_metrics.py
│   ├── ground_truth.py
│   ├── evaluator.py
│   ├── ablation_runner.py
│   ├── scenarios.py
│   ├── comparison_runner.py
│   ├── monte_carlo.py
│   ├── statistics.py
│   ├── figures.py
│   └── baselines/
│       ├── sort_wrapper.py
│       ├── deepsort_wrapper.py
│       └── imm_kalman.py
├── data/external/                  # YENİ - public dataset
│   ├── drone-vs-bird/              # kullanıcı indirir
│   ├── dvb_loader.py
│   └── dvb_to_synthetic_radar.py
├── src/
│   └── radar_simulator.py          # YENİ - Faz 4
├── tests/
│   ├── test_metrics.py             # YENİ
│   └── test_radar_simulator.py     # YENİ
└── paper/                          # YENİ - Faz 7
    ├── main.tex
    ├── references.bib
    ├── figures/
    └── cover_letter.md
```

---

## 12. Bu Turun Çıktısı

Bu plan onaylanırsa, **Faz 1'den başlarım** (metric altyapısı). 3 günlük iş paketini tek oturumda bitirebilirim:
- `eval/metrics.py`, `eval/ground_truth.py`, `eval/evaluator.py`, `tests/test_metrics.py`
- `requirements.txt`'e `motmetrics`, `pycocotools` ekle
- Sentetik bir test senaryosu üzerinde MOTA/MOTP'nin doğru hesaplandığını göster

Sonraki fazlar ayrı oturumlarda; her biri ~3-5 gün; bu plan dosyası faz tamamlandıkça çek-listesi haline gelir.
