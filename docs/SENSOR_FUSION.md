# Sensör Füzyon Algoritması — Teknik Detay

Bu belge, [src/sensor_fusion.py](../src/sensor_fusion.py) içindeki radar+kamera füzyon mantığını jüri/akademik raporlama için açıklar.

## 1. Eşleştirme Skoru

Her radar tespiti $r$ ile her aday kamera tespiti $k$ için bir skor hesaplanır; düşük skor = iyi eşleşme.

$$
S_{r,k} = \frac{|\Delta\theta|}{\theta_{esik}} + \frac{|\Delta R|}{R_{esik}}
$$

Burada:
- $\Delta\theta = \theta_{radar} - \theta_{kamera}$, kamera bearing'i $\theta_{kamera} = (c_x - 0.5) \cdot \text{FOV}_{kamera}$ ile hesaplanır (cx normalize bbox merkezi)
- $\Delta R = R_{radar} - \hat R_{kamera}$, kamera mesafe tahmini $\hat R = (1 - 2 d_{bbox}) \cdot 5$ km (bbox köşegen büyüklüğü)
- $\theta_{esik} = 15°$, $R_{esik} = 1$ km (yapılandırılabilir, `SensorFusion.ACI_ESIGI_DEG`/`MESAFE_ESIGI_KM`)

**Eşleşme koşulu:** $|\Delta\theta| \le \theta_{esik}$ AND $S_{r,k} < 2.0$. Çoklu eşleşme adayında en düşük skor seçilir (greedy assignment, tek tek tahsis).

## 2. İz Kaynak Sınıflandırması

Eşleştirme sonucuna göre `FusedTrack.kaynak`:
- **`fuzyon`** — radar+kamera birlikte eşleşti; en yüksek güven
- **`yalniz_radar`** — radar tespiti var, eşleşen kamera yok (gece/sis)
- **`yalniz_kamera`** — kamera tespiti var, radar yok (radar kapsamı dışı)

Bir iz `fuzyon`'a yükseldiyse, sonraki adımlarda yalnız tek sensör güncellese bile `kaynak="fuzyon"` korunur (downgrade yok). Bu, ölçüm geçici olarak kaybolduğunda iz güvenilirliğini bozmaz.

## 3. Kalman Filtresi (2D, Linear, Sabit Hız)

Durum ve ölçüm vektörleri:

$$
\mathbf{X} = \begin{bmatrix} x \\ y \\ \dot x \\ \dot y \end{bmatrix}, \qquad
\mathbf{Z} = \begin{bmatrix} x_{olculen} \\ y_{olculen} \end{bmatrix}
$$

Geçiş matrisi $F$ (sabit hız modeli, $\Delta t$ = 0.5 s):

$$
F = \begin{bmatrix}
1 & 0 & \Delta t & 0 \\
0 & 1 & 0 & \Delta t \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}, \qquad
H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}
$$

Süreç gürültüsü $Q$ "discrete white noise acceleration" formundadır (bkz. `kalman_filter.py:36-42`); ölçüm gürültüsü $R = r \cdot I_{2}$ ile $r$ varsayılan $1.0$.

**Coasting davranışı:** Eğer bir izin son güncellemesi mevcut zaman penceresinin altındaysa, yalnızca `predict()` çağrılır ve `range_km`/`bearing_deg` Kalman tahminine dayalı olarak yansıtılır. Bu, kısa süreli sensör kesintilerinde izin canlı kalmasını sağlar.

**İz düşme:** `son_guncelleme < simdi - IZ_ZAMAN_ASIMI_S` (varsayılan 5 s) izler `_aktif_izler`'den silinir.

## 4. XAI Tabanlı Tehdit Skoru (0–100)

Açıklanabilir, ağırlıklı toplam:

$$
S_{tehdit} = \big(\underbrace{r_{factor}}_{\text{mesafe}} + \underbrace{v_{factor}}_{\text{yaklaşma hızı}} + \underbrace{c_{factor}}_{\text{sınıf}}\big) \cdot \mu_{kaynak}
$$

**Bileşenler:**

| Faktör | Formül | Aralık |
|--------|--------|--------|
| $r_{factor}$ | $\max\{0, (5 - R_{km}) / 4.5\} \cdot 40$ | 0..40 |
| $v_{factor}$ | $\min\{40, |v|/2\}$ eğer $v < -10$ m/s, aksi 0 | 0..40 |
| $c_{factor}$ | Missile=50, UAV/Drone=25, Aircraft/Helicopter=15 | 15..50 |
| $\mu_{kaynak}$ | 1.1 eğer kaynak=`fuzyon`, aksi 1.0 | 1.0 veya 1.1 |

Final skor 100 ile sınırlanır; etiketleme:

| Skor | Seviye |
|------|--------|
| ≥ 80 | KRİTİK |
| ≥ 50 | YÜKSEK |
| ≥ 30 | ORTA |
| < 30 | DÜŞÜK |

## 5. Tasarım Tercihleri ve Kısıtlar

- **Greedy assignment** (Hungarian/JPDA değil): O(R·K) basitlik, küçük tespit sayılarında (≤ 20) yeterli
- **Polar→Kartezyen Kalman**: bearing'in modulo 360° ambiguity'si nedeniyle tahmin Cartesian'da yapılır, görselleştirme polar'da
- **Sabit hız modeli**: ivme öngörmez; manevra yapan hedeflerde IMM (Interacting Multiple Model) gerekirdi — yarışma kapsamı dışı
- **Sınıflandırma kamerada**: radar sınıf bilgisi vermez; her zaman `Bilinmeyen` etiketi taşır

## 6. Test Kapsamı

- Eşleştirme transition: [tests/test_sensor_fusion_edges.py](../tests/test_sensor_fusion_edges.py) `test_camera_only_then_radar_promotes_to_fusion`
- Tehdit skor: aynı dosya `test_threat_score_higher_for_close_fast_missile`, `test_distant_slow_target_is_low_threat`
- Kalman yakınsama: [tests/test_kalman_filter.py](../tests/test_kalman_filter.py) `test_constant_velocity_motion_converges_to_truth`
- İz timeout: `test_stale_track_dropped_after_timeout`

## Kaynaklar
- Bar-Shalom, Y., Li, X. R. *Estimation with Applications to Tracking and Navigation* (2001)
- Blackman, S. *Multiple Hypothesis Tracking for Multiple Target Tracking* (2004) — alternatif assignment
- Hwang, Y. K., et al. *XAI for Threat Assessment in C2 Systems* (2022) — açıklanabilir skorlama prensibi
