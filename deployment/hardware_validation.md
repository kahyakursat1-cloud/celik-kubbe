# Donanım Validasyon Rehberi — Mock'tan Gerçek Donanıma Geçiş

Bu belge, Çelik Kubbe sisteminin mock modundan gerçek donanım moduna güvenli geçişi için adım adım bir validasyon protokolü sunar. Her aşamada bir bileşen test edilir, sorun olursa sonraki adıma geçmeden çözülür.

## Ön Koşullar

- `python diagnose.py` → tüm bağımlılıklar OK + `[READY]` çıktısı
- `pyftdi`, `pymavlink`, `opencv-python` paketleri kurulu
- Linux: `udev` kuralları yüklü (aşağıda Adım 1)
- Windows: COM port atama kontrol edilmiş (Aygıt Yöneticisi)

---

## Adım 1 — USB ve Cihaz Tespiti

### Linux (Jetson Nano/Orin, Ubuntu)

```bash
# Udev kurallarını yükle (tek seferlik)
sudo ./deployment/setup_usb.sh

# USB kablosunu çıkar-tak, sonra cihazı doğrula
lsusb | grep 0403
# FT2232H için bekleniyor: ID 0403:6010 Future Technology Devices ...
# FT601 için bekleniyor:    ID 0403:601f Future Technology Devices ...
```

Görünmüyorsa: kablo değişimi → farklı USB portu → `dmesg | tail -20` (kernel hatası)

### Windows

```powershell
# Cihaz Yöneticisi → Universal Serial Bus controllers → "USB Serial Converter A/B"
# COM port: Ports (COM & LPT) → "USB Serial Port (COM3)" veya benzeri
```

`config.yaml` içinde `gimbal.port` doğru COM atamasıyla eşleşmeli.

---

## Adım 2 — Radar Mock Doğrulama

```bash
python deployment/hw_smoke_test.py --radar-only --radar-duration 10
```

**Beklenen:** `[OK] Radar bağlantısı kuruldu [MOCK]` + birkaç frame ve CFAR tespiti.

Geçmezse: `radar_bridge.py` PLFM_RADAR yolu hardcoded — `D:\bilsem_beyin\radar\PLFM_RADAR\9_Firmware\9_3_GUI\radar_protocol.py` mevcut olmalı.

---

## Adım 3 — Radar Gerçek Donanım

```bash
# FT2232H (USB 2.0)
python deployment/hw_smoke_test.py --radar-only --real --usb ft2232h --radar-duration 15

# FT601 (USB 3.0, AERIS-10E)
python deployment/hw_smoke_test.py --radar-only --real --usb ft601 --radar-duration 15
```

**Beklenen:** `[OK] Radar bağlantısı kuruldu [GERÇEK (FT2232H)]` + frame sayısı > 0.

### Sorun Giderme — Radar

| Hata | Sebep | Çözüm |
|------|-------|-------|
| `pyftdi.ftdi.FtdiError: UsbError(... Errno 13... Access denied)` | udev kuralı yok / yüklenmemiş | `sudo ./deployment/setup_usb.sh`, kablo çıkar-tak |
| `UsbError: Errno 16 (Resource busy)` | başka bir süreç FTDI'yı tutuyor | `sudo lsof | grep ftdi`, çakışan süreci durdur (örn. `ftdi_sio` kernel modülü: `sudo rmmod ftdi_sio`) |
| `BAĞLANTI KESİLDİ` ama hata mesajı yok | kısa kablo / EMI gürültüsü | shielded USB 3.0 kablo; FPGA güç beslemesi yeterli mi? |
| 0 frame | FPGA bitstream yüklü değil | AERIS-10 dökümantasyonuna göre bitstream flashla |
| Frame var ama 0 CFAR tespiti | sahne boş + CFAR eşiği yüksek | `bridge.komut_gonder(opcode_cfar, value)` ile eşik düşür |

---

## Adım 4 — Gimbal Mock Doğrulama

```bash
python deployment/hw_smoke_test.py --gimbal-only
```

**Beklenen:** `[OK] Gimbal bağlantısı kuruldu [MOCK]` + ≥ 3 durum sinyali.

---

## Adım 5 — Gimbal Gerçek Donanım

```bash
# Linux
python deployment/hw_smoke_test.py --gimbal-only --real --port /dev/ttyUSB0 --baud 115200

# Windows
python deployment/hw_smoke_test.py --gimbal-only --real --port COM3 --baud 115200
```

**Beklenen:** `[OK] Gimbal bağlantısı kuruldu [GERÇEK (...)]`

### Sorun Giderme — Gimbal

| Hata | Sebep | Çözüm |
|------|-------|-------|
| `serial.SerialException: could not open port 'COM3'` | port mevcut değil / başka uygulama açık | Aygıt Yöneticisi'nde COM doğrula; MissionPlanner/QGroundControl açıksa kapat |
| `serial.SerialException: Permission denied` (Linux) | kullanıcı `dialout` grubunda değil | `sudo usermod -a -G dialout $USER`, oturum yenile |
| Bağlandı ama `gimbal_durum_sinyal` yok | MAVLink target_system uyuşmazlığı | `mavutil.mavlink_connection().wait_heartbeat()` ile system_id öğren, kod gerekiyorsa güncelle |
| Pan/Tilt komutları kabul edilmiyor | gimbal MAV_MOUNT_MODE ayarı | Gimbal firmware'inde "MAVLink targeting" modunu etkinleştir |

---

## Adım 6 — Kamera Doğrulama

```bash
python deployment/hw_smoke_test.py --camera-only --cam-index 0 --cam-duration 5
```

**Beklenen:** `[OK] Kamera frame okundu` + > 30 frame, çözünürlük (H, W, 3).

Birden fazla kamera varsa `--cam-index 1` ile dene.

---

## Adım 7 — Tam Sistem Smoke Test (Gerçek Donanım)

```bash
python deployment/hw_smoke_test.py --all --real --usb ft2232h --port COM3
```

Tüm bileşenler PASS verirse `config.yaml`'da mock flag'lerini kapat:

```yaml
radar:
  mock: false        # → false
gimbal:
  mock: false        # → false
```

---

## Adım 8 — Tam GUI Çalıştırma (60 sn Gözlem)

```bash
python main.py
```

**Gözlem kontrol listesi:**
- [ ] GUI açılıyor, hata diyaloğu yok
- [ ] Radar paneli "Bağlandı" durumunda (yeşil)
- [ ] Gimbal paneli aktif, MAVLink heartbeat alındı
- [ ] Kamera akışı görünüyor
- [ ] Tespit listesi güncelleniyor (en az 30 sn)
- [ ] `logs/celik_kubbe.log` hata içermiyor
- [ ] `logs/blackbox/tracks_*.csv` yeni dosya oluşturuldu

Tüm madde geçerse sistem **demo'ya hazır** sayılır.

---

## Hızlı Geri Dönüş (Rollback)

Sahada bir bileşen arızalanırsa o bileşeni mock'a alıp diğerleriyle devam et:

```yaml
radar:
  mock: true         # radar arızalı, mock fallback
gimbal:
  mock: false        # gimbal sağlam, gerçek
```

Bu hibrit mod test edilmiş ve desteklenir; `main.py` her bileşeni bağımsız initialize eder.
