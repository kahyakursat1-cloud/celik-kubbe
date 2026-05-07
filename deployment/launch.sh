#!/usr/bin/env bash
# =====================================================================
# ÇELİK KUBBE — Başlatma Scripti (Jetson Nano / Edge Device)
# =====================================================================

# Hata durumunda scripti durdur
set -e

# Proje dizinini bul
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo " ÇELİK KUBBE SİSTEMİ BAŞLATILIYOR"
echo "=========================================="
echo "Dizin: $PROJECT_DIR"

# 1. Sanal ortamı etkinleştir (varsa)
if [ -d ".venv" ]; then
    echo "[1/4] Sanal ortam etkinleştiriliyor..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "[1/4] Sanal ortam etkinleştiriliyor..."
    source venv/bin/activate
else
    echo "[1/4] Uyarı: Sanal ortam bulunamadı, sistem Python kullanılacak."
fi

# 2. USB Yetkileri (FTDI / Radar için)
echo "[2/4] USB donanım yetkileri kontrol ediliyor..."
# udev rules yüklenmemişse uyarı ver (root gerektirir)
if [ ! -f "/etc/udev/rules.d/99-ftdi.rules" ]; then
    echo "  -> FTDI udev kuralları eksik olabilir. Radar bağlantısında hata alırsanız 'deployment/setup_usb.sh' çalıştırın."
fi

# 3. Model formatı kontrolü (TensorRT)
echo "[3/4] YOLO modeli kontrol ediliyor..."
MODEL_PT="models/yolo11m_celikkubbe.pt"
MODEL_TRT="models/yolo11m_celikkubbe.engine"

if [ -f "$MODEL_TRT" ]; then
    echo "  -> TensorRT optimize edilmiş model bulundu: $MODEL_TRT"
    # config.yaml içinde model yolunu trt olarak güncellemek gerekebilir
elif [ -f "$MODEL_PT" ]; then
    echo "  -> PyTorch modeli bulundu ($MODEL_PT). Daha yüksek performans için TensorRT'ye dönüştürmeniz önerilir."
else
    echo "  -> DİKKAT: Model dosyası bulunamadı. Sadece mock radar verisiyle çalışacak."
fi

# 4. Uygulamayı başlat
echo "[4/4] Çelik Kubbe GUI başlatılıyor..."
export QT_QPA_PLATFORM=xcb  # Jetson / Linux display için
python3 main.py

echo "Çelik Kubbe kapatıldı."
