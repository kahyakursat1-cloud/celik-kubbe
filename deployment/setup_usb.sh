#!/usr/bin/env bash
# =====================================================================
# AERIS-10 Radar (FTDI FT2232H / FT601) USB Yetki Kurulumu
# =====================================================================

if [ "$EUID" -ne 0 ]; then
  echo "Lütfen bu scripti root yetkisiyle (sudo) çalıştırın."
  exit 1
fi

echo "FTDI USB yetkileri ayarlanıyor..."

# pyftdi için root olmadan erişim kuralları
cat > /etc/udev/rules.d/99-ftdi.rules << EOF
# FTDI FT2232H (USB 2.0)
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="6010", GROUP="plugdev", MODE="0666"

# FTDI FT601 (USB 3.0)
SUBSYSTEM=="usb", ATTR{idVendor}=="0403", ATTR{idProduct}=="601f", GROUP="plugdev", MODE="0666"
EOF

# Udev kurallarını yeniden yükle
udevadm control --reload-rules
udevadm trigger

echo "Udev kuralları başarıyla eklendi."
echo "Radar köprüsü (pyftdi) artık root yetkisi olmadan çalışabilir."
