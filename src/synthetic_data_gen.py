"""
Çelik Kubbe — Sentetik Veri Üretici (Synthetic Dataset Generator)
Fiziksel donanım (Radar veya Drone) olmadığında yapay zeka modelini eğitmek ve
sensör füzyon algoritmalarını test etmek için eşzamanlı (paired) veri üretir.

Ürettiği Veriler:
1. Sentetik Kamera Görüntüsü (İçinde hareket eden obje)
2. YOLOv11 Etiket Dosyası (.txt)
3. AERIS-10 Radar CFAR Tespit Verisi (.h5)
"""

import os
import cv2
import math
import time
import json
import numpy as np
import h5py

def uret_sentetik_veri(adet=100, cikti_dizini="../data/synthetic"):
    os.makedirs(cikti_dizini, exist_ok=True)
    images_dir = os.path.join(cikti_dizini, "images")
    labels_dir = os.path.join(cikti_dizini, "labels")
    radar_dir = os.path.join(cikti_dizini, "radar")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(radar_dir, exist_ok=True)

    print(f"{adet} adet sentetik eşzamanlı veri üretiliyor...")

    # Kamera parametreleri
    img_w, img_h = 640, 480
    hfov_deg = 120.0
    
    # Radar parametreleri
    num_range_bins = 64
    num_doppler_bins = 32
    max_range_km = 3.0
    range_bin_size = max_range_km / num_range_bins

    for i in range(adet):
        # 1. Rastgele Drone Hedefi Oluştur
        # Kamera merkezinden x ekseninde açısal pozisyon
        bearing_deg = np.random.uniform(-45.0, 45.0)
        # Radar mesafesi
        distance_km = np.random.uniform(0.5, 2.5)
        # Hız
        velocity_ms = np.random.uniform(-40.0, -10.0)  # Yaklaşan drone

        # 2. Kamera Bounding Box Hesaplama
        # Bearing açısını kamera pikseline dönüştür
        cx_norm = (bearing_deg / hfov_deg) + 0.5
        cx_pixel = int(cx_norm * img_w)
        cy_pixel = int(img_h * 0.4)  # Gökyüzü hizası
        
        # Uzaklığa göre büyüklük (yakınsa büyük)
        w_norm = max(0.02, 0.1 - (distance_km * 0.03))
        h_norm = w_norm * 0.6
        w_pixel = int(w_norm * img_w)
        h_pixel = int(h_norm * img_h)

        # 3. Görüntü Üret (Gökyüzü arkaplanı ve drone)
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        img[:] = (235, 206, 135)  # Mavi gökyüzü (BGR)
        # Drone çizimi (Koyu gri dörtgen)
        pt1 = (cx_pixel - w_pixel//2, cy_pixel - h_pixel//2)
        pt2 = (cx_pixel + w_pixel//2, cy_pixel + h_pixel//2)
        cv2.rectangle(img, pt1, pt2, (50, 50, 50), -1)
        
        # 4. Dosyaları Kaydet (Görüntü ve Label)
        img_filename = f"synth_{i:04d}.jpg"
        cv2.imwrite(os.path.join(images_dir, img_filename), img)
        
        label_filename = f"synth_{i:04d}.txt"
        with open(os.path.join(labels_dir, label_filename), "w") as f:
            # Sınıf 0 (Drone), cx, cy, w, h
            f.write(f"0 {cx_norm:.4f} {cy_pixel/img_h:.4f} {w_norm:.4f} {h_norm:.4f}\n")

        # 5. Radar Verisi Üret (HDF5 formatında)
        # Hangi binlere düşüyor?
        rbin = int(distance_km / range_bin_size)
        rbin = min(num_range_bins - 1, max(0, rbin))
        
        # velocity_resolution_ms yaklaşık 0.75 m/s (10.5 GHz, 5 kHz PRF, 32 bin)
        v_res = 0.75
        dbin = int(velocity_ms / v_res) + (num_doppler_bins // 2)
        dbin = min(num_doppler_bins - 1, max(0, dbin))

        detections = np.zeros((num_range_bins, num_doppler_bins), dtype=np.uint8)
        detections[rbin, dbin] = 1
        
        magnitude = np.random.uniform(10, 50, (num_range_bins, num_doppler_bins)).astype(np.float64)
        magnitude[rbin, dbin] = 5000.0  # Güçlü yankı

        h5_filename = f"synth_{i:04d}.h5"
        with h5py.File(os.path.join(radar_dir, h5_filename), "w") as h5f:
            h5f.attrs["timestamp"] = time.time()
            h5f.attrs["bearing_deg"] = bearing_deg
            h5f.attrs["distance_km"] = distance_km
            h5f.create_dataset("detections", data=detections, compression="gzip")
            h5f.create_dataset("magnitude", data=magnitude, compression="gzip")

    print(f"Başarılı! Çıktı dizini: {os.path.abspath(cikti_dizini)}")

if __name__ == "__main__":
    uret_sentetik_veri(50)
