"""
Blackbox Logger (Kara Kutu Veri Kaydedici)
Çelik Kubbe'nin operasyonel uçuş izlerini, angajman kararlarını ve sistem durumunu
görev sonrası analiz (Post-Mission Analysis) için kaydeder.
"""

import os
import csv
import time
import logging
from datetime import datetime
from PySide6.QtCore import QObject, Signal, QThread
import queue

logger = logging.getLogger("blackbox")

class BlackboxLogger(QThread):
    def __init__(self, log_dir="logs/blackbox"):
        super().__init__()
        self._log_dir = log_dir
        self._queue = queue.Queue()
        self._dur = False
        
        os.makedirs(self._log_dir, exist_ok=True)
        
        # Güncel görev için dosya adları oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._track_file = os.path.join(self._log_dir, f"tracks_{timestamp}.csv")
        self._event_file = os.path.join(self._log_dir, f"events_{timestamp}.csv")
        
        # Dosya başlıklarını yaz
        self._init_csv_headers()

    def _init_csv_headers(self):
        with open(self._track_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Threat_ID", "Class", "Threat_Level", "Range_km", "Bearing_deg", "Velocity_ms", "Altitude_m", "Source", "Status"])
            
        with open(self._event_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Event_Type", "Target_ID", "Description"])

    def log_tehdit(self, thr):
        """Tehdidin anlık durumunu kaydeder."""
        try:
            durum = "Engaged" if getattr(thr, 'engaged', False) else "Active"
            data = {
                "type": "track",
                "timestamp": datetime.now().isoformat(),
                "id": thr.id,
                "sinif": getattr(thr, 'sinif', 'Bilinmeyen'),
                "level": thr.threat_level,
                "range": round(thr.range_km(), 3),
                "bearing": round(thr.bearing(), 2),
                "velocity": round(getattr(thr, 'velocity_ms', 0.0), 2),
                "altitude": getattr(thr, 'altitude', 0),
                "source": getattr(thr, 'kaynak', 'Bilinmeyen'),
                "status": durum
            }
            self._queue.put(data)
        except Exception as e:
            logger.error(f"Blackbox threat log hatası: {e}")

    def log_olay(self, event_type: str, target_id: str, description: str):
        """Sistem olaylarını (Angajman, Hedef Seçimi vb.) kaydeder."""
        data = {
            "type": "event",
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "target_id": target_id,
            "description": description
        }
        self._queue.put(data)

    def run(self):
        logger.info(f"Kara Kutu kaydı başladı: {self._track_file}")
        while not self._dur:
            try:
                # 1 saniye timeout ile kuyruktan veri bekle
                data = self._queue.get(timeout=1.0)
                
                if data["type"] == "track":
                    with open(self._track_file, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            data["timestamp"], data["id"], data["sinif"], data["level"],
                            data["range"], data["bearing"], data["velocity"], data["altitude"],
                            data["source"], data["status"]
                        ])
                elif data["type"] == "event":
                    with open(self._event_file, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            data["timestamp"], data["event_type"], data["target_id"], data["description"]
                        ])
                        
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Blackbox yazma hatası: {e}")

    def durdur(self):
        self._dur = True
        self.wait(2000)
        logger.info("Kara Kutu kaydı durduruldu.")
