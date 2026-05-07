# Çelik Kubbe — Sistem Mimarisi

Bu belge, ana modüller arası veri akışını ve sorumluluk bölünmesini özetler. Ayrıntılı algoritma detayları için [SENSOR_FUSION.md](SENSOR_FUSION.md) belgesine bakın.

## Üst Seviye Akış

```mermaid
flowchart LR
    subgraph Sensorler[Sensorler]
        CAM[Kamera<br/>OpenCV VideoCapture]
        RAD[AERIS-10 Radar<br/>FT2232H/FT601 USB]
    end

    subgraph Algilama[Algilama Katmani]
        YOLO[tespit_pipeline.py<br/>YOLOv11m + ByteTrack]
        RB[radar_bridge.py<br/>RadarBridge - QThread]
    end

    subgraph Fuzyon[Fuzyon ve Karar]
        SF[sensor_fusion.py<br/>SensorFusion + KalmanFilter2D]
        WTA[wta_optimizer.py<br/>WTAOptimizer]
    end

    subgraph Cikti[Cikti ve Aktuator]
        GUI[main.py<br/>CelikKubbeGUI]
        GMB[gimbal_controller.py<br/>MAVLink]
        BB[blackbox_logger.py<br/>CSV kayit]
    end

    CAM -->|frame| YOLO
    RAD -->|RadarFrame| RB

    YOLO -->|tespit_sinyal| SF
    RB -->|radar_tespit_sinyal| SF

    SF -->|fuzyon_sinyal<br/>FusedTrack listesi| GUI
    SF -->|tehdit listesi| WTA

    WTA -->|atama| GUI
    GUI -->|hedef koordinati| GMB
    GUI --> BB
    SF --> BB

    BB -->|tracks_*.csv<br/>events_*.csv| Disk[(logs/blackbox/)]
```

## Modul Sorumluluklari

| Modul | Tip | Sorumluluk |
|-------|-----|------------|
| `tespit_pipeline.TespitPipeline` | QThread | YOLOv11 inference + ByteTrack — kameradan tespit_sinyal yayin |
| `radar_bridge.RadarBridge` | QThread | USB FTDI baglanti, RadarFrame al, CFAR tespit cikarma, radar_tespit_sinyal yayin |
| `sensor_fusion.SensorFusion` | QObject | Radar+Kamera tespitlerini esleştir, FusedTrack olustur, Kalman ile filtrele, XAI tehdit skoru |
| `kalman_filter.KalmanFilter2D` | yardimci | 2D lineer Kalman filter, state=[x,y,vx,vy] |
| `wta_optimizer.WTAOptimizer` | static | Tehdit-batarya atama optimizasyonu |
| `gimbal_controller.GimbalController` | QObject | MAVLink ile pan/tilt, mock fallback |
| `blackbox_logger.BlackboxLogger` | QThread | tracks/events CSV |
| `main.CelikKubbeGUI` | QMainWindow | UI, sinyal dispatchi, kullanici kontrolu |

## Sinyal-Slot Topolojisi

```mermaid
sequenceDiagram
    participant Cam as Kamera
    participant TP as TespitPipeline
    participant SF as SensorFusion
    participant RB as RadarBridge
    participant GUI as CelikKubbeGUI
    participant GMB as GimbalController
    participant BB as BlackboxLogger

    Cam->>TP: frame
    TP->>SF: kamera_girdisi (tespit_sinyal)
    RB->>SF: radar_girdisi (radar_tespit_sinyal)
    SF->>GUI: fuzyon_sinyal (FusedTrack listesi)
    GUI->>GMB: hedefe_yonel(tid, bearing, distance, alt)
    GUI->>BB: log_tehdit / log_olay
    GMB->>GUI: gimbal_durum_sinyal
```

## Calistirma Modu

`config.yaml` icindeki bayraklar:

| Bayrak | Mock anlami | Gercek anlami |
|--------|-------------|---------------|
| `radar.mock` | radar_protocol.FT2232HConnection(mock=True) — sentetik veri | Gercek FTDI USB acilir |
| `gimbal.mock` | komutlar yalnizca sinyal yayinlar, donanima gitmez | pymavlink.mavutil baglantisi |
| `radar.aktif` | RadarBridge baslatilmaz | RadarBridge baslatilir |
| `fuzyon.aktif` | SensorFusion devre disi | SensorFusion baglanir |

Hibrit mod desteklenir (orn. `radar.mock=True, gimbal.mock=False`) — modullerin baglanti adimlari bagimsizdir.

## Kara Kutu Veri Bicimi

`logs/blackbox/tracks_YYYYMMDD_HHMMSS.csv` sutunlari:
```
Timestamp, Threat_ID, Class, Threat_Level, Range_km, Bearing_deg, Velocity_ms, Altitude_m, Source, Status
```

`Source` alani: `fuzyon` | `yalniz_radar` | `yalniz_kamera`. Gorev sonrasi analiz icin kritik metadata.
