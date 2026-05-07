import sys
import os
import math
import time
import random
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QFrame,
    QHeaderView, QSizePolicy, QGridLayout, QProgressBar
)
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QLinearGradient,
    QRadialGradient, QPainterPath, QPolygonF, QPalette
)

# ── Config & Logging ───────────────────────────────────────────────────────────
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from konfig import cfg as _app_cfg
    import logging, logging.handlers
    _lc = _app_cfg.get("loglama", {})
    _fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)-20s — %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    _root = logging.getLogger()
    _root.setLevel(getattr(logging, _lc.get("seviye", "INFO").upper(), logging.INFO))
    _root.handlers.clear()
    _ch = logging.StreamHandler(sys.stdout); _ch.setFormatter(_fmt); _root.addHandler(_ch)
    _lf = _lc.get("dosya")
    if _lf:
        os.makedirs(os.path.dirname(_lf), exist_ok=True)
        _fh = logging.handlers.RotatingFileHandler(
            _lf, maxBytes=_lc.get("max_bayt", 5_242_880),
            backupCount=_lc.get("yedek_sayisi", 3), encoding="utf-8")
        _fh.setFormatter(_fmt); _root.addHandler(_fh)
except Exception:
    import logging; logging.basicConfig(level=logging.INFO)

from src.battery_profiles import profile_for_battery
from src.coordinate_utils import display_radius_to_km, polar_to_display_xy

# ── Styling constants ──────────────────────────────────────────────────────────
BG_DARK   = "#0f1419"
BG_MID    = "#1a2332"
BG_PANEL  = "#111d27"
ORANGE    = "#ff6600"
ORANGE_DIM = "#803300"
GREEN_RDR = "#00ff00"
GREEN_DIM = "#004400"
BLUE_FRN  = "#3399ff"
RED_THR   = "#ff2222"
YELLOW_EZ = "#ffcc00"
TEXT_MAIN = "#e8e8e8"
TEXT_DIM  = "#7a8a9a"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_MAIN};
    font-family: 'Inter', "Segoe UI", "Consolas", monospace;
}}
QFrame#panel {{
    background-color: rgba(17, 29, 39, 0.75);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 12px;
}}
QLabel#title {{
    color: {ORANGE};
    font-size: 20px;
    font-weight: 800;
    letter-spacing: 3px;
}}
QLabel#alert {{
    color: {RED_THR};
    font-size: 14px;
    font-weight: 900;
    background: rgba(255, 34, 34, 0.15);
    padding: 2px 8px;
    border-radius: 4px;
}}
QLabel#status_ok {{
    color: {GREEN_RDR};
    font-size: 13px;
    font-weight: bold;
}}
QLabel#stat_val {{
    color: #38bdf8;
    font-size: 24px;
    font-weight: 800;
}}
QLabel#stat_lbl {{
    color: #94a3b8;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
}}
QLabel#section_hdr {{
    color: #38bdf8;
    font-size: 12px;
    font-weight: 800;
    border-bottom: 2px solid rgba(56, 189, 248, 0.2);
    padding-bottom: 4px;
    letter-spacing: 1px;
}}
QPushButton {{
    background-color: rgba(26, 35, 50, 0.8);
    color: #f8fafc;
    border: 1px solid rgba(56, 189, 248, 0.4);
    border-radius: 8px;
    padding: 8px 16px;
    font-family: "Inter", "Segoe UI", sans-serif;
    font-size: 13px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: rgba(56, 189, 248, 0.2);
    border: 1px solid #38bdf8;
}}
QPushButton#btn_engage {{
    background-color: rgba(153, 27, 27, 0.8);
    color: #fca5a5;
    border: 1px solid #ef4444;
    font-weight: 900;
    font-size: 14px;
    letter-spacing: 2px;
}}
QPushButton#btn_engage:hover {{
    background-color: #dc2626;
    color: white;
}}
QPushButton#btn_defend {{
    background-color: rgba(30, 64, 175, 0.8);
    color: #93c5fd;
    border: 1px solid #3b82f6;
    font-weight: bold;
    letter-spacing: 1px;
}}
QTableWidget {{
    background-color: transparent;
    color: #e2e8f0;
    border: 1px solid rgba(56, 189, 248, 0.2);
    gridline-color: rgba(56, 189, 248, 0.1);
    font-size: 12px;
    border-radius: 8px;
}}
QTableWidget::item:selected {{
    background-color: rgba(56, 189, 248, 0.3);
    color: white;
}}
QHeaderView::section {{
    background-color: rgba(15, 23, 42, 0.9);
    color: #38bdf8;
    border: none;
    border-right: 1px solid rgba(56, 189, 248, 0.2);
    border-bottom: 1px solid rgba(56, 189, 248, 0.2);
    padding: 6px;
    font-size: 11px;
    font-weight: 800;
}}
QProgressBar {{
    background-color: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 4px;
    height: 12px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0, stop: 0 #10b981, stop: 1 #34d399);
    border-radius: 3px;
}}
"""


# ── Data models ────────────────────────────────────────────────────────────────
class Threat:
    _id_counter = 1

    def __init__(self):
        self.id = f"THR-{Threat._id_counter:03d}"
        Threat._id_counter += 1
        angle = random.uniform(0, 360)
        dist  = random.uniform(0.6, 0.95)
        self.x = dist * math.cos(math.radians(angle))
        self.y = dist * math.sin(math.radians(angle))
        self.vx = random.uniform(-0.003, 0.003)
        self.vy = random.uniform(-0.003, 0.003)
        self.altitude = random.randint(500, 12000)
        self.threat_level = random.choice(["DÜŞÜK", "ORTA", "YÜKSEK", "KRİTİK"])
        self.engaged = False
        self.fade   = 1.0
        # ── Radar / Füzyon alanları ──
        self.velocity_ms = 0.0       # Radyal hız (m/s, negatif=yaklaşıyor)
        self.snr_db = 0.0            # Sinyal-gürültü oranı (dB)
        self.sinif = "Bilinmeyen"    # YOLO sınıfı
        self.kaynak = "simülasyon"   # "fuzyon", "yalniz_radar", "yalniz_kamera", "simülasyon"
        self.fusion_track_id = -1    # SensorFusion iz ID'si
        self.physical_range_km: float | None = None
        self.tehdit_skoru = 0.0

    def range_km(self):
        if self.physical_range_km is not None:
            return self.physical_range_km
        return math.sqrt(self.x ** 2 + self.y ** 2) * 200

    def bearing(self):
        b = math.degrees(math.atan2(self.y, self.x))
        return (b + 360) % 360

    def update(self):
        if self.engaged:
            self.fade = max(0.0, self.fade - 0.05)
            return
        self.x += self.vx
        self.y += self.vy
        # Move toward center
        d = math.sqrt(self.x ** 2 + self.y ** 2)
        if d > 0:
            self.x -= self.x / d * 0.0005
            self.y -= self.y / d * 0.0005


class FriendlyAsset:
    def __init__(self):
        angle = random.uniform(0, 360)
        dist  = random.uniform(0.1, 0.45)
        self.x = dist * math.cos(math.radians(angle))
        self.y = dist * math.sin(math.radians(angle))


# ── Radar Widget ───────────────────────────────────────────────────────────────
class RadarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(500, 500)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sweep_angle = 0.0
        self.trail_angles: list[float] = []
        self.threats: list[Threat] = []
        self.friendlies: list[FriendlyAsset] = [FriendlyAsset() for _ in range(3)]
        self.selected_threat: str | None = None

    def set_threats(self, threats: list[Threat]):
        self.threats = threats

    def set_selected(self, tid: str | None):
        self.selected_threat = tid

    def advance_sweep(self, delta_deg: float):
        self.sweep_angle = (self.sweep_angle + delta_deg) % 360
        self.trail_angles.insert(0, self.sweep_angle)
        if len(self.trail_angles) > 60:
            self.trail_angles = self.trail_angles[:60]
        self.update()

    # ── paint ──────────────────────────────────────────────────────────────────
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        radius = min(cx, cy) - 10

        # Background
        painter.fillRect(0, 0, w, h, QColor(BG_PANEL))

        # Radial gradient background for scope
        grad = QRadialGradient(cx, cy, radius)
        grad.setColorAt(0,   QColor(0, 20, 0, 180))
        grad.setColorAt(0.7, QColor(0, 10, 0, 200))
        grad.setColorAt(1,   QColor(0, 0, 0, 230))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QRectF(cx - radius, cy - radius, radius * 2, radius * 2))

        # Range rings
        ring_ranges = [50, 100, 150, 200]
        for km in ring_ranges:
            r = km / 200 * radius
            pen = QPen(QColor(0, 100, 0, 120), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))
            # Label
            painter.setPen(QPen(QColor(0, 160, 0, 200)))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(int(cx + r + 3), int(cy - 3), f"{km}km")

        # Azimuth lines (every 30°)
        for deg in range(0, 360, 30):
            rad = math.radians(deg)
            x2 = cx + radius * math.cos(rad)
            y2 = cy + radius * math.sin(rad)
            painter.setPen(QPen(QColor(0, 80, 0, 100), 1))
            painter.drawLine(int(cx), int(cy), int(x2), int(y2))
            # Degree label
            lx = cx + (radius + 14) * math.cos(rad)
            ly = cy + (radius + 14) * math.sin(rad)
            painter.setPen(QPen(QColor(0, 140, 0, 200)))
            painter.setFont(QFont("Consolas", 7))
            painter.drawText(int(lx - 10), int(ly + 4), f"{deg}°")

        # Sweep trail
        for i, angle in enumerate(self.trail_angles):
            alpha = int(120 * (1 - i / len(self.trail_angles)))
            rad = math.radians(angle)
            x2 = cx + radius * math.cos(rad)
            y2 = cy + radius * math.sin(rad)
            painter.setPen(QPen(QColor(0, 255, 0, alpha), 2))
            painter.drawLine(int(cx), int(cy), int(x2), int(y2))

        # Main sweep line
        rad = math.radians(self.sweep_angle)
        x2 = cx + radius * math.cos(rad)
        y2 = cy + radius * math.sin(rad)
        painter.setPen(QPen(QColor(0, 255, 0, 255), 2))
        painter.drawLine(int(cx), int(cy), int(x2), int(y2))

        # Engagement zones (yellow arcs)
        ez_color = QColor(255, 204, 0, 50)
        painter.setBrush(QBrush(ez_color))
        painter.setPen(QPen(QColor(255, 204, 0, 150), 1))
        for arc_start, arc_span in [(30, 60), (150, 50), (250, 70)]:
            r_ez = 0.5 * radius
            painter.drawPie(
                QRectF(cx - r_ez, cy - r_ez, r_ez * 2, r_ez * 2),
                int(arc_start * 16), int(arc_span * 16)
            )

        # Friendly assets
        for fa in self.friendlies:
            fx = cx + fa.x * radius
            fy = cy + fa.y * radius
            painter.setPen(QPen(QColor(BLUE_FRN), 2))
            painter.setBrush(QBrush(QColor(0, 60, 130, 180)))
            painter.drawEllipse(QRectF(fx - 7, fy - 7, 14, 14))
            painter.setPen(QPen(QColor(BLUE_FRN)))
            painter.setFont(QFont("Consolas", 7, QFont.Bold))
            painter.drawText(int(fx - 4), int(fy + 4), "F")

        # Threats — kaynak ve sınıfa göre farklı ikon/renk
        for thr in self.threats:
            if thr.fade <= 0:
                continue
            tx = cx + thr.x * radius
            ty = cy + thr.y * radius
            alpha = int(255 * thr.fade)

            # Kaynak tipine göre renk
            kaynak = getattr(thr, 'kaynak', 'simülasyon')
            if kaynak == "fuzyon":
                col = QColor(0, 200, 255, alpha)       # Cyan — füzyon
                fill = QColor(0, 150, 200, 60)
            elif kaynak == "yalniz_radar":
                col = QColor(0, 255, 100, alpha)       # Yeşil — yalnız radar
                fill = QColor(0, 180, 60, 60)
            elif kaynak == "yalniz_kamera":
                col = QColor(255, 200, 0, alpha)       # Sarı — yalnız kamera
                fill = QColor(200, 160, 0, 60)
            else:
                col = QColor(255, 34, 34, alpha)       # Kırmızı — simülasyon
                fill = QColor(180, 0, 0, 60)

            # Seçili hedef vurgusu
            if thr.id == self.selected_threat:
                fill = QColor(255, 100, 0, 120)

            # Tehdit seviyesine göre ikon boyutu
            threat_lvl = getattr(thr, 'threat_level', 'DÜŞÜK')
            sz = 12 if threat_lvl in ('YÜKSEK', 'KRİTİK') else 9

            # İkon çizimi — kaynak tipine göre farklı şekil
            painter.setPen(QPen(col, 2))
            painter.setBrush(QBrush(fill))
            if kaynak == "fuzyon":
                # Elmas (◇) — füzyon tespiti
                diamond = QPolygonF([
                    QPointF(tx, ty - sz),
                    QPointF(tx + sz * 0.7, ty),
                    QPointF(tx, ty + sz),
                    QPointF(tx - sz * 0.7, ty),
                ])
                painter.drawPolygon(diamond)
            elif kaynak == "yalniz_radar":
                # Kare (□) — yalnız radar
                painter.drawRect(QRectF(tx - sz * 0.6, ty - sz * 0.6, sz * 1.2, sz * 1.2))
            else:
                # Üçgen (△) — kamera veya simülasyon
                tri = QPolygonF([
                    QPointF(tx, ty - sz),
                    QPointF(tx - sz * 0.8, ty + sz * 0.7),
                    QPointF(tx + sz * 0.8, ty + sz * 0.7),
                ])
                painter.drawPolygon(tri)

            # KRİTİK tehditte yanıp sönen halka
            if threat_lvl == 'KRİTİK':
                pulse = int(abs(math.sin(time.time() * 4)) * 180)
                painter.setPen(QPen(QColor(255, 0, 0, pulse), 2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QRectF(tx - sz - 4, ty - sz - 4, (sz + 4) * 2, (sz + 4) * 2))

            # Hız vektörü çizgisi (yaklaşıyor ise)
            v_ms = getattr(thr, 'velocity_ms', 0.0)
            if v_ms < -5.0:
                v_len = min(30, abs(v_ms) * 0.3)
                d = math.sqrt(thr.x ** 2 + thr.y ** 2)
                if d > 0.01:
                    vx_dir = -thr.x / d * v_len
                    vy_dir = -thr.y / d * v_len
                    painter.setPen(QPen(QColor(255, 100, 0, alpha), 1, Qt.DashLine))
                    painter.drawLine(int(tx), int(ty), int(tx + vx_dir), int(ty + vy_dir))

            # Etiket: ID + sınıf
            sinif = getattr(thr, 'sinif', 'Bilinmeyen')
            lbl_text = thr.id[-3:]
            if sinif != 'Bilinmeyen':
                lbl_text += f" {sinif[:6]}"
            painter.setPen(QPen(col))
            painter.setFont(QFont("Consolas", 7))
            painter.drawText(int(tx + sz + 3), int(ty + 3), lbl_text)

        # Radar border ring
        painter.setPen(QPen(QColor(0, 180, 0, 200), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QRectF(cx - radius, cy - radius, radius * 2, radius * 2))

        # Center KUBBE star
        self._draw_star(painter, cx, cy, 12, 6)

        # Corner decorations
        painter.setPen(QPen(QColor(ORANGE), 2))
        for dx, dy, a1, a2 in [(-1, -1, 180, 90), (1, -1, 270, 90), (-1, 1, 90, 90), (1, 1, 0, 90)]:
            ox = cx + dx * (radius + 6)
            oy = cy + dy * (radius + 6)
            painter.drawLine(int(ox), int(oy), int(ox + dx * (-12)), int(oy))
            painter.drawLine(int(ox), int(oy), int(ox), int(oy + dy * (-12)))

    def _draw_star(self, painter: QPainter, cx: float, cy: float, r_outer: float, points: int):
        r_inner = r_outer * 0.45
        poly = QPolygonF()
        for i in range(points * 2):
            angle = math.radians(i * 180 / points - 90)
            r = r_outer if i % 2 == 0 else r_inner
            poly.append(QPointF(cx + r * math.cos(angle), cy + r * math.sin(angle)))
        painter.setPen(QPen(QColor(ORANGE), 2))
        painter.setBrush(QBrush(QColor(255, 120, 0, 200)))
        painter.drawPolygon(poly)


# ── Battery Status Widget ──────────────────────────────────────────────────────
class BatteryStatusWidget(QWidget):
    def __init__(self, name: str, ammo: int, parent=None):
        super().__init__(parent)
        self.name = name
        self.ammo = ammo
        self.max_ammo = ammo
        self.setFixedHeight(58)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(4)
        hdr = QHBoxLayout()
        hdr.setSpacing(8)
        self.lbl_name = QLabel(name)
        self.lbl_name.setStyleSheet(f"color: {ORANGE}; font-weight: bold; font-size: 11px;")
        self.lbl_name.setMinimumWidth(105)
        self.lbl_ammo = QLabel(f"{ammo}/{self.max_ammo}")
        self.lbl_ammo.setStyleSheet(f"color: {GREEN_RDR}; font-size: 11px;")
        self.lbl_ammo.setFixedWidth(48)
        self.lbl_ammo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hdr.addWidget(self.lbl_name)
        hdr.addStretch()
        hdr.addWidget(self.lbl_ammo)
        self.bar = QProgressBar()
        self.bar.setMaximum(self.max_ammo)
        self.bar.setValue(ammo)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        layout.addLayout(hdr)
        layout.addWidget(self.bar)

    def fire(self):
        if self.ammo > 0:
            self.ammo -= 1
            self.lbl_ammo.setText(f"{self.ammo}/{self.max_ammo}")
            self.bar.setValue(self.ammo)
            if self.ammo == 0:
                self.bar.setStyleSheet("QProgressBar::chunk { background-color: #550000; }")


# ── Main Window ────────────────────────────────────────────────────────────────
class CelikKubbeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ÇELİK KUBBE - HAVA SAVUNMA SİSTEMİ")
        self.setMinimumSize(1200, 780)
        self.setStyleSheet(STYLESHEET)

        self.threats: list[Threat] = []
        self.intercepted_count = 0
        self.selected_tid: str | None = None
        self._spawn_initial_threats()

        central = QWidget()
        self.setCentralWidget(central)
        main_vbox = QVBoxLayout(central)
        main_vbox.setContentsMargins(8, 6, 8, 6)
        main_vbox.setSpacing(6)

        # Title bar
        main_vbox.addWidget(self._build_title_bar())
        # Status bar
        main_vbox.addWidget(self._build_status_bar())

        # Content
        content_hbox = QHBoxLayout()
        content_hbox.setSpacing(6)

        # Radar (center dominant)
        self.radar = RadarWidget()
        self.radar.set_threats(self.threats)
        radar_frame = QFrame()
        radar_frame.setObjectName("panel")
        radar_fl = QVBoxLayout(radar_frame)
        radar_fl.setContentsMargins(4, 4, 4, 4)
        radar_lbl = QLabel("● RADAR TARAMA - AKTİF")
        radar_lbl.setStyleSheet(f"color: {GREEN_RDR}; font-size: 10px; font-weight: bold;")
        radar_fl.addWidget(radar_lbl)
        radar_fl.addWidget(self.radar, stretch=1)
        content_hbox.addWidget(radar_frame, stretch=3)

        # Right panel
        content_hbox.addWidget(self._build_right_panel(), stretch=0)
        main_vbox.addLayout(content_hbox, stretch=1)

        # Bottom panel
        main_vbox.addWidget(self._build_bottom_panel())

        # Timers
        self.sweep_timer = QTimer()
        self.sweep_timer.timeout.connect(self._on_sweep)
        self.sweep_timer.start(22)  # ~45fps, 360° in 4s → 2° per tick at 45fps

        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self._on_data_update)
        self.data_timer.start(500)

        self.threat_timer = QTimer()
        self.threat_timer.timeout.connect(self._on_spawn_threat)
        self.threat_timer.start(4000)

        # ── Kamera + YOLOv11 pipeline ─────────────────────────────────────────
        self._kamera_tespitler: dict[int, Threat] = {}  # track_id → Threat
        self._cap = None
        try:
            import cv2 as _cv2
            _cap = _cv2.VideoCapture(0)
            if _cap.isOpened():
                self._cap = _cap
        except Exception:
            pass

        self._pipeline = None
        try:
            from src.tespit_pipeline import TespitPipeline
            _model = os.path.join(os.path.dirname(__file__), "models", "yolo11m_celikkubbe.pt")
            self._pipeline = TespitPipeline(
                model_yolu=_model if os.path.isfile(_model) else None,
            )
            self._pipeline.tespit_sinyal.connect(self._on_kamera_tespiti)
            self._pipeline.start()
        except Exception:
            pass

        # Kamera karesi gönderme zamanlayıcısı (~30 fps)
        if self._cap is not None:
            self._cam_timer = QTimer()
            self._cam_timer.timeout.connect(self._on_cam_frame)
            self._cam_timer.start(33)

        # ── AERIS-10 Radar Köprüsü ───────────────────────────────────────────
        self._radar_bridge = None
        self._sensor_fusion = None
        self._fuzyon_izleri: dict[int, Threat] = {}  # fusion_track_id → Threat

        _rcfg = _app_cfg.get("radar", {})
        _fcfg = _app_cfg.get("fuzyon", {})

        if _rcfg.get("aktif", False):
            try:
                from src.radar_bridge import RadarBridge, RadarParams
                _params = RadarParams(
                    center_freq_hz=_rcfg.get("merkez_frekans_ghz", 10.5) * 1e9,
                    bandwidth_hz=_rcfg.get("bant_genisligi_mhz", 200) * 1e6,
                    prf_hz=_rcfg.get("prf_hz", 5000),
                    max_range_km=_rcfg.get("maks_menzil_km", 3.0),
                    scan_rate_rpm=_rcfg.get("tarama_hizi_rpm", 15.0),
                )
                self._radar_bridge = RadarBridge(
                    mock=_rcfg.get("mock", True),
                    usb_type=_rcfg.get("usb_tipi", "ft2232h"),
                    kayit_aktif=_rcfg.get("kayit_aktif", False),
                    params=_params,
                )
                self._radar_bridge.radar_durum_sinyal.connect(self._on_radar_durum)
                self._radar_bridge.radar_baglanti_sinyal.connect(self._on_radar_baglanti)

                # Sensör Füzyon
                if _fcfg.get("aktif", False):
                    from src.sensor_fusion import SensorFusion
                    self._sensor_fusion = SensorFusion()
                    self._sensor_fusion.ACI_ESIGI_DEG = _fcfg.get("aci_esigi_deg", 15.0)
                    self._sensor_fusion.MESAFE_ESIGI_KM = _fcfg.get("mesafe_esigi_km", 1.0)
                    self._sensor_fusion.IZ_ZAMAN_ASIMI_S = _fcfg.get("zaman_asimi_s", 5.0)
                    self._sensor_fusion.fuzyon_sinyal.connect(self._on_fuzyon_tespiti)

                    # Radar → Füzyon
                    self._radar_bridge.radar_tespit_sinyal.connect(
                        self._sensor_fusion.radar_girdisi
                    )
                    # Kamera → Füzyon (pipeline varsa)
                    if self._pipeline is not None:
                        self._pipeline.tespit_sinyal.connect(
                            self._sensor_fusion.kamera_girdisi
                        )
                else:
                    # Füzyon kapalı — radar tespitlerini doğrudan kullan
                    self._radar_bridge.radar_tespit_sinyal.connect(
                        self._on_radar_dogrudan_tespit
                    )

                self._radar_bridge.start()
                logging.getLogger(__name__).info("AERIS-10 Radar köprüsü başlatıldı")
            except Exception as _re:
                logging.getLogger(__name__).warning(f"Radar köprüsü başlatılamadı: {_re}")

        # ── Gimbal / Takip Kontrolcüsü ─────────────────────────────────────────
        self._gimbal_controller = None
        _gcfg = _app_cfg.get("gimbal", {})
        if _gcfg.get("aktif", False):
            try:
                from src.gimbal_controller import GimbalController
                self._gimbal_controller = GimbalController(
                    mock=_gcfg.get("mock", True),
                    port=_gcfg.get("port", "COM3"),
                    baud=_gcfg.get("baud", 115200)
                )
                self._gimbal_controller.gimbal_durum_sinyal.connect(self._on_gimbal_durum)
                self._gimbal_controller.baslat()
            except Exception as _ge:
                logging.getLogger(__name__).warning(f"Gimbal başlatılamadı: {_ge}")

        # ── Blackbox (Kara Kutu) ──────────────────────────────────────────────
        self._blackbox = None
        try:
            from src.blackbox_logger import BlackboxLogger
            self._blackbox = BlackboxLogger()
            self._blackbox.start()
        except Exception as _be:
            logging.getLogger(__name__).warning(f"Kara Kutu başlatılamadı: {_be}")

        # ── Bilgi paneli sekmeleri ────────────────────────────────────────────
        try:
            import sys as _sys, os as _os
            _shared = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "shared")
            if _shared not in _sys.path:
                _sys.path.insert(0, _shared)
            from bilgi_paneli import sekme_ekle
            sekme_ekle(self, "celik_kubbe")
        except Exception as _bp_e:
            print(f"[BilgiPaneli] yüklenemedi: {_bp_e}")

    # ── UI Builders ─────────────────────────────────────────────────────────
    def _build_title_bar(self) -> QWidget:
        w = QWidget()
        w.setFixedHeight(50)
        w.setStyleSheet(f"background-color: {BG_MID}; border-bottom: 2px solid {ORANGE};")
        h = QHBoxLayout(w)
        h.setContentsMargins(12, 4, 12, 4)

        icon_lbl = QLabel("⬡")
        icon_lbl.setStyleSheet(f"color: {ORANGE}; font-size: 24px;")
        title_lbl = QLabel("ÇELİK KUBBE - HAVA SAVUNMA SİSTEMİ")
        title_lbl.setObjectName("title")

        self.alert_lbl = QLabel("● TEHDİT ALGILANDI")
        self.alert_lbl.setObjectName("alert")

        # AERIS-10 entegrasyon göstergesi
        self._aeris_badge = QLabel("📡 AERIS-10")
        self._aeris_badge.setStyleSheet(
            f"color: #00c8ff; font-size: 11px; font-weight: bold; "
            f"background: rgba(0,200,255,0.1); padding: 2px 8px; border-radius: 4px; "
            f"border: 1px solid rgba(0,200,255,0.3);"
        )

        ver_lbl = QLabel("v3.0.0 | AERIS-10 ENTEGRE | NATO SINIF-A")
        ver_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")

        h.addWidget(icon_lbl)
        h.addSpacing(8)
        h.addWidget(title_lbl)
        h.addSpacing(20)
        h.addWidget(self.alert_lbl)
        h.addStretch()
        h.addWidget(self._aeris_badge)
        h.addSpacing(12)
        h.addWidget(ver_lbl)
        return w

    def _build_status_bar(self) -> QWidget:
        w = QFrame()
        w.setObjectName("panel")
        w.setFixedHeight(56)
        h = QHBoxLayout(w)
        h.setContentsMargins(16, 6, 16, 6)
        h.setSpacing(30)

        stats = [
            ("TESPİT EDİLEN", "0", "detected"),
            ("ENGELLENENLERİ", "0", "intercepted"),
            ("AKTİF TEHDİT", "0", "active"),
            ("SİSTEM DURUMU", "AKTİF", "sys_status"),
            ("BÖLGE KAPSAMI", "98%", "coverage"),
        ]
        self._stat_labels = {}
        for lbl_text, val_text, key in stats:
            box = QVBoxLayout()
            box.setSpacing(0)
            val = QLabel(val_text)
            val.setObjectName("stat_val")
            val.setAlignment(Qt.AlignCenter)
            lbl = QLabel(lbl_text)
            lbl.setObjectName("stat_lbl")
            lbl.setAlignment(Qt.AlignCenter)
            box.addWidget(val)
            box.addWidget(lbl)
            self._stat_labels[key] = val
            h.addLayout(box)
            if stats.index((lbl_text, val_text, key)) < len(stats) - 1:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet(f"color: #1e3348;")
                h.addWidget(sep)
        return w

    def _build_right_panel(self) -> QWidget:
        w = QFrame()
        w.setObjectName("panel")
        w.setMinimumWidth(560)
        w.setMaximumWidth(640)
        w.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)

        columns = QHBoxLayout()
        columns.setSpacing(14)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        left_col.setSpacing(8)
        right_col.setSpacing(8)

        def section(parent_layout, text: str):
            lbl = QLabel(text)
            lbl.setObjectName("section_hdr")
            lbl.setStyleSheet(
                "color: #38bdf8; font-size: 13px; font-weight: 900; "
                "border-bottom: 2px solid rgba(56, 189, 248, 0.25); "
                "padding-bottom: 4px; letter-spacing: 1px;"
            )
            parent_layout.addWidget(lbl)
            return lbl

        def kv_row(parent_layout, key: str, value: str, color: str, key_width: int = 72):
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(key)
            lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
            lbl.setFixedWidth(key_width)
            vlbl = QLabel(value)
            vlbl.setStyleSheet(f"color: {color}; font-size: 12px; font-family: Consolas; font-weight: bold;")
            vlbl.setMinimumWidth(110)
            row.addWidget(lbl)
            row.addWidget(vlbl, stretch=1)
            parent_layout.addLayout(row)
            return vlbl

        def bar_row(parent_layout, key: str, value: int):
            row = QHBoxLayout()
            row.setSpacing(8)
            lbl = QLabel(key)
            lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
            lbl.setFixedWidth(82)
            bar = QProgressBar()
            bar.setValue(value)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            bar.setStyleSheet(f"""
                QProgressBar {{ background: {BG_MID}; border: 1px solid #1e3348; border-radius: 3px; }}
                QProgressBar::chunk {{ background-color: {GREEN_RDR}; border-radius: 3px; }}
            """)
            row.addWidget(lbl)
            row.addWidget(bar, stretch=1)
            parent_layout.addLayout(row)

        # Battery status header
        section(left_col, "▸ FÜZE PİL DURUMU")

        self.batteries: list[BatteryStatusWidget] = []
        battery_data = [
            ("PİL-ALFA", 12),
            ("PİL-BETA", 12),
            ("PİL-GAMMA", 8),
            ("PİL-DELTA", 8),
        ]
        battery_grid = QGridLayout()
        battery_grid.setHorizontalSpacing(8)
        battery_grid.setVerticalSpacing(6)
        for idx, (name, ammo) in enumerate(battery_data):
            bw = BatteryStatusWidget(name, ammo)
            battery_grid.addWidget(bw, idx // 2, idx % 2)
            self.batteries.append(bw)
        left_col.addLayout(battery_grid)

        left_col.addSpacing(4)

        # System health
        section(left_col, "▸ SİSTEM SAĞLIĞI")

        health_items = [
            ("RADAR", 100),
            ("İLETİŞİM", 98),
            ("ENERJİ", 94),
            ("YAZILIM", 100),
        ]
        for name, val in health_items:
            bar_row(left_col, name, val)

        # AERIS-10 Radar Durumu
        section(right_col, "▸ AERIS-10 RADAR")

        self._radar_status_lbl = QLabel("● BAĞLANTI BEKLENİYOR")
        self._radar_status_lbl.setStyleSheet(f"color: {YELLOW_EZ}; font-size: 12px; font-weight: bold;")
        right_col.addWidget(self._radar_status_lbl)

        radar_info = [
            ("FREKANS", "10.5 GHz"),
            ("MENZIL", "3 km"),
            ("USB", "FT2232H"),
            ("MOD", "MOCK"),
        ]
        self._radar_info_labels = {}
        for k, val in radar_info:
            vlbl = kv_row(right_col, k, val, GREEN_RDR)
            self._radar_info_labels[k] = vlbl

        # Sensör Füzyon Durumu
        self._fusion_status_lbl = QLabel("◉ FÜZYON: AKTİF")
        self._fusion_status_lbl.setStyleSheet("color: #00c8ff; font-size: 12px; font-weight: bold;")
        right_col.addWidget(self._fusion_status_lbl)
        
        right_col.addSpacing(4)

        # Gimbal Durumu
        section(right_col, "▸ GİMBAL / KAMERA")

        self._gimbal_status_lbl = QLabel("● BEKLİYOR")
        self._gimbal_status_lbl.setStyleSheet(f"color: {YELLOW_EZ}; font-size: 12px; font-weight: bold;")
        right_col.addWidget(self._gimbal_status_lbl)

        self._gimbal_info_labels = {}
        gimbal_info = [("PAN", "0.0°"), ("TILT", "0.0°"), ("HEDEF", "YOK")]
        for k, val in gimbal_info:
            vlbl = kv_row(right_col, k, val, ORANGE, key_width=62)
            self._gimbal_info_labels[k] = vlbl

        right_col.addSpacing(4)

        # Lejant
        section(left_col, "▸ İKON LEJANTI")

        legends = [
            ("◇", "#00c8ff", "Füzyon (Radar+Kam.)"),
            ("□", "#00ff64", "Yalnız Radar"),
            ("△", "#ffc800", "Yalnız Kamera"),
            ("△", "#ff2222", "Simülasyon"),
        ]
        legend_grid = QGridLayout()
        legend_grid.setHorizontalSpacing(12)
        legend_grid.setVerticalSpacing(4)
        for icon, color, desc in legends:
            row = QHBoxLayout()
            row.setSpacing(6)
            icon_lbl = QLabel(icon)
            icon_lbl.setStyleSheet(f"color: {color}; font-size: 15px; font-weight: bold;")
            icon_lbl.setFixedWidth(20)
            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
            desc_lbl.setWordWrap(False)
            row.addWidget(icon_lbl)
            row.addWidget(desc_lbl)
            cell = QWidget()
            cell.setLayout(row)
            legend_grid.addWidget(cell, legends.index((icon, color, desc)) // 2, legends.index((icon, color, desc)) % 2)
        left_col.addLayout(legend_grid)

        right_col.addSpacing(4)

        # Coordinates
        section(right_col, "▸ KONUM BİLGİSİ")

        coords = [
            ("LAT", "39°54'12\"N"),
            ("LON", "32°51'44\"E"),
            ("YÜK.", "1234m"),
        ]
        for k, val in coords:
            kv_row(right_col, k, val, GREEN_RDR, key_width=62)

        left_col.addStretch()
        right_col.addStretch()
        columns.addLayout(left_col, stretch=1)
        columns.addLayout(right_col, stretch=1)
        v.addLayout(columns)
        return w

    def _build_bottom_panel(self) -> QWidget:
        w = QFrame()
        w.setObjectName("panel")
        w.setFixedHeight(200)
        h = QHBoxLayout(w)
        h.setContentsMargins(8, 8, 8, 8)
        h.setSpacing(10)

        # Threat table
        tbl_box = QVBoxLayout()
        tbl_hdr = QLabel("▶ AKTİF TEHDİT LİSTESİ")
        tbl_hdr.setObjectName("section_hdr")
        tbl_box.addWidget(tbl_hdr)
        self.threat_table = QTableWidget(0, 9)
        self.threat_table.setHorizontalHeaderLabels(
            ["ID", "POZİSYON (°)", "MESAFİ (km)", "HIZ (m/s)", "İRTİFA (m)", "SINIF", "TEHDİT", "XAI SKOR", "KAYNAK"]
        )
        self.threat_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.threat_table.verticalHeader().setVisible(False)
        self.threat_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.threat_table.setFixedHeight(160)
        self.threat_table.itemSelectionChanged.connect(self._on_threat_select)
        tbl_box.addWidget(self.threat_table)
        h.addLayout(tbl_box, stretch=3)

        # Engagement controls
        ctrl_box = QVBoxLayout()
        ctrl_box.setSpacing(8)
        ctrl_hdr = QLabel("▶ ANGAJ KONTROL")
        ctrl_hdr.setObjectName("section_hdr")
        ctrl_box.addWidget(ctrl_hdr)

        self.lbl_selected = QLabel("Hedef: —")
        self.lbl_selected.setStyleSheet(f"color: {YELLOW_EZ}; font-size: 11px;")
        ctrl_box.addWidget(self.lbl_selected)

        btn_select = QPushButton("🎯  HEDEF SEÇ")
        btn_select.clicked.connect(self._on_select_first_threat)
        ctrl_box.addWidget(btn_select)

        btn_engage = QPushButton("🚀  ANGAJ ET")
        btn_engage.setObjectName("btn_engage")
        btn_engage.clicked.connect(self._on_engage)
        ctrl_box.addWidget(btn_engage)

        btn_defend = QPushButton("🛡️  SAVUN")
        btn_defend.setObjectName("btn_defend")
        btn_defend.clicked.connect(self._on_defend)
        ctrl_box.addWidget(btn_defend)

        ctrl_box.addStretch()
        h.addLayout(ctrl_box, stretch=1)

        return w

    # ── Logic ────────────────────────────────────────────────────────────────
    def _spawn_initial_threats(self):
        for _ in range(random.randint(3, 6)):
            self.threats.append(Threat())

    def _on_sweep(self):
        self.radar.advance_sweep(2.0)
        for thr in self.threats:
            thr.update()
        self.threats = [t for t in self.threats if t.fade > 0]
        self.radar.set_threats(self.threats)

    def _on_data_update(self):
        active = [t for t in self.threats if not t.engaged]
        self._stat_labels["detected"].setText(str(len(self.threats)))
        self._stat_labels["intercepted"].setText(str(self.intercepted_count))
        self._stat_labels["active"].setText(str(len(active)))

        self._update_threat_table()

        # Gimbal takibi (sürekli güncelleme) ve Blackbox loglama
        if self._gimbal_controller and self.selected_tid:
            for t in active:
                if t.id == self.selected_tid:
                    self._gimbal_controller.hedefe_yonel(
                        tid=t.id,
                        bearing_deg=t.bearing(),
                        distance_km=t.range_km(),
                        altitude_m=t.altitude
                    )
                    break

        if self._blackbox:
            for t in active:
                self._blackbox.log_tehdit(t)

    def _on_spawn_threat(self):
        if len([t for t in self.threats if not t.engaged]) < 8:
            self.threats.append(Threat())
            self.radar.set_threats(self.threats)

    def _update_threat_table(self):
        active = [t for t in self.threats if not t.engaged]
        self.threat_table.setRowCount(len(active))
        threat_colors = {
            "DÜŞÜK":   QColor(100, 200, 100),
            "ORTA":    QColor(255, 204, 0),
            "YÜKSEK":  QColor(255, 140, 0),
            "KRİTİK":  QColor(255, 50, 50),
        }
        kaynak_colors = {
            "fuzyon":        QColor(0, 200, 255),
            "yalniz_radar":  QColor(0, 255, 100),
            "yalniz_kamera": QColor(255, 200, 0),
            "simülasyon":    QColor(150, 150, 150),
        }
        kaynak_labels = {
            "fuzyon":        "FÜZYON",
            "yalniz_radar":  "RADAR",
            "yalniz_kamera": "KAMERA",
            "simülasyon":    "SİMÜL.",
        }
        for row, thr in enumerate(active):
            items = [
                thr.id,
                f"{thr.bearing():.1f}°",
                f"{thr.range_km():.1f}",
                f"{thr.velocity_ms:+.1f}",
                f"{thr.altitude}",
                thr.sinif,
                thr.threat_level,
                f"{getattr(thr, 'tehdit_skoru', 0.0):.0f}",
                kaynak_labels.get(thr.kaynak, thr.kaynak),
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 6:  # TEHDİT
                    item.setForeground(threat_colors.get(text, QColor(TEXT_MAIN)))
                if col == 7:  # XAI SKOR
                    score = float(text)
                    if score >= 80: color = QColor(255, 50, 50)
                    elif score >= 50: color = QColor(255, 140, 0)
                    elif score >= 30: color = QColor(255, 204, 0)
                    else: color = QColor(100, 200, 100)
                    item.setForeground(color)
                    item.setText(f"%{score:.0f}")
                if col == 8:  # KAYNAK
                    item.setForeground(kaynak_colors.get(thr.kaynak, QColor(TEXT_DIM)))
                self.threat_table.setItem(row, col, item)

    def _on_threat_select(self):
        rows = self.threat_table.selectedItems()
        if rows:
            tid = self.threat_table.item(self.threat_table.currentRow(), 0).text()
            self.selected_tid = tid
            self.radar.set_selected(tid)
            self.lbl_selected.setText(f"Hedef: {tid}")

    def _on_select_first_threat(self):
        active = [t for t in self.threats if not t.engaged]
        if active:
            t = active[0]
            self.selected_tid = t.id
            self.radar.set_selected(t.id)
            self.lbl_selected.setText(f"Hedef: {t.id}")

    def _on_engage(self):
        if not self.selected_tid:
            return
        for thr in self.threats:
            if thr.id == self.selected_tid and not thr.engaged:
                thr.engaged = True
                self.intercepted_count += 1
                if self._blackbox:
                    self._blackbox.log_olay("ENGAGEMENT", thr.id, f"Angaje olundu: {thr.sinif}")
                # Fire from next available battery
                for batt in self.batteries:
                    if batt.ammo > 0:
                        batt.fire()
                        break
                self.selected_tid = None
                self.radar.set_selected(None)
                self.lbl_selected.setText("Hedef: —")
                if self._gimbal_controller:
                    self._gimbal_controller.serbest_mod()
                break

    def _max_radar_range_km(self) -> float:
        if self._radar_bridge is not None and getattr(self._radar_bridge, "params", None):
            return float(self._radar_bridge.params.max_range_km)
        return float(_app_cfg.get("radar", {}).get("maks_menzil_km", 3.0))

    def _on_defend(self):
        """
        Optimum Silah-Hedef Atama (WTA - Weapon Target Assignment) Algoritması ile
        birden fazla hedefi eşzamanlı olarak uygun bataryalara atar.
        """
        active = [t for t in self.threats if not t.engaged]
        if not active:
            return

        try:
            from src.wta_optimizer import WTAOptimizer, BatteryState, ThreatState
            
            # 1. Batarya durumlarını oluştur
            bat_states = []
            for i, b in enumerate(self.batteries):
                profile = profile_for_battery(b.lbl_name.text())
                bat_states.append(
                    BatteryState(
                        f"BAT-{i}",
                        b.ammo,
                        max_range_km=profile.max_range_km,
                        prob_kill=profile.prob_kill,
                    )
                )

            # 2. Tehdit durumlarını oluştur
            thr_states = [ThreatState(t.id, t.range_km(), getattr(t, 'tehdit_skoru', 50.0)) for t in active]

            # 3. Yöneylem (Operations Research) optimizasyonunu çalıştır
            assignments = WTAOptimizer.optimize(bat_states, thr_states)

            # 4. Atamaları uygula
            for bat_id, thr_id in assignments:
                # Bataryayı bul ve ateşle
                bat_idx = int(bat_id.split("-")[1])
                self.batteries[bat_idx].fire()
                
                # Hedefi angaje et
                for t in self.threats:
                    if t.id == thr_id:
                        t.engaged = True
                        self.intercepted_count += 1
                        if self._blackbox:
                            self._blackbox.log_olay("WTA_ENGAGEMENT", thr_id, f"{bat_id} tarafından optimum angajman (Skor: {getattr(t, 'tehdit_skoru', 0):.0f})")
                        break

        except ImportError:
            # Fallback (scipy yoksa eski sistem)
            for thr in active[:2]:
                thr.engaged = True
                self.intercepted_count += 1
                for batt in self.batteries:
                    if batt.ammo > 0:
                        batt.fire()
                        break

    # ── Kamera entegrasyonu ───────────────────────────────────────────────────

    def _on_cam_frame(self):
        """Kamera karesini pipeline'a gönder."""
        if self._cap is None or self._pipeline is None:
            return
        try:
            import cv2
            ret, frame = self._cap.read()
            if ret:
                self._pipeline.kare_gonder(frame)
        except Exception:
            pass

    def _on_kamera_tespiti(self, tespitler: list):
        """
        YOLOv11 tespitlerini radar tehditlerine dönüştür.
        Kamerada görülen her yeni track_id için bir Threat oluşturur;
        bbox pozisyonundan yaklaşık bearing, bbox büyüklüğünden mesafe tahmin eder.
        """
        gelen_idler = set()
        for d in tespitler:
            tid = d["track_id"]
            gelen_idler.add(tid)

            if tid not in self._kamera_tespitler:
                # Yeni tehdit — kamera çerçevesinden radar koordinatına dönüştür
                # cx [0..1] → bearing [-60°..+60°] (kameranın görüş açısı ~120°)
                bearing_deg = (d["cx"] - 0.5) * 120.0
                # bbox diyagonalı → mesafe (büyük = yakın, küçük = uzak)
                diag = (d["w"] ** 2 + d["h"] ** 2) ** 0.5
                dist_norm = max(0.3, min(0.95, 1.0 - diag * 2))
                estimated_range_km = display_radius_to_km(dist_norm, self._max_radar_range_km())

                thr = Threat()
                thr.x = dist_norm * math.cos(math.radians(bearing_deg))
                thr.y = dist_norm * math.sin(math.radians(bearing_deg))
                thr.physical_range_km = estimated_range_km
                # Hız — kameradan göreli harekete göre ayarlanabilir, şimdilik yavaş
                thr.vx = random.uniform(-0.001, 0.001)
                thr.vy = random.uniform(-0.001, 0.001)
                thr.threat_level = "YÜKSEK" if d["guven"] > 0.7 else "ORTA"
                thr.tehdit_skoru = 65.0 if d["guven"] > 0.7 else 40.0
                thr.sinif = d.get("sinif", "Bilinmeyen")
                thr.kaynak = "yalniz_kamera"

                self._kamera_tespitler[tid] = thr
                self.threats.append(thr)
                self.radar.set_threats(self.threats)
            else:
                # Mevcut tehdidin pozisyonunu güncelle
                thr = self._kamera_tespitler[tid]
                if not thr.engaged:
                    bearing_deg = (d["cx"] - 0.5) * 120.0
                    diag = (d["w"] ** 2 + d["h"] ** 2) ** 0.5
                    dist_norm = max(0.3, min(0.95, 1.0 - diag * 2))
                    thr.x = dist_norm * math.cos(math.radians(bearing_deg))
                    thr.y = dist_norm * math.sin(math.radians(bearing_deg))
                    thr.physical_range_km = display_radius_to_km(
                        dist_norm, self._max_radar_range_km()
                    )
                    thr.tehdit_skoru = 65.0 if d["guven"] > 0.7 else 40.0
                    thr.sinif = d.get("sinif", thr.sinif)

        # Kayıp track'leri temizle
        kayip = [tid for tid in self._kamera_tespitler if tid not in gelen_idler]
        for tid in kayip:
            del self._kamera_tespitler[tid]

    # ── Radar / Füzyon callback'leri ──────────────────────────────────────────

    def _on_fuzyon_tespiti(self, fused_tracks: list):
        """
        SensorFusion'dan birleştirilmiş izleri al ve Threat modeline dönüştür.
        Her FusedTrack, radar mesafesi + kamera sınıfını içerir.
        """
        gelen_idler = set()
        for ft in fused_tracks:
            ftid = ft["track_id"]
            gelen_idler.add(ftid)

            if ftid not in self._fuzyon_izleri:
                # Yeni birleştirilmiş tehdit
                thr = Threat()
                range_km = ft.get("range_km", 1.0)
                bearing_deg = ft.get("bearing_deg", 0.0)
                thr.x, thr.y = polar_to_display_xy(
                    range_km, bearing_deg, self._max_radar_range_km()
                )
                thr.physical_range_km = range_km
                thr.velocity_ms = ft.get("velocity_ms", 0.0)
                thr.snr_db = ft.get("radar_snr_db", 0.0)
                thr.sinif = ft.get("sinif", "Bilinmeyen")
                thr.kaynak = ft.get("kaynak", "fuzyon")
                thr.threat_level = ft.get("tehdit_seviyesi", "ORTA")
                thr.tehdit_skoru = ft.get("tehdit_skoru", 0.0)
                thr.altitude = int(ft.get("altitude_m", thr.altitude))
                thr.fusion_track_id = ftid
                # Hız vektörü (yaklaşıyor ise merkeze doğru)
                d = math.sqrt(thr.x ** 2 + thr.y ** 2)
                if d > 0 and thr.velocity_ms < 0:
                    thr.vx = -thr.x / d * abs(thr.velocity_ms) * 0.00001
                    thr.vy = -thr.y / d * abs(thr.velocity_ms) * 0.00001
                self._fuzyon_izleri[ftid] = thr
                self.threats.append(thr)
            else:
                # Mevcut izi güncelle
                thr = self._fuzyon_izleri[ftid]
                if not thr.engaged:
                    range_km = ft.get("range_km", 1.0)
                    bearing_deg = ft.get("bearing_deg", 0.0)
                    thr.x, thr.y = polar_to_display_xy(
                        range_km, bearing_deg, self._max_radar_range_km()
                    )
                    thr.physical_range_km = range_km
                    thr.velocity_ms = ft.get("velocity_ms", 0.0)
                    thr.snr_db = ft.get("radar_snr_db", 0.0)
                    thr.sinif = ft.get("sinif", thr.sinif)
                    thr.kaynak = ft.get("kaynak", thr.kaynak)
                    thr.threat_level = ft.get("tehdit_seviyesi", thr.threat_level)
                    thr.tehdit_skoru = ft.get("tehdit_skoru", thr.tehdit_skoru)
                    thr.altitude = int(ft.get("altitude_m", thr.altitude))

        self.radar.set_threats(self.threats)

        # Kaybolan izleri temizle
        kayip_idler = [tid for tid in self._fuzyon_izleri if tid not in gelen_idler]
        for tid in kayip_idler:
            thr = self._fuzyon_izleri[tid]
            if thr in self.threats:
                self.threats.remove(thr)
            del self._fuzyon_izleri[tid]

    def _on_radar_dogrudan_tespit(self, tespitler: list):
        """
        Füzyon kapalıyken radar tespitlerini doğrudan Threat'e dönüştür.
        """
        for det in tespitler:
            thr = Threat()
            range_km = det.get("range_km", 1.0)
            bearing_deg = det.get("bearing_deg", 0.0)
            thr.x, thr.y = polar_to_display_xy(
                range_km, bearing_deg, self._max_radar_range_km()
            )
            thr.physical_range_km = range_km
            thr.velocity_ms = det.get("velocity_ms", 0.0)
            thr.snr_db = det.get("snr_db", 0.0)
            thr.kaynak = "yalniz_radar"
            thr.threat_level = "ORTA" if det.get("snr_db", 0) > 15 else "DÜŞÜK"
            thr.tehdit_skoru = 35.0 if det.get("snr_db", 0) > 15 else 15.0
            self.threats.append(thr)
        self.radar.set_threats(self.threats)

    def _on_radar_durum(self, mesaj: str):
        """Radar durum mesajını GUI'de göster."""
        logging.getLogger(__name__).info(f"[Radar] {mesaj}")

    def _on_radar_baglanti(self, bagli: bool):
        """Radar bağlantı durumu değiştiğinde GUI'yi güncelle."""
        if bagli:
            self._stat_labels["sys_status"].setText("RADAR AKTİF")
            self._stat_labels["sys_status"].setStyleSheet(f"color: {GREEN_RDR}; font-size: 24px; font-weight: 800;")
            self._radar_status_lbl.setText("● BAĞLI — AKTİF")
            self._radar_status_lbl.setStyleSheet(f"color: {GREEN_RDR}; font-size: 10px; font-weight: bold;")
            self._aeris_badge.setStyleSheet(
                f"color: {GREEN_RDR}; font-size: 11px; font-weight: bold; "
                f"background: rgba(0,255,0,0.1); padding: 2px 8px; border-radius: 4px; "
                f"border: 1px solid rgba(0,255,0,0.3);"
            )
            # Radar bilgilerini güncelle
            if self._radar_bridge and self._radar_bridge.params:
                p = self._radar_bridge.params
                self._radar_info_labels.get("FREKANS", QLabel()).setText(f"{p.center_freq_hz/1e9:.1f} GHz")
                self._radar_info_labels.get("MENZIL", QLabel()).setText(f"{p.max_range_km:.0f} km")
                self._radar_info_labels.get("MOD", QLabel()).setText("MOCK" if self._radar_bridge._mock else "DONANIM")
        else:
            self._stat_labels["sys_status"].setText("RADAR YOK")
            self._stat_labels["sys_status"].setStyleSheet(f"color: {YELLOW_EZ}; font-size: 24px; font-weight: 800;")
            self._radar_status_lbl.setText("● BAĞLANTI KESİK")
            self._radar_status_lbl.setStyleSheet(f"color: {RED_THR}; font-size: 10px; font-weight: bold;")
            self._aeris_badge.setStyleSheet(
                f"color: {RED_THR}; font-size: 11px; font-weight: bold; "
                f"background: rgba(255,34,34,0.1); padding: 2px 8px; border-radius: 4px; "
                f"border: 1px solid rgba(255,34,34,0.3);"
            )

    def _on_gimbal_durum(self, durum: dict):
        """Gimbal'dan gelen güncel durum bilgisini arayüze yansıt."""
        if "pan" in durum:
            self._gimbal_info_labels["PAN"].setText(f"{durum['pan']:.1f}°")
        if "tilt" in durum:
            self._gimbal_info_labels["TILT"].setText(f"{durum['tilt']:.1f}°")
        if "hedef_id" in durum:
            self._gimbal_info_labels["HEDEF"].setText(durum['hedef_id'])
            if durum['hedef_id'] != "YOK":
                self._gimbal_status_lbl.setText("● TAKİP EDİYOR")
                self._gimbal_status_lbl.setStyleSheet(f"color: {GREEN_RDR}; font-size: 10px; font-weight: bold;")
            else:
                self._gimbal_status_lbl.setText("● SERBEST MOD")
                self._gimbal_status_lbl.setStyleSheet(f"color: {YELLOW_EZ}; font-size: 10px; font-weight: bold;")

    def closeEvent(self, event):
        if self._blackbox is not None:
            self._blackbox.durdur()
        if self._gimbal_controller is not None:
            self._gimbal_controller.durdur()
        if self._radar_bridge is not None:
            self._radar_bridge.durdur()
        if self._sensor_fusion is not None:
            self._sensor_fusion.temizle()
        if self._pipeline is not None:
            self._pipeline.durdur()
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        print("[DEBUG] QApplication baslatiliyor...")
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        
        print("[DEBUG] Palet ayarlaniyor...")
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(BG_DARK))
        palette.setColor(QPalette.WindowText, QColor(TEXT_MAIN))
        palette.setColor(QPalette.Base, QColor(BG_PANEL))
        palette.setColor(QPalette.AlternateBase, QColor(BG_MID))
        palette.setColor(QPalette.Text, QColor(TEXT_MAIN))
        palette.setColor(QPalette.Button, QColor(BG_MID))
        palette.setColor(QPalette.ButtonText, QColor(TEXT_MAIN))
        app.setPalette(palette)

        win = CelikKubbeGUI()
        win.show()
        sys.exit(app.exec())
    except Exception as e:
        print("\n" + "!" * 60)
        print(f"KRITIK HATA: Uygulama calisirken bir sorun olustu!")
        print(f"Hata Mesaji: {e}")
        import traceback
        traceback.print_exc()
        print("!" * 60 + "\n")
        input("Devam etmek icin bir tusa basin...")
        sys.exit(1)
