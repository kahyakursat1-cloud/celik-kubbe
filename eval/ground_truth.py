"""
eval/ground_truth.py — Sentetik veri için Ground Truth üreticisi.

3D model konumu + kamera projeksiyonu → BBox GT
Radar: fiziksel mesafe + açı doğruluk değerleri.

Kullanım:
    from eval.ground_truth import SyntheticGTGenerator
    gen = SyntheticGTGenerator(img_w=1280, img_h=720, fov_deg=60.0)
    gt = gen.generate_sequence(n_frames=100, n_targets=3)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TargetState:
    target_id: int
    x_m: float
    y_m: float
    z_m: float
    vx_ms: float
    vy_ms: float
    vz_ms: float
    sinif: str = "Drone"
    rcs_m2: float = 0.01

    def step(self, dt: float = 0.1) -> None:
        self.x_m += self.vx_ms * dt
        self.y_m += self.vy_ms * dt
        self.z_m += self.vz_ms * dt


@dataclass
class GTFrame:
    frame_id: int
    timestamp_s: float
    track_ids: list[int] = field(default_factory=list)
    bboxes: list[tuple] = field(default_factory=list)
    ranges_m: list[float] = field(default_factory=list)
    radar_detections: list[dict] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    radar_available: bool = True  # sensor_dropout senaryosunda False olur


CLASS_RCS = {
    "Drone": 0.01,
    "Helikopter": 1.5,
    "Balistik_Fuze": 0.05,
    "Jet": 5.0,
    "Artillery": 0.1,
    "FixedWingUAV": 0.03,
}

DEFAULT_CLASSES = list(CLASS_RCS.keys())


class SyntheticGTGenerator:
    """
    Basit sahne simülatörü: hedefler sabit hızla hareket eder,
    kamera ve radar GT üretilir.
    """

    def __init__(
        self,
        img_w: int = 1280,
        img_h: int = 720,
        fov_deg: float = 60.0,
        focal_length_px: Optional[float] = None,
        camera_range_m: float = 1000.0,
        radar_range_m: float = 15000.0,
        radar_noise_m: float = 5.0,
        radar_noise_deg: float = 0.1,
        seed: int = 42,
    ):
        self.img_w = img_w
        self.img_h = img_h
        self.fov_rad = math.radians(fov_deg)
        self.focal_px = focal_length_px or (img_w / 2) / math.tan(self.fov_rad / 2)
        self.camera_range_m = camera_range_m
        self.radar_range_m = radar_range_m
        self.radar_noise_m = radar_noise_m
        self.radar_noise_deg = radar_noise_deg
        self._rng = random.Random(seed)

    def _project_to_image(
        self, target: TargetState
    ) -> Optional[tuple[tuple[float, float, float, float], float]]:
        """3D pozisyon → ((x,y,w,h), dist_m); görünmüyorsa None."""
        if target.z_m <= 5.0:  # Kamera arkasında veya çok yakın
            return None
        dist = math.sqrt(target.x_m**2 + target.y_m**2 + target.z_m**2)
        if dist < 1.0 or dist > self.camera_range_m:
            return None

        cx_px = self.img_w / 2 + (target.x_m / target.z_m) * self.focal_px
        cy_px = self.img_h / 2 - (target.y_m / target.z_m) * self.focal_px

        apparent_size_m = max(0.5, math.sqrt(CLASS_RCS.get(target.sinif, 0.01)) * 3)
        box_px = (apparent_size_m / dist) * self.focal_px * 2
        box_px = max(4.0, min(box_px, 200.0))

        x1 = cx_px - box_px / 2
        y1 = cy_px - box_px / 2
        if x1 > self.img_w or y1 > self.img_h or x1 + box_px < 0 or y1 + box_px < 0:
            return None

        bbox = (
            max(0.0, x1),
            max(0.0, y1),
            min(box_px, self.img_w - max(0.0, x1)),
            min(box_px, self.img_h - max(0.0, y1)),
        )
        return bbox, dist

    def _radar_detection(self, target: TargetState) -> Optional[dict]:
        dist = math.sqrt(target.x_m**2 + target.y_m**2 + target.z_m**2)
        if dist > self.radar_range_m:
            return None

        bearing = math.degrees(math.atan2(target.x_m, target.z_m))
        range_km = dist / 1000.0

        noisy_range = range_km + self._rng.gauss(0, self.radar_noise_m / 1000.0)
        noisy_bearing = bearing + self._rng.gauss(0, self.radar_noise_deg)

        velocity = math.sqrt(target.vx_ms**2 + target.vy_ms**2 + target.vz_ms**2)
        snr = max(0.0, 20 * math.log10(
            (target.rcs_m2 * (self.radar_range_m**4)) / (dist**4 + 1e-6) + 1e-9
        ) + 30)

        return {
            "target_id": target.target_id,
            "range_km": float(noisy_range),
            "bearing_deg": float(noisy_bearing),
            "velocity_ms": float(velocity),
            "snr_db": float(snr),
            "range_km_true": float(range_km),
            "bearing_deg_true": float(bearing),
        }

    def _random_target(self, tid: int) -> TargetState:
        sinif = self._rng.choice(DEFAULT_CLASSES)
        # z ∈ [200, 700] — kamera önünde, içinde kalacak şekilde
        z = self._rng.uniform(200, min(700, self.camera_range_m * 0.55))
        x = self._rng.uniform(-150, 150)
        y = self._rng.uniform(20, 180)
        speed = self._rng.uniform(10, 40)
        # Ağırlıklı olarak yatay hareket; z bileşeni küçük (kamerayı aşmaz)
        lateral_frac = self._rng.uniform(0.7, 0.95)
        sign = self._rng.choice([-1, 1])
        vx = sign * speed * lateral_frac
        vz = -speed * (1.0 - lateral_frac) * 0.5  # çok yavaş yaklaşma
        return TargetState(
            target_id=tid,
            x_m=x, y_m=y, z_m=z,
            vx_ms=vx,
            vy_ms=self._rng.uniform(-2, 2),
            vz_ms=vz,
            sinif=sinif,
            rcs_m2=CLASS_RCS.get(sinif, 0.01),
        )

    def generate_sequence(
        self,
        n_frames: int = 100,
        n_targets: int = 3,
        dt: float = 0.1,
        radar_dropout_frames: Optional[list[int]] = None,
    ) -> list[GTFrame]:
        """
        n_frames boyunca n_targets hedef için GT frame listesi üretir.

        radar_dropout_frames: radar verisinin kesildiği frame ID listesi.
        """
        targets = [self._random_target(i) for i in range(n_targets)]
        frames: list[GTFrame] = []
        dropout = set(radar_dropout_frames or [])

        for fid in range(n_frames):
            ts = fid * dt
            radar_ok = fid not in dropout
            gt_frame = GTFrame(frame_id=fid, timestamp_s=ts,
                               radar_available=radar_ok)

            for t in targets:
                result = self._project_to_image(t)
                if result is not None:
                    bbox, dist = result
                    gt_frame.track_ids.append(t.target_id)
                    gt_frame.bboxes.append(bbox)
                    gt_frame.ranges_m.append(dist)
                    gt_frame.classes.append(t.sinif)

                if radar_ok:
                    det = self._radar_detection(t)
                    if det is not None:
                        gt_frame.radar_detections.append(det)

                t.step(dt)

            frames.append(gt_frame)
        return frames

    def to_mot_format(
        self, frames: list[GTFrame]
    ) -> tuple[list[dict], list[dict]]:
        """
        GTFrame listesi → (gt_frames, hyp_frames) motmetrics formatına dönüştürür.
        hyp_frames = gt_frames (mükemmel tracker simülasyonu — MOTA=1.0 beklenir).
        """
        mot_frames = []
        for f in frames:
            mot_frames.append({
                "frame_id": f.frame_id,
                "ids": f.track_ids,
                "boxes": list(f.bboxes),
            })
        return mot_frames, mot_frames  # gt ve perfect hyp aynı

    def to_coco_format(
        self, frames: list[GTFrame], category_map: Optional[dict[str, int]] = None
    ) -> dict:
        """GTFrame listesi → COCO detection GT formatına dönüştürür."""
        if category_map is None:
            category_map = {c: i + 1 for i, c in enumerate(DEFAULT_CLASSES)}

        images = []
        annotations = []
        ann_id = 0

        for f in frames:
            images.append({"id": f.frame_id, "width": self.img_w, "height": self.img_h})
            for tid, bbox, cls in zip(f.track_ids, f.bboxes,
                                      f.classes if f.classes else [""]*len(f.track_ids)):
                x, y, w, h = bbox
                ann_id += 1
                annotations.append({
                    "id": ann_id,
                    "image_id": f.frame_id,
                    "category_id": category_map.get(cls, 1),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })

        categories = [{"id": v, "name": k} for k, v in category_map.items()]
        return {"images": images, "annotations": annotations, "categories": categories}
