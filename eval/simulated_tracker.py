"""
eval/simulated_tracker.py — Ablasyon için parametrik tracker simülatörü.

Gerçek SensorFusion yerine, config flag'lerine göre GT üzerinden
kontrollü hata enjeksiyonu yaparak hypothesis frame'leri üretir.

Flag semantiği:
  fusion_on    — False: yalnız kamera (sınırlı mesafe, radar yok)
                 True:  radar+kamera füzyonu (geniş mesafe, daha doğru)
  kalman_on    — False: ham ölçüm gürültüsü tracks'e yansır
                 True:  Kalman-like gürültü azaltımı
  xai_adaptive — False: sabit güven eşiği
                 True:  bağlama duyarlı eşik (daha az FN, biraz fazla FP)

Deterministik detection: (frame_id, target_id, seed) hash'i → RNG cross-
contamination olmadan temiz ablasyon karşılaştırması.
"""

from __future__ import annotations

import math
import random
import struct
from dataclasses import dataclass
from typing import Optional

from eval.ground_truth import GTFrame

BBox = tuple[float, float, float, float]


@dataclass
class TrackerConfig:
    fusion_on: bool = True
    kalman_on: bool = True
    xai_adaptive: bool = False
    camera_only_range_m: float = 550.0   # fusion_on=False: kamera kör eşiği
    raw_noise_px: float = 2.2            # kalman_on=False'da piksel gürültüsü (σ)
    kalman_noise_scale: float = 0.30     # kalman_on=True: 2.2×0.30=0.66 σ
    # Detection miss probabilities
    miss_prob_fusion_on:  float = 0.05
    miss_prob_fusion_off: float = 0.16   # kamera-only daha çok kaçırır
    miss_prob_beyond_range: float = 0.92  # radar yok, hedef uzakta
    fp_prob_fixed:    float = 0.008
    fp_prob_adaptive: float = 0.018

    def label(self) -> str:
        return (
            f"fu{'1' if self.fusion_on else '0'}"
            f"_ka{'1' if self.kalman_on else '0'}"
            f"_xa{'1' if self.xai_adaptive else '0'}"
        )

    def __repr__(self) -> str:
        return (
            f"TrackerConfig(fusion={self.fusion_on}, "
            f"kalman={self.kalman_on}, xai_adaptive={self.xai_adaptive})"
        )


ALL_CONFIGS = [
    TrackerConfig(fusion_on=f, kalman_on=k, xai_adaptive=x)
    for f in (False, True)
    for k in (False, True)
    for x in (False, True)
]


def _lcg_float(seed: int) -> float:
    """Basit LCG → [0,1) float. RNG state bağımsız deterministik karar."""
    val = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return (val >> 11) / (2**53)


class SimulatedTracker:
    """
    GT frame listesinden, TrackerConfig'e göre hypothesis üretir.

    Detection kararları deterministik (frame_id × target_id × global_seed hash).
    Bbox pertürbasyonu ve FP için ayrı RNG → clean ablation comparison.
    """

    def __init__(self, config: TrackerConfig, seed: int = 0):
        self.cfg = config
        self._seed = seed
        self._noise_rng = random.Random(seed ^ 0xDEAD)
        self._fp_rng = random.Random(seed ^ 0xBEEF)

    def _miss_prob(self, dist_m: float, radar_available: bool) -> float:
        beyond_range = dist_m > self.cfg.camera_only_range_m
        if self.cfg.fusion_on:
            if radar_available:
                base = self.cfg.miss_prob_fusion_on
            else:
                # Radar dropout → füzyon faydası kaybolur, kamera-only'ye düşer
                base = (self.cfg.miss_prob_beyond_range if beyond_range
                        else self.cfg.miss_prob_fusion_off)
        else:
            base = (self.cfg.miss_prob_beyond_range if beyond_range
                    else self.cfg.miss_prob_fusion_off)
        if self.cfg.xai_adaptive:
            base *= 0.75
        return base

    def _should_see(self, frame_id: int, target_id: int, dist_m: float,
                    radar_available: bool = True) -> bool:
        miss_p = self._miss_prob(dist_m, radar_available)
        det_hash = _lcg_float(self._seed * 131071 + frame_id * 9973 + target_id * 2654435761)
        return det_hash > miss_p

    def _perturb_bbox(self, bbox: BBox) -> BBox:
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        noise = (self.cfg.raw_noise_px * self.cfg.kalman_noise_scale
                 if self.cfg.kalman_on else self.cfg.raw_noise_px)
        cx += self._noise_rng.gauss(0, noise)
        cy += self._noise_rng.gauss(0, noise)
        return (cx - w / 2, cy - h / 2, w, h)

    def _maybe_add_fp(self, hyp_ids: list[int], hyp_boxes: list[BBox]) -> None:
        fp_prob = (self.cfg.fp_prob_adaptive if self.cfg.xai_adaptive
                   else self.cfg.fp_prob_fixed)
        if self._fp_rng.random() < fp_prob:
            fake_id = 9000 + self._fp_rng.randint(0, 999)
            while fake_id in hyp_ids:
                fake_id += 1
            fx = self._fp_rng.uniform(50, 1200)
            fy = self._fp_rng.uniform(50, 650)
            hyp_ids.append(fake_id)
            hyp_boxes.append((fx, fy, 35.0, 35.0))

    def process(self, frames: list[GTFrame]) -> list[dict]:
        hyp_frames = []
        for f in frames:
            hyp_ids: list[int] = []
            hyp_boxes: list[BBox] = []
            ranges = f.ranges_m if f.ranges_m else [300.0] * len(f.track_ids)
            radar_ok = getattr(f, "radar_available", True)

            for tid, bbox, dist_m in zip(f.track_ids, f.bboxes, ranges):
                if self._should_see(f.frame_id, tid, dist_m, radar_ok):
                    hyp_ids.append(tid)
                    hyp_boxes.append(self._perturb_bbox(bbox))

            if f.track_ids:  # GT var → FP mümkün
                self._maybe_add_fp(hyp_ids, hyp_boxes)

            hyp_frames.append({
                "frame_id": f.frame_id,
                "ids": hyp_ids,
                "boxes": hyp_boxes,
            })
        return hyp_frames
