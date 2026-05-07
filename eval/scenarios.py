"""
eval/scenarios.py — Ablation için standart senaryo üreticileri.

Dört senaryo türü:
  single_threat   — 1 hedef, ideal koşullar (taban çizgisi)
  multi_threat    — 5 hedef, yakın geçişler + potansiyel ID swap
  sensor_dropout  — Radar 5 saniye kesiliyor (50 frame @10Hz)
  low_snr         — Yüksek radar gürültüsü, düşük SNR

Her senaryo bir dict döndürür:
  {
    "name": str,
    "frames": list[GTFrame],
    "gt_mot": list[dict],
    "coco_gt": dict,
    "meta": dict,
  }
"""

from __future__ import annotations

from typing import Optional

from eval.ground_truth import SyntheticGTGenerator, GTFrame

N_FRAMES_DEFAULT = 150
DT = 0.1


def single_threat(seed: int = 0, n_frames: int = N_FRAMES_DEFAULT) -> dict:
    # GT tüm sensör menzilini kapsar (kamera+radar); tracker kendi menzilini uygular
    gen = SyntheticGTGenerator(
        img_w=1280, img_h=720, fov_deg=60.0,
        camera_range_m=1200.0, radar_range_m=15000.0,
        radar_noise_m=3.0, radar_noise_deg=0.05,
        seed=seed,
    )
    frames = gen.generate_sequence(n_frames=n_frames, n_targets=1, dt=DT)
    gt_mot, _ = gen.to_mot_format(frames)
    coco_gt = gen.to_coco_format(frames)
    return {
        "name": "single_threat",
        "frames": frames,
        "gt_mot": gt_mot,
        "coco_gt": coco_gt,
        "generator": gen,
        "meta": {"n_targets": 1, "radar_noise_m": 3.0, "dropout": False},
    }


def multi_threat(seed: int = 0, n_frames: int = N_FRAMES_DEFAULT,
                 n_targets: int = 5) -> dict:
    gen = SyntheticGTGenerator(
        img_w=1280, img_h=720, fov_deg=60.0,
        camera_range_m=1200.0, radar_range_m=15000.0,
        radar_noise_m=5.0, radar_noise_deg=0.15,
        seed=seed,
    )
    frames = gen.generate_sequence(n_frames=n_frames, n_targets=n_targets, dt=DT)
    gt_mot, _ = gen.to_mot_format(frames)
    coco_gt = gen.to_coco_format(frames)
    return {
        "name": "multi_threat",
        "frames": frames,
        "gt_mot": gt_mot,
        "coco_gt": coco_gt,
        "generator": gen,
        "meta": {"n_targets": n_targets, "radar_noise_m": 5.0, "dropout": False},
    }


def sensor_dropout(seed: int = 0, n_frames: int = N_FRAMES_DEFAULT,
                   dropout_start: int = 50, dropout_len: int = 50) -> dict:
    dropout_frames = list(range(dropout_start, dropout_start + dropout_len))
    gen = SyntheticGTGenerator(
        img_w=1280, img_h=720, fov_deg=60.0,
        camera_range_m=1200.0, radar_range_m=15000.0,
        radar_noise_m=4.0, radar_noise_deg=0.10,
        seed=seed,
    )
    frames = gen.generate_sequence(
        n_frames=n_frames, n_targets=3, dt=DT,
        radar_dropout_frames=dropout_frames,
    )
    gt_mot, _ = gen.to_mot_format(frames)
    coco_gt = gen.to_coco_format(frames)
    return {
        "name": "sensor_dropout",
        "frames": frames,
        "gt_mot": gt_mot,
        "coco_gt": coco_gt,
        "generator": gen,
        "meta": {
            "n_targets": 3,
            "dropout": True,
            "dropout_frames": len(dropout_frames),
            "dropout_start": dropout_start,
        },
    }


def low_snr(seed: int = 0, n_frames: int = N_FRAMES_DEFAULT) -> dict:
    gen = SyntheticGTGenerator(
        img_w=1280, img_h=720, fov_deg=60.0,
        camera_range_m=1200.0, radar_range_m=15000.0,
        radar_noise_m=25.0,
        radar_noise_deg=1.5,
        seed=seed,
    )
    frames = gen.generate_sequence(n_frames=n_frames, n_targets=2, dt=DT)
    gt_mot, _ = gen.to_mot_format(frames)
    coco_gt = gen.to_coco_format(frames)
    return {
        "name": "low_snr",
        "frames": frames,
        "gt_mot": gt_mot,
        "coco_gt": coco_gt,
        "generator": gen,
        "meta": {"n_targets": 2, "radar_noise_m": 25.0, "dropout": False},
    }


ALL_SCENARIOS = [single_threat, multi_threat, sensor_dropout, low_snr]


def build_all(seed: int = 0) -> list[dict]:
    return [fn(seed=seed) for fn in ALL_SCENARIOS]
