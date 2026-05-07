"""
eval/evaluator.py — Birleşik değerlendirici: MOT + Detection metrikleri.

Kullanım:
    from eval.evaluator import Evaluator
    from eval.ground_truth import SyntheticGTGenerator

    gen = SyntheticGTGenerator(seed=0)
    frames = gen.generate_sequence(n_frames=50, n_targets=2)
    gt_mot, hyp_mot = gen.to_mot_format(frames)
    coco_gt = gen.to_coco_format(frames)

    ev = Evaluator()
    result = ev.evaluate_mot(gt_mot, hyp_mot)
    print(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from eval.metrics import compute_mot_metrics, format_metrics_table, MM_AVAILABLE
from eval.detection_metrics import compute_detection_metrics, COCO_AVAILABLE


@dataclass
class EvalResult:
    mot: dict[str, float] = field(default_factory=dict)
    detection: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        if self.meta:
            lines.append(f"  Config: {self.meta}")
        if self.mot:
            lines.append(format_metrics_table(self.mot, "MOT Metrics"))
        if self.detection:
            lines.append("─" * 40)
            lines.append(" Detection Metrics")
            lines.append("─" * 40)
            for k in ("mAP@0.5", "mAP@0.5:0.95", "mAP@0.75"):
                if k in self.detection:
                    lines.append(f"  {k:<18} {self.detection[k]:.3f}")
        return "\n".join(lines)


class Evaluator:
    """
    Tek nokta değerlendirme arayüzü.

    Ablation runner, comparison runner ve monte carlo tüm metrikleri
    bu sınıf üzerinden alır — bağımlılığı merkezi tutar.
    """

    def __init__(self, max_iou_dist: float = 0.5):
        self.max_iou_dist = max_iou_dist

    def evaluate_mot(
        self,
        gt_frames: list[dict],
        hyp_frames: list[dict],
        meta: Optional[dict] = None,
    ) -> EvalResult:
        if not MM_AVAILABLE:
            raise RuntimeError("motmetrics kurulu değil")
        mot = compute_mot_metrics(gt_frames, hyp_frames, self.max_iou_dist)
        return EvalResult(mot=mot, meta=meta or {})

    def evaluate_detection(
        self,
        gt_coco_dict: dict,
        predictions: list[dict],
        meta: Optional[dict] = None,
    ) -> EvalResult:
        if not COCO_AVAILABLE:
            raise RuntimeError("pycocotools kurulu değil")
        det = compute_detection_metrics(gt_coco_dict, predictions)
        return EvalResult(detection=det, meta=meta or {})

    def evaluate_full(
        self,
        gt_frames_mot: list[dict],
        hyp_frames_mot: list[dict],
        gt_coco_dict: dict,
        predictions: list[dict],
        meta: Optional[dict] = None,
    ) -> EvalResult:
        mot = compute_mot_metrics(gt_frames_mot, hyp_frames_mot, self.max_iou_dist)
        det = compute_detection_metrics(gt_coco_dict, predictions)
        return EvalResult(mot=mot, detection=det, meta=meta or {})

    @staticmethod
    def perfect_detections_from_gt(
        coco_gt: dict, score: float = 1.0
    ) -> list[dict]:
        """COCO GT'den mükemmel tahmin listesi üretir (test için)."""
        preds = []
        for ann in coco_gt.get("annotations", []):
            preds.append({
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "score": score,
            })
        return preds
