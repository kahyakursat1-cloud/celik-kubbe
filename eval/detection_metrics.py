"""
eval/detection_metrics.py — Detection metrikleri: mAP@0.5, mAP@0.5:0.95.

pycocotools üzerine wrapper. COCO annotation formatında GT ve prediction alır.

Kullanım:
    from eval.detection_metrics import compute_detection_metrics
    result = compute_detection_metrics(gt_coco, pred_list)
    print(result["mAP@0.5"], result["mAP@0.5:0.95"])
"""

from __future__ import annotations

import copy
from typing import Any

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_AVAILABLE = True
except ImportError:
    COCO_AVAILABLE = False


def compute_detection_metrics(
    gt_coco_dict: dict[str, Any],
    predictions: list[dict[str, Any]],
    iou_type: str = "bbox",
) -> dict[str, float]:
    """
    mAP@0.5 ve mAP@0.5:0.95 hesaplar.

    Parametreler
    ----------
    gt_coco_dict : COCO formatında GT dict (images, annotations, categories)
    predictions : list of dict — her biri:
        {"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}
    iou_type : "bbox" (default) veya "segm"

    Döndürür
    --------
    dict: mAP@0.5, mAP@0.5:0.95, mAP@S, mAP@M, mAP@L (COCO standart)
    """
    if not COCO_AVAILABLE:
        raise ImportError("pycocotools kurulu değil: pip install pycocotools")

    if not predictions:
        return {
            "mAP@0.5": 0.0,
            "mAP@0.5:0.95": 0.0,
            "mAP@small": float("nan"),
            "mAP@medium": float("nan"),
            "mAP@large": float("nan"),
            "AR@1": 0.0,
            "AR@10": 0.0,
        }

    import io
    import contextlib
    import sys

    coco_gt = COCO()
    coco_gt.dataset = copy.deepcopy(gt_coco_dict)
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(predictions)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        evaluator = COCOeval(coco_gt, coco_dt, iou_type)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = evaluator.stats
    return {
        "mAP@0.5:0.95": float(stats[0]),
        "mAP@0.5": float(stats[1]),
        "mAP@0.75": float(stats[2]),
        "mAP@small": float(stats[3]),
        "mAP@medium": float(stats[4]),
        "mAP@large": float(stats[5]),
        "AR@1": float(stats[6]),
        "AR@10": float(stats[7]),
        "AR@100": float(stats[8]),
    }


def build_coco_gt(
    images: list[dict],
    annotations: list[dict],
    categories: list[dict],
) -> dict:
    """COCO GT dict oluşturur. images/annotations/categories listelerinden."""
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def bbox_from_xyxy(x1: float, y1: float, x2: float, y2: float) -> list[float]:
    """[x1,y1,x2,y2] → [x,y,w,h] COCO formatına dönüştürür."""
    return [x1, y1, x2 - x1, y2 - y1]
