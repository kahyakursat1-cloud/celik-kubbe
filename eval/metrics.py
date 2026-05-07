"""
eval/metrics.py — Multi-Object Tracking metrikleri (MOTA, MOTP, IDF1, HOTA).

motmetrics kütüphanesi üzerine ince wrapper. Giriş: GT ve hypothesis frame listesi.

Kullanım:
    from eval.metrics import compute_mot_metrics
    result = compute_mot_metrics(gt_frames, hyp_frames, max_iou=0.5)
    print(result["MOTA"], result["IDF1"])
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

try:
    import motmetrics as mm
    MM_AVAILABLE = True
except ImportError:
    MM_AVAILABLE = False


BBox = tuple[float, float, float, float]  # x, y, w, h (COCO format)


def _iou_distance(gt_boxes: list[BBox], hyp_boxes: list[BBox]) -> np.ndarray:
    """IoU tabanlı maliyet matrisi döndürür (1 - IoU)."""
    if not gt_boxes or not hyp_boxes:
        return np.empty((len(gt_boxes), len(hyp_boxes)), dtype=float)

    def box_area(b):
        return b[2] * b[3]

    def intersect(a, b):
        ax1, ay1 = a[0], a[1]
        ax2, ay2 = a[0] + a[2], a[1] + a[3]
        bx1, by1 = b[0], b[1]
        bx2, by2 = b[0] + b[2], b[1] + b[3]
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        return iw * ih

    C = np.ones((len(gt_boxes), len(hyp_boxes)), dtype=float)
    for i, g in enumerate(gt_boxes):
        for j, h in enumerate(hyp_boxes):
            inter = intersect(g, h)
            union = box_area(g) + box_area(h) - inter
            iou = inter / union if union > 0 else 0.0
            C[i, j] = 1.0 - iou
    return C


def compute_mot_metrics(
    gt_frames: list[dict],
    hyp_frames: list[dict],
    max_iou_dist: float = 0.5,
) -> dict[str, float]:
    """
    MOTA, MOTP, IDF1, IDSW hesaplar.

    Parametreler
    ----------
    gt_frames : list of dict — her frame için:
        {"frame_id": int, "ids": list[int], "boxes": list[BBox]}
    hyp_frames : list of dict — aynı format, hypothesis (tracker çıktısı)
    max_iou_dist : IoU eşiği için maliyet eşiği (1 - IoU_min)

    Döndürür
    --------
    dict: MOTA, MOTP, IDF1, MT (mostly tracked), ML (mostly lost),
          FP, FN, IDSW, num_frames
    """
    if not MM_AVAILABLE:
        raise ImportError("motmetrics kurulu değil: pip install motmetrics")

    acc = mm.MOTAccumulator(auto_id=False)

    gt_by_frame = {f["frame_id"]: f for f in gt_frames}
    hyp_by_frame = {f["frame_id"]: f for f in hyp_frames}
    all_frames = sorted(set(gt_by_frame) | set(hyp_by_frame))

    for fid in all_frames:
        gt = gt_by_frame.get(fid, {"ids": [], "boxes": []})
        hyp = hyp_by_frame.get(fid, {"ids": [], "boxes": []})

        dist = _iou_distance(gt["boxes"], hyp["boxes"])
        # Eşik üstündeki mesafeleri NaN yap → motmetrics bu çiftleri eşleştirmez
        if dist.size > 0:
            dist[dist > max_iou_dist] = np.nan
        acc.update(
            gt["ids"],
            hyp["ids"],
            dist,
            frameid=fid,
        )

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["num_frames", "mota", "motp", "idf1", "idp", "idr",
                 "num_switches", "num_false_positives", "num_misses",
                 "mostly_tracked", "mostly_lost", "num_unique_objects"],
        name="eval",
    )

    row = summary.iloc[0]
    motp_val = float(row["motp"]) if not pd.isna(row["motp"]) else float("nan")
    motp_iou = 1.0 - motp_val if not np.isnan(motp_val) else float("nan")

    return {
        "MOTA": float(row["mota"]),
        "MOTP_dist": motp_val,
        "MOTP_IoU": motp_iou,
        "IDF1": float(row["idf1"]),
        "IDP": float(row["idp"]),
        "IDR": float(row["idr"]),
        "IDSW": int(row["num_switches"]),
        "FP": int(row["num_false_positives"]),
        "FN": int(row["num_misses"]),
        "MT": int(row["mostly_tracked"]),
        "ML": int(row["mostly_lost"]),
        "num_objects": int(row["num_unique_objects"]),
        "num_frames": int(row["num_frames"]),
    }


def format_metrics_table(results: dict[str, float], title: str = "MOT Metrics") -> str:
    lines = [f"{'─'*40}", f" {title}", f"{'─'*40}"]
    fmt = {
        "MOTA": ".3f", "MOTP_IoU": ".3f", "IDF1": ".3f",
        "IDP": ".3f", "IDR": ".3f",
        "IDSW": "d", "FP": "d", "FN": "d",
        "MT": "d", "ML": "d",
    }
    for k, f in fmt.items():
        if k in results:
            lines.append(f"  {k:<12} {results[k]:{f}}")
    lines.append(f"{'─'*40}")
    return "\n".join(lines)
