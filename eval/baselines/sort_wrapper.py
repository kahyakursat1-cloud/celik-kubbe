"""
eval/baselines/sort_wrapper.py — SORT tracker wrapper.

SORT (Simple Online and Realtime Tracking): Kalman + Hungarian.
sort-tracker-py kütüphanesi üzerine GTFrame uyumlu wrapper.

Referans: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016.
"""

from __future__ import annotations

from typing import Optional

try:
    from sort import Sort as _SortImpl
    SORT_AVAILABLE = True
except ImportError:
    SORT_AVAILABLE = False

from eval.ground_truth import GTFrame

BBox = tuple[float, float, float, float]


class SORTWrapper:
    """
    SORT tracker'ı GTFrame listesinden hypothesis frame listesi üretir.

    Input detections: GT bboxes + noise (SimulatedTracker'dan) veya doğrudan GT.
    """

    name = "SORT"

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
    ):
        if not SORT_AVAILABLE:
            raise ImportError("sort-tracker-py kurulu değil: pip install sort-tracker-py")
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def process(self, det_frames: list[dict]) -> list[dict]:
        """
        det_frames: SimulatedTracker.process() çıktısı gibi
            [{"frame_id": int, "ids": [...], "boxes": [(x,y,w,h),...]}]

        SORT'un track ID'leri ile hypothesis döndürür.
        """
        import numpy as np
        tracker = _SortImpl(
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold,
        )
        hyp_frames = []
        for f in det_frames:
            # SORT: [x1, y1, x2, y2, score] formatı
            if f["boxes"]:
                dets = []
                for x, y, w, h in f["boxes"]:
                    dets.append([x, y, x + w, y + h, 0.9])
                dets_np = np.array(dets, dtype=float)
            else:
                dets_np = np.empty((0, 5), dtype=float)

            tracks = tracker.update(dets_np)  # → [[x1,y1,x2,y2,id], ...]

            hyp_ids = []
            hyp_boxes = []
            for t in tracks:
                x1, y1, x2, y2, tid = t
                hyp_ids.append(int(tid))
                hyp_boxes.append((float(x1), float(y1),
                                  float(x2 - x1), float(y2 - y1)))

            hyp_frames.append({
                "frame_id": f["frame_id"],
                "ids": hyp_ids,
                "boxes": hyp_boxes,
            })
        return hyp_frames
