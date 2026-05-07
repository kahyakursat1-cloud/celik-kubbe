"""
eval/baselines/deepsort_wrapper.py — DeepSORT tracker wrapper.

DeepSORT = SORT + appearance feature embedding (Re-ID).
Sentetik veride gerçek görüntü yok → rastgele birim vektör kullanılır
(appearance etkisi sıfır → pratikte SORT'a eşdeğer, fakat parametre farkı var).

Referans: Wojke et al., "Simple Online and Realtime Tracking with a Deep
Association Metric", ICIP 2017.
"""

from __future__ import annotations

import numpy as np

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort as _DeepSortImpl
    DEEPSORT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    DEEPSORT_AVAILABLE = False
    _DeepSortImpl = None


class DeepSORTWrapper:
    """
    DeepSORT wrapper. Sentetik veride appearance embedding olmaması nedeniyle
    random embedding kullanılır — motion-only DeepSORT'u simüle eder.
    """

    name = "DeepSORT"

    def __init__(
        self,
        max_age: int = 5,
        n_init: int = 1,
        max_iou_distance: float = 0.7,
        embedder: str | None = None,
        embedding_dim: int = 128,
    ):
        if not DEEPSORT_AVAILABLE:
            raise ImportError(
                "deep-sort-realtime kurulu değil: pip install deep-sort-realtime"
            )
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(0)

    def process(self, det_frames: list[dict]) -> list[dict]:
        """
        det_frames: [{"frame_id", "ids", "boxes": [(x,y,w,h),...]}]
        """
        # embedder=None → pkg_resources import'ından kaçın (Python 3.14 uyumlu)
        tracker = _DeepSortImpl(
            max_age=self.max_age,
            n_init=self.n_init,
            max_iou_distance=self.max_iou_distance,
            embedder=None,
        )
        hyp_frames = []
        for f in det_frames:
            if f["boxes"]:
                raw_dets = [([x, y, w, h], 0.9, 0) for x, y, w, h in f["boxes"]]
                embeds = [
                    self._rng.standard_normal(self.embedding_dim).astype(np.float32)
                    for _ in raw_dets
                ]
                for e in embeds:
                    norm = np.linalg.norm(e)
                    if norm > 0:
                        e /= norm
                tracks = tracker.update_tracks(raw_dets, embeds=embeds)
            else:
                tracks = tracker.update_tracks([], embeds=[])

            hyp_ids = []
            hyp_boxes = []
            for t in tracks:
                if not t.is_confirmed():
                    continue
                tid = t.track_id
                ltrb = t.to_ltrb()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = ltrb
                hyp_ids.append(int(tid))
                hyp_boxes.append((float(x1), float(y1),
                                  float(x2 - x1), float(y2 - y1)))

            hyp_frames.append({
                "frame_id": f["frame_id"],
                "ids": hyp_ids,
                "boxes": hyp_boxes,
            })
        return hyp_frames
