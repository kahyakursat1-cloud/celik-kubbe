"""
eval/baselines/bytetrack_wrapper.py — ByteTrack (vanilla, füzyonsuz) wrapper.

Ultralytics ByteTracker'ını doğrudan kullananmak için gerçek video gerekirken,
burada Kalman+IoU-Hungarian kombinasyonu olarak ByteTrack'ın çekirdek
mantığını simüle eden hafif wrapper kullanıyoruz.

ByteTrack'ın temel farkı: düşük güven tespitlerini ayrı bir buffer'da tutar,
IoU ile yüksek güvenli tracklere bağlamayı dener. Sentetik veride bütün
detections high-confidence → pratikte SORT'a yakın davranır.

Referans: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

BBox = tuple[float, float, float, float]


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class _KF2D:
    """Basit 4-state Kalman (x, y, vx, vy) ByteTrack motion model için."""

    def __init__(self, dt: float = 0.1):
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.Q = np.eye(4) * 1.0
        self.R = np.eye(2) * 4.0
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 50.0

    def init(self, cx: float, cy: float) -> None:
        self.x = np.array([[cx], [cy], [0.0], [0.0]])

    def predict(self) -> tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, cx: float, cy: float) -> None:
        z = np.array([[cx], [cy]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


class _ByteTrack:
    def __init__(self, track_id: int, bbox: BBox, dt: float = 0.1):
        self.track_id = track_id
        self.kf = _KF2D(dt)
        x, y, w, h = bbox
        self.kf.init(x + w / 2, y + h / 2)
        self.w, self.h = w, h
        self.age = 0
        self.hits = 1
        self.time_since_update = 0

    def predict(self) -> BBox:
        cx, cy = self.kf.predict()
        return (cx - self.w / 2, cy - self.h / 2, self.w, self.h)

    def update(self, bbox: BBox) -> None:
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        self.kf.update(cx, cy)
        self.w, self.h = w, h
        self.hits += 1
        self.time_since_update = 0

    def bbox(self) -> BBox:
        cx, cy = float(self.kf.x[0, 0]), float(self.kf.x[1, 0])
        return (cx - self.w / 2, cy - self.h / 2, self.w, self.h)


class ByteTrackWrapper:
    """
    ByteTrack vanilla — füzyon yok, sadece kamera detections.
    SORT ile karşılaştırıldığında: daha agresif track buffer (low-conf),
    sentetik veride yüksek-conf only → SORT benzeri.
    """

    name = "ByteTrack"

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
        dt: float = 0.1,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.dt = dt
        self._tracks: list[_ByteTrack] = []
        self._next_id = 1

    def _match(
        self, pred_bboxes: list[BBox], det_bboxes: list[BBox]
    ) -> tuple[list, list, list]:
        if not pred_bboxes or not det_bboxes:
            return [], list(range(len(pred_bboxes))), list(range(len(det_bboxes)))

        cost = 1.0 - np.array([
            [_iou(p, d) for d in det_bboxes]
            for p in pred_bboxes
        ])
        row_ind, col_ind = linear_sum_assignment(cost)
        matched, unmatched_t, unmatched_d = [], set(), set()
        matched_rows, matched_cols = set(), set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= 1.0 - self.iou_threshold:
                matched.append((r, c))
                matched_rows.add(r)
                matched_cols.add(c)
        unmatched_t = [i for i in range(len(pred_bboxes)) if i not in matched_rows]
        unmatched_d = [j for j in range(len(det_bboxes)) if j not in matched_cols]
        return matched, unmatched_t, unmatched_d

    def _update_step(self, det_bboxes: list[BBox]) -> list[tuple[int, BBox]]:
        predicted = [t.predict() for t in self._tracks]
        for t in self._tracks:
            t.age += 1
            t.time_since_update += 1

        matched, unmatched_t, unmatched_d = self._match(predicted, det_bboxes)

        for ti, di in matched:
            self._tracks[ti].update(det_bboxes[di])
        for di in unmatched_d:
            self._tracks.append(_ByteTrack(self._next_id, det_bboxes[di], self.dt))
            self._next_id += 1

        active = []
        for t in self._tracks:
            if t.time_since_update == 0 or t.time_since_update <= self.max_age:
                if t.hits >= self.min_hits:
                    active.append((t.track_id, t.bbox()))
        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= self.max_age]
        return active

    def process(self, det_frames: list[dict]) -> list[dict]:
        self._tracks = []
        self._next_id = 1
        hyp_frames = []
        for f in det_frames:
            active = self._update_step(f["boxes"])
            hyp_frames.append({
                "frame_id": f["frame_id"],
                "ids": [tid for tid, _ in active],
                "boxes": [bbox for _, bbox in active],
            })
        return hyp_frames
