"""
eval/baselines/imm_kalman.py — IMM-Kalman tracker (manual implementation).

Interacting Multiple Model (IMM): sabit hız + sabit ivme modellerini
paralel çalıştırıp olasılıksal birleştirme yapar. Manevra eden hedefler
(hava tehditleri) için CV+CA IMM klasik bir referanstır.

Referans: Bar-Shalom et al., "Estimation with Applications to Tracking
and Navigation", Wiley 2001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

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


class KalmanModel:
    """
    Tek model Kalman filtresi: state = [x, y, vx, vy] (CV) veya
    [x, y, vx, vy, ax, ay] (CA).
    """

    def __init__(self, model: str = "CV", dt: float = 0.1,
                 process_noise: float = 5.0, meas_noise: float = 4.0):
        self.model = model
        self.dt = dt
        n = 4 if model == "CV" else 6
        self.n = n
        self.x = np.zeros((n, 1))
        self.P = np.eye(n) * 100.0

        if model == "CV":
            self.F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1,  0],
                [0, 0, 0,  1],
            ], dtype=float)
            self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        else:  # CA
            dt2 = 0.5 * dt**2
            self.F = np.array([
                [1, 0, dt, 0, dt2, 0],
                [0, 1, 0, dt, 0, dt2],
                [0, 0, 1,  0, dt,  0],
                [0, 0, 0,  1,  0, dt],
                [0, 0, 0,  0,  1,  0],
                [0, 0, 0,  0,  0,  1],
            ], dtype=float)
            self.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ], dtype=float)

        q = process_noise**2
        self.Q = np.eye(n) * q
        self.R = np.eye(2) * meas_noise**2

    def init(self, cx: float, cy: float) -> None:
        self.x[:2] = [[cx], [cy]]
        self.P = np.eye(self.n) * 50.0

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, cx: float, cy: float) -> None:
        z = np.array([[cx], [cy]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.n)
        self.P = (I - K @ self.H) @ self.P

    def position(self) -> tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    def likelihood(self, cx: float, cy: float) -> float:
        """Ölçüm olasılığı (Gaussian)."""
        z = np.array([[cx], [cy]])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        det_S = max(np.linalg.det(S), 1e-9)
        exponent = float((-0.5 * y.T @ np.linalg.inv(S) @ y).flat[0])
        return math.exp(exponent) / math.sqrt((2 * math.pi)**2 * det_S)


@dataclass
class IMMTrack:
    track_id: int
    models: list[KalmanModel]
    mu: np.ndarray              # model probabilities [CV, CA]
    w: float = 1.0              # bbox half-size
    h: float = 1.0
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    def predict(self) -> None:
        for m in self.models:
            m.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, cx: float, cy: float, w: float, h: float) -> None:
        likelihoods = np.array([m.likelihood(cx, cy) for m in self.models])
        likelihoods = np.maximum(likelihoods, 1e-300)
        weights = self.mu * likelihoods
        total = weights.sum()
        self.mu = weights / total if total > 0 else self.mu

        for m in self.models:
            m.update(cx, cy)

        self.w, self.h = w, h
        self.hits += 1
        self.time_since_update = 0

    def position(self) -> tuple[float, float]:
        positions = np.array([m.position() for m in self.models])
        x = float(np.dot(self.mu, positions[:, 0]))
        y = float(np.dot(self.mu, positions[:, 1]))
        return x, y

    def bbox(self) -> BBox:
        cx, cy = self.position()
        return (cx - self.w / 2, cy - self.h / 2, self.w, self.h)


class IMMKalmanTracker:
    """
    IMM-Kalman tracker: Sabit Hız (CV) + Sabit İvme (CA) modelleri.
    Manevra eden hava tehditlerini daha iyi yakalar.
    """

    name = "IMM-Kalman"

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
        dt: float = 0.1,
        process_noise: float = 5.0,
        meas_noise: float = 3.0,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.dt = dt
        self.process_noise = process_noise
        self.meas_noise = meas_noise
        self._next_id = 1
        self._tracks: list[IMMTrack] = []

    def _new_track(self, cx: float, cy: float, w: float, h: float) -> IMMTrack:
        models = [
            KalmanModel("CV", self.dt, self.process_noise, self.meas_noise),
            KalmanModel("CA", self.dt, self.process_noise * 1.5, self.meas_noise),
        ]
        for m in models:
            m.init(cx, cy)
        mu = np.array([0.7, 0.3])
        track = IMMTrack(self._next_id, models, mu, w, h)
        self._next_id += 1
        return track

    def _match(self, dets: list[BBox]) -> tuple[list, list, list]:
        if not self._tracks or not dets:
            return [], list(range(len(self._tracks))), list(range(len(dets)))

        cost = np.zeros((len(self._tracks), len(dets)))
        for i, t in enumerate(self._tracks):
            tb = t.bbox()
            for j, d in enumerate(dets):
                cost[i, j] = 1.0 - _iou(tb, d)

        row_ind, col_ind = linear_sum_assignment(cost)
        matched, unmatched_t, unmatched_d = [], [], []
        matched_d = set()
        matched_t = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= 1.0 - self.iou_threshold:
                matched.append((r, c))
                matched_t.add(r)
                matched_d.add(c)
        unmatched_t = [i for i in range(len(self._tracks)) if i not in matched_t]
        unmatched_d = [j for j in range(len(dets)) if j not in matched_d]
        return matched, unmatched_t, unmatched_d

    def update(self, dets: list[BBox]) -> list[tuple[int, BBox]]:
        for t in self._tracks:
            t.predict()

        matched, unmatched_t, unmatched_d = self._match(dets)

        for ti, di in matched:
            x, y, w, h = dets[di]
            cx, cy = x + w / 2, y + h / 2
            self._tracks[ti].update(cx, cy, w, h)

        for di in unmatched_d:
            x, y, w, h = dets[di]
            cx, cy = x + w / 2, y + h / 2
            self._tracks.append(self._new_track(cx, cy, w, h))

        active = []
        for t in self._tracks:
            if t.time_since_update <= self.max_age:
                if t.hits >= self.min_hits or t.age <= self.min_hits:
                    active.append((t.track_id, t.bbox()))
        self._tracks = [t for t in self._tracks
                        if t.time_since_update <= self.max_age]
        return active

    def process(self, det_frames: list[dict]) -> list[dict]:
        self._tracks = []
        self._next_id = 1
        hyp_frames = []
        for f in det_frames:
            active = self.update(f["boxes"])
            hyp_frames.append({
                "frame_id": f["frame_id"],
                "ids": [tid for tid, _ in active],
                "boxes": [bbox for _, bbox in active],
            })
        return hyp_frames
