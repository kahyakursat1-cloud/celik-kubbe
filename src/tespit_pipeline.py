"""
YOLOv11 hava tehdidi tespit pipeline'i (Celik Kubbe).

Kamera karesini alir, YOLOv11 ile gelen tehditleri tespit eder.
Tespit sonuclari radar koordinatina donusturulerek sinyal olarak gonderilir.

Cikti formati (tespit_sinyal):
    list[dict]:
        track_id  : int    — ByteTrack ID
        sinif     : str    — BalisticMissile | Drone | Helicopter | F16 | ...
        guven     : float  — 0..1
        cx        : float  — normalize yatay konum [0..1]
        cy        : float  — normalize dikey konum [0..1]
        w, h      : float  — normalize boyut
"""

import os
import queue
import numpy as np
from typing import Optional
from PySide6.QtCore import QThread, Signal


class TespitPipeline(QThread):
    """
    YOLOv11 + ByteTrack hava tehdit tespit worker'i.
    Model yoksa bos liste dondurur.
    """

    tespit_sinyal = Signal(list)
    log_sinyal    = Signal(str)

    HEDEF_SINIFLARI = {
        "BalisticMissile", "Missile", "Drone", "drone",
        "IHA", "UAV", "Helicopter", "Helikopter",
        "F16", "Aircraft", "FixedWing", "threat",
        "Balistik_Fuze", "Mini_IHA",
    }

    def __init__(
        self,
        model_yolu: Optional[str] = None,
        guven_esik: float = 0.40,
        tracker: str = "bytetrack.yaml",
    ):
        super().__init__()
        self._model_yolu  = model_yolu
        self._guven_esik  = guven_esik
        self._tracker     = tracker
        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._model       = None
        self._model_yuklu = False
        self._dur         = False

    @property
    def model_yuklu(self) -> bool:
        return self._model_yuklu

    def kare_gonder(self, frame: np.ndarray):
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def run(self):
        if self._model_yolu and os.path.isfile(self._model_yolu):
            try:
                from ultralytics import YOLO
                self._model = YOLO(self._model_yolu)
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                self._model(dummy, verbose=False)
                self._model_yuklu = True
                self.log_sinyal.emit(
                    f"[TespitPipeline] YOLOv11 yuklendi: {self._model_yolu}"
                )
            except Exception as e:
                self.log_sinyal.emit(f"[TespitPipeline] Model yuklenemedi: {e}")

        self.log_sinyal.emit(
            f"[TespitPipeline] Hazir — model={'VAR' if self._model_yuklu else 'YOK'}"
        )

        while not self._dur:
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            tespitler = []

            if self._model is not None:
                try:
                    results = self._model.track(
                        frame,
                        tracker=self._tracker,
                        persist=True,
                        conf=self._guven_esik,
                        verbose=False,
                    )
                    for r in results:
                        if r.boxes is None:
                            continue
                        ids   = r.boxes.id
                        clss  = r.boxes.cls
                        confs = r.boxes.conf
                        xywhn = r.boxes.xywhn

                        for i in range(len(clss)):
                            sinif    = r.names[int(clss[i])]
                            guven    = float(confs[i])
                            cx, cy, bw, bh = xywhn[i].tolist()
                            track_id = int(ids[i]) if ids is not None else -(i + 1)
                            tespitler.append({
                                "track_id": track_id,
                                "sinif":    sinif,
                                "guven":    guven,
                                "cx":       cx,
                                "cy":       cy,
                                "w":        bw,
                                "h":        bh,
                            })
                except Exception as e:
                    self.log_sinyal.emit(f"[TespitPipeline] Inference hatasi: {e}")

            self.tespit_sinyal.emit(tespitler)

    def durdur(self):
        self._dur = True
        self.wait(3000)
