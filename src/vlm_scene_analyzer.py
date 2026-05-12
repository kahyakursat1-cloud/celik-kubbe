"""
vlm_scene_analyzer.py — Yerel Vision-Language Model entegrasyonu.

ContextFusion'a 4. evidence kanalı (e_vlm = anomaly_score) ve operatöre
doğal dil tehdit brifi sağlar. Tasarım hedefleri:

  • Cloud-bağımsız: Qwen2-VL-2B INT8 (~2.5 GB) yerel inference.
  • Async: PySide6 QThread; ana GUI/fusion loop'u bloklamaz.
  • Graceful fallback: transformers/bitsandbytes yoksa veya CUDA yoksa
    deterministik "mock" çıktısı üretir (sentetik MC ablation için
    fusion-katkısını ölçmeye yarar; gerçek VLM kalitesi cross-dataset
    değerlendirmesinde ayrı ölçülür).
  • Throttle + cache: Aynı frame'i tekrar analiz etmez.

Tipik kullanım:
    analyzer = VlmSceneAnalyzer(model_id="Qwen/Qwen2-VL-2B-Instruct",
                                int8=True, mock=False)
    analyzer.analyze_frame_signal.connect(gui_handler)
    analyzer.queue_analysis(frame_np, tracks_payload)
"""

from __future__ import annotations

import json
import math
import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from PySide6.QtCore import QObject, QThread, Signal
    PYSIDE_AVAILABLE = True
except ImportError:
    # CLI / pytest ortamı için stub
    PYSIDE_AVAILABLE = False

    class QObject:  # type: ignore
        def __init__(self, *_, **__): pass

    class QThread:  # type: ignore
        def __init__(self, *_, **__): pass
        def start(self): pass

    class Signal:  # type: ignore
        def __init__(self, *_): pass
        def emit(self, *_): pass
        def connect(self, *_): pass


logger = logging.getLogger("vlm_scene_analyzer")


# ── Çıktı veri yapısı ──────────────────────────────────────────────────
@dataclass
class VlmAnalysis:
    """Bir frame için VLM analiz çıktısı."""
    summary: str = ""
    per_track: dict[int, str] = field(default_factory=dict)
    anomaly_score: float = 0.5   # neutral prior
    latency_ms: int = 0
    model_id: str = "mock"
    timestamp: float = 0.0
    is_mock: bool = False
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "per_track": {str(k): v for k, v in self.per_track.items()},
            "anomaly_score": round(self.anomaly_score, 3),
            "latency_ms": self.latency_ms,
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "is_mock": self.is_mock,
            "error": self.error,
        }


# ── Mock VLM (transformers yok / CUDA yok / test) ──────────────────────
def _mock_analyze(tracks_payload: list[dict]) -> VlmAnalysis:
    """
    Sentetik MC ve test için deterministik mock çıktı.

    Stratejisi: track sınıfı + kinetiğine göre kural-tabanlı anomaly_score
    ve şablon-tabanlı brief üretir. Gerçek VLM'in ideal davranışını taklit
    eder (mükemmel sınıflandırma + payload duyarlılığı). Ablation'da fusion
    katkısını izole etmek için kullanılır; cross-dataset VLM kalitesi
    AYRICA ölçülür.
    """
    if not tracks_payload:
        return VlmAnalysis(
            summary="No active tracks in scene.",
            anomaly_score=0.5,
            model_id="mock",
            is_mock=True,
            timestamp=time.time(),
        )

    sınıf_anomali = {
        "BalisticMissile": 0.95,
        "Missile": 0.92,
        "Drone": 0.55,
        "UAV": 0.55,
        "FixedWingUAV": 0.50,
        "Helicopter": 0.35,
        "Aircraft": 0.30,
        "Jet": 0.40,
    }
    skorlar = [sınıf_anomali.get(t.get("sinif", ""), 0.50) for t in tracks_payload]
    # Çoklu yüksek-anomali → korelasyonlu yükseliş
    anomaly_score = max(skorlar)
    if sum(1 for s in skorlar if s > 0.6) >= 2:
        anomaly_score = min(0.98, anomaly_score + 0.05)

    n = len(tracks_payload)
    if n == 1:
        özet = f"Single target tracked at {tracks_payload[0].get('range_km', 0):.1f} km."
    elif n <= 3:
        özet = f"{n} simultaneous targets; mixed-class formation under tracking."
    else:
        özet = f"{n} targets — multi-threat density; potential coordinated approach."

    per_track = {}
    for t in tracks_payload:
        sınıf = t.get("sinif", "Unknown")
        mesafe = t.get("range_km", 0.0)
        hız = t.get("velocity_ms", 0.0)
        yaklaşıyor = "approaching" if hız < -5 else ("static" if abs(hız) < 5 else "departing")
        per_track[t.get("track_id", 0)] = (
            f"{sınıf} at {mesafe:.1f} km, {yaklaşıyor} "
            f"({abs(hız):.0f} m/s); no payload indicators visible in mock mode."
        )

    return VlmAnalysis(
        summary=özet,
        per_track=per_track,
        anomaly_score=anomaly_score,
        latency_ms=5,  # mock = ~5ms deterministic
        model_id="mock",
        is_mock=True,
        timestamp=time.time(),
    )


# ── Gerçek VLM (Qwen2-VL-2B, INT8) ─────────────────────────────────────
class _RealVlmBackend:
    """
    HuggingFace transformers + bitsandbytes INT8 ile Qwen2-VL-2B yükler.
    Lazy load: ilk analyze() çağrısında modeli belleğe alır.
    """

    def __init__(self, model_id: str, device: str, int8: bool, prompt_template_path: str):
        self.model_id = model_id
        self.device = device
        self.int8 = int8
        self.prompt_template_path = prompt_template_path
        self._model = None
        self._processor = None
        self._prompt_template: Optional[str] = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        # Lazy imports — bağımlılıklar yoksa import-time crash etme
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        try:
            from transformers import BitsAndBytesConfig
            qcfg = BitsAndBytesConfig(load_in_8bit=True) if self.int8 else None
        except ImportError:
            qcfg = None

        logger.info(f"VLM yükleniyor: {self.model_id} (INT8={self.int8}, device={self.device})")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=qcfg,
            device_map=self.device,
            torch_dtype="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._prompt_template = Path(self.prompt_template_path).read_text(encoding="utf-8")

    def analyze(self, frame_np, tracks_payload: list[dict]) -> VlmAnalysis:
        from PIL import Image
        import numpy as np

        self._ensure_loaded()
        start = time.time()

        # Prompt rendering
        tracks_json = json.dumps(tracks_payload, ensure_ascii=False, indent=2)
        prompt = self._prompt_template.replace("{tracks_json}", tracks_json)

        # Frame → PIL
        if hasattr(frame_np, "shape"):
            pil_img = Image.fromarray(frame_np)
        else:
            pil_img = frame_np

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
        ).to(self.device)

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,  # temperature=0 deterministic
        )
        text = self._processor.decode(outputs[0], skip_special_tokens=True)
        latency = int((time.time() - start) * 1000)

        # JSON parse (hallucination mitigation: fail-soft)
        try:
            # Bazı modeller JSON'u ```json fence içinde döndürür
            text_clean = text.strip()
            if "```" in text_clean:
                text_clean = text_clean.split("```")[1]
                if text_clean.startswith("json"):
                    text_clean = text_clean[4:]
            parsed = json.loads(text_clean)
            summary = parsed.get("summary", "")
            per_track = {int(k): v for k, v in parsed.get("tracks", {}).items()}
            anomaly = float(parsed.get("anomaly_score", 0.5))
            anomaly = max(0.0, min(1.0, anomaly))
            return VlmAnalysis(
                summary=summary,
                per_track=per_track,
                anomaly_score=anomaly,
                latency_ms=latency,
                model_id=self.model_id,
                timestamp=time.time(),
                is_mock=False,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"VLM JSON parse hatası: {e}; ham metin: {text[:200]}")
            return VlmAnalysis(
                summary=text[:200],
                anomaly_score=0.5,
                latency_ms=latency,
                model_id=self.model_id,
                timestamp=time.time(),
                error=f"JSON parse failed: {e}",
                is_mock=False,
            )


# ── Ana sınıf (PySide6 entegre) ────────────────────────────────────────
class VlmSceneAnalyzer(QObject):
    """
    Async VLM analyzer. queue_analysis() ile çağrılır, sonuç sinyalle döner.

    Args:
        model_id: HuggingFace model adı (örn. "Qwen/Qwen2-VL-2B-Instruct")
        device: "cuda" | "cpu" | "auto"
        int8: bitsandbytes INT8 quantization
        prompt_template: prompt dosyası yolu
        mock: True → backend yüklenmez, deterministik mock kullanılır
        cache_size: LRU cache boyutu (aynı frame_hash + tracks_hash)
        throttle_s: aynı tracks için minimum tetikleme aralığı
    """

    analyze_frame_signal = Signal(object)  # VlmAnalysis
    log_signal = Signal(str)

    def __init__(self,
                 model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
                 device: str = "cuda",
                 int8: bool = True,
                 prompt_template: str = "src/prompts/vlm_threat_brief.txt",
                 mock: bool = False,
                 cache_size: int = 64,
                 throttle_s: float = 2.0,
                 parent=None):
        super().__init__(parent)
        self.model_id = model_id
        self.device = device
        self.int8 = int8
        self.mock = mock
        self.throttle_s = throttle_s
        self._cache: dict[str, VlmAnalysis] = {}
        self._cache_size = cache_size
        self._son_cagri: float = 0.0
        self._backend: Optional[_RealVlmBackend] = None
        if not mock:
            try:
                self._backend = _RealVlmBackend(model_id, device, int8, prompt_template)
            except Exception as e:
                logger.warning(f"VLM backend init başarısız: {e}; mock'a düşülüyor.")
                self.mock = True
                self._backend = None

    def _track_hash(self, tracks_payload: list[dict]) -> str:
        """Track listesi için deterministik hash (cache key parçası)."""
        key = json.dumps(
            [(t.get("track_id"), round(t.get("range_km", 0), 1),
              round(t.get("velocity_ms", 0), 1), t.get("sinif", ""))
             for t in tracks_payload],
            sort_keys=True,
        )
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _frame_hash(self, frame_np) -> str:
        """Frame için kaba hash (downsample sonrası bytes)."""
        if frame_np is None:
            return "noframe"
        try:
            import numpy as np
            small = frame_np[::8, ::8] if hasattr(frame_np, "shape") else frame_np
            return hashlib.md5(small.tobytes()).hexdigest()[:16]
        except Exception:
            return "nohash"

    def queue_analysis(self, frame_np, tracks_payload: list[dict]) -> Optional[VlmAnalysis]:
        """
        Senkron çağrı (testler/CLI için). Throttle + cache kontrolü yapar,
        eligible'sa analiz çalıştırır ve sonucu hem döndürür hem sinyal yayar.

        Production'da GUI thread'inden bu çağrı QThread içinden yapılmalı.
        """
        simdi = time.time()
        # Throttle
        if simdi - self._son_cagri < self.throttle_s:
            return None
        # Cache key
        cache_key = self._frame_hash(frame_np) + ":" + self._track_hash(tracks_payload)
        if cache_key in self._cache:
            sonuc = self._cache[cache_key]
            self.analyze_frame_signal.emit(sonuc)
            return sonuc

        # Analyze
        self._son_cagri = simdi
        if self.mock or self._backend is None:
            sonuc = _mock_analyze(tracks_payload)
        else:
            try:
                sonuc = self._backend.analyze(frame_np, tracks_payload)
            except Exception as e:
                logger.error(f"VLM analiz hatası: {e}; mock'a düşülüyor.")
                sonuc = _mock_analyze(tracks_payload)
                sonuc.error = str(e)

        # Cache (basit LRU: boyut aşılırsa en eski 1 öğeyi at)
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = sonuc
        self.analyze_frame_signal.emit(sonuc)
        return sonuc

    def reset_cache(self):
        self._cache.clear()
        self._son_cagri = 0.0


# ── CLI test girişi ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="Mock backend kullan")
    ap.add_argument("--tracks-json", type=str, help="Test track JSON dosyası")
    args = ap.parse_args()

    tracks = []
    if args.tracks_json:
        tracks = json.loads(Path(args.tracks_json).read_text(encoding="utf-8"))
    else:
        # Default örnek
        tracks = [
            {"track_id": 1, "sinif": "Drone", "range_km": 2.0, "velocity_ms": -25.0},
            {"track_id": 2, "sinif": "BalisticMissile", "range_km": 4.5, "velocity_ms": -120.0},
        ]

    analyzer = VlmSceneAnalyzer(mock=args.mock or True)
    sonuc = analyzer.queue_analysis(None, tracks)
    print(json.dumps(sonuc.to_dict(), indent=2, ensure_ascii=False))
