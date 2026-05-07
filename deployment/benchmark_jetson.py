"""
benchmark_jetson.py — Edge cihaz inference performans ölçümü.

Verilen YOLOv11 modeli (.pt veya .engine) ile gerçek bir kamera/sentetik
girdi üzerinde ardışık inference çalıştırır; FPS, latency dağılımı ve GPU
bellek kullanımını JSON olarak yazdırır.

Kullanım:
    # TensorRT engine (Jetson'da önerilir)
    python deployment/benchmark_jetson.py --model models/yolo11m_celikkubbe.engine

    # Kamera yerine sentetik girdi
    python deployment/benchmark_jetson.py --model ... --synthetic --frames 200

    # Belirli kamera
    python deployment/benchmark_jetson.py --model ... --cam-index 0 --frames 300

Çıktı: stdout'a JSON + opsiyonel --output dosyası.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def gpu_info() -> dict:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"cuda": False}
        return {
            "cuda": True,
            "device_name": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory / 1024 / 1024,
        }
    except ImportError:
        return {"cuda": False, "torch": "missing"}


def synthetic_frames(n: int, h: int, w: int) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def camera_frames(n: int, cam_index: int, h: int, w: int) -> list[np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera açılamadı: index={cam_index}")
    frames: list[np.ndarray] = []
    while len(frames) < n:
        ret, f = cap.read()
        if ret and f is not None:
            frames.append(f)
    cap.release()
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Çelik Kubbe inference benchmark")
    parser.add_argument("--model", type=str, required=True,
                        help="Model dosyası (.pt veya .engine)")
    parser.add_argument("--frames", type=int, default=200,
                        help="Inference frame sayısı (warm-up hariç)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cam-index", type=int, default=0)
    parser.add_argument("--synthetic", action="store_true",
                        help="Kamera yerine rastgele görüntü kullan")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=str, default=None,
                        help="Sonucu JSON olarak yaz")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[FAIL] Model bulunamadı: {model_path}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[FAIL] ultralytics kurulu değil", file=sys.stderr)
        return 1

    print(f"Model yükleniyor: {model_path}")
    model = YOLO(str(model_path))

    total_frames = args.frames + args.warmup
    if args.synthetic:
        print(f"Sentetik {total_frames} frame üretiliyor ({args.imgsz}x{args.imgsz})...")
        frames = synthetic_frames(total_frames, args.imgsz, args.imgsz)
        source_label = "synthetic"
    else:
        print(f"Kameradan {total_frames} frame yakalanıyor (index={args.cam_index})...")
        try:
            frames = camera_frames(total_frames, args.cam_index, args.imgsz, args.imgsz)
        except RuntimeError as e:
            print(f"[FAIL] {e}", file=sys.stderr)
            return 1
        source_label = f"camera_{args.cam_index}"

    print(f"Warm-up ({args.warmup} frame)...")
    for f in frames[:args.warmup]:
        model.predict(f, imgsz=args.imgsz, conf=args.conf, verbose=False)

    print(f"Benchmark ({args.frames} frame)...")
    latencies_ms: list[float] = []
    t_start = time.perf_counter()
    for f in frames[args.warmup:]:
        t0 = time.perf_counter()
        model.predict(f, imgsz=args.imgsz, conf=args.conf, verbose=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    t_total = time.perf_counter() - t_start

    fps = args.frames / t_total
    p50 = statistics.median(latencies_ms)
    p95 = statistics.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 20 else max(latencies_ms)
    p99 = statistics.quantiles(latencies_ms, n=100)[98] if len(latencies_ms) >= 100 else max(latencies_ms)

    result = {
        "model": str(model_path),
        "model_format": model_path.suffix.lstrip("."),
        "frames": args.frames,
        "imgsz": args.imgsz,
        "source": source_label,
        "fps": round(fps, 2),
        "total_seconds": round(t_total, 3),
        "latency_ms": {
            "min": round(min(latencies_ms), 2),
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "max": round(max(latencies_ms), 2),
            "mean": round(statistics.mean(latencies_ms), 2),
            "stdev": round(statistics.stdev(latencies_ms), 2) if len(latencies_ms) > 1 else 0.0,
        },
        "gpu": gpu_info(),
    }

    print()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
        print(f"\n→ Sonuç yazıldı: {args.output}")

    target_fps = 25.0 if model_path.suffix == ".engine" else 8.0
    if fps < target_fps:
        print(f"\n[WARN] FPS hedefin altında (hedef >{target_fps}, ölçüm {fps:.1f})",
              file=sys.stderr)
        return 1
    print(f"\n[OK] Hedef FPS karşılandı (>{target_fps})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
