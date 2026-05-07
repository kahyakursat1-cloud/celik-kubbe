"""
YOLOv11 modelini Jetson Nano / Edge cihazlar için TensorRT (Engine)
formatına dönüştürür. FP16 ve INT8 (kalibrasyon ile) desteklenir.

Kullanım:
    # FP16 (varsayılan, kalibrasyon gerekmez)
    python deployment/export_trt.py --half

    # INT8 (kalibrasyon dataseti gerekli)
    python deployment/export_trt.py --int8 --calib-dir deployment/calib_data

    # Özel model + boyut
    python deployment/export_trt.py --model models/yolo11m_celikkubbe.pt --imgsz 480 --half

NOT: TensorRT engine GPU'ya özeldir. Hedef cihazda (Jetson) çalıştırın,
geliştirme PC'sinde değil — engine taşınamaz.
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_trt")


def build_calib_yaml(calib_dir: Path, dest: Path) -> Path:
    """INT8 kalibrasyon için Ultralytics-uyumlu geçici dataset YAML oluştur."""
    images = sorted([p for p in calib_dir.iterdir()
                     if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if len(images) < 50:
        raise RuntimeError(
            f"INT8 kalibrasyon için en az 50 görüntü gerekir, bulundu: {len(images)} "
            f"({calib_dir}). Önerilen: 100-200 temsili görüntü."
        )

    yaml_text = (
        f"path: {calib_dir.parent.resolve().as_posix()}\n"
        f"train: {calib_dir.name}\n"
        f"val: {calib_dir.name}\n"
        f"names:\n"
    )
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from konfig import cfg
        for i, name in enumerate(cfg["tehditler"]["siniflar"]):
            yaml_text += f"  {i}: {name}\n"
    except Exception:
        for i in range(6):
            yaml_text += f"  {i}: class_{i}\n"

    dest.write_text(yaml_text, encoding="utf-8")
    logger.info(f"Kalibrasyon YAML: {dest} ({len(images)} görüntü)")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv11 → TensorRT Engine")
    parser.add_argument("--model", type=str,
                        default="models/yolo11m_celikkubbe.pt",
                        help="PyTorch model yolu (proje köküne göreli)")
    parser.add_argument("--imgsz", type=int, default=640, help="Görüntü boyutu")
    parser.add_argument("--half", action="store_true",
                        help="FP16 optimizasyonu (Jetson için önerilir)")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 quantization (--calib-dir gerekir)")
    parser.add_argument("--calib-dir", type=str, default=None,
                        help="INT8 kalibrasyon görüntü dizini (50-200 temsili görüntü)")
    parser.add_argument("--workspace", type=int, default=4,
                        help="TensorRT workspace boyutu (GB)")
    parser.add_argument("--device", type=int, default=0, help="GPU id")
    args = parser.parse_args()

    if args.int8 and args.half:
        logger.error("--int8 ve --half birlikte kullanılamaz; INT8 zaten düşük precision")
        return 2
    if args.int8 and not args.calib_dir:
        logger.error("--int8 için --calib-dir gerekli (50-200 görüntülük temsili dataset)")
        return 2

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / args.model
    if not model_path.is_file():
        logger.error(f"Model bulunamadı: {model_path}")
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics kurulu değil: pip install ultralytics")
        return 1

    export_kwargs: dict = dict(
        format="engine",
        imgsz=args.imgsz,
        workspace=args.workspace,
        device=args.device,
    )

    if args.int8:
        calib_dir = Path(args.calib_dir)
        if not calib_dir.is_dir():
            logger.error(f"Kalibrasyon dizini yok: {calib_dir}")
            return 1
        calib_yaml = project_root / "deployment" / "_calib_dataset.yaml"
        try:
            build_calib_yaml(calib_dir, calib_yaml)
        except RuntimeError as e:
            logger.error(str(e))
            return 1
        export_kwargs["int8"] = True
        export_kwargs["data"] = str(calib_yaml)
        precision = "INT8"
    elif args.half:
        export_kwargs["half"] = True
        precision = "FP16"
    else:
        precision = "FP32"

    logger.info(f"Model: {model_path}")
    logger.info(f"Hedef precision: {precision} | imgsz={args.imgsz} | workspace={args.workspace}GB")
    logger.info("Bu işlem Jetson Nano üzerinde 15-20 dakika sürebilir...")

    try:
        model = YOLO(str(model_path))
        export_path = model.export(**export_kwargs)
        logger.info(f"BAŞARILI: {export_path}")
        logger.info(
            f"config.yaml içindeki model.yolu değerini güncelleyin: "
            f"models/{Path(export_path).name}"
        )
        return 0
    except Exception as e:
        logger.error(f"Dönüşüm hatası: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
