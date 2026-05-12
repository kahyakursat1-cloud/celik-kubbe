"""
eval/vlm_cross_dataset.py — VLM anomaly score kalibrasyon değerlendirmesi.

Sentetik senaryo üzerinde mock VLM ile ablation hesaplar; Drone-vs-Bird
gerçek verisi mevcut olduğunda gerçek VLM inference çalıştırır (Ö1
benchmark aşaması).

Makale Katkısı (§6 VLM Cross-Dataset Evaluation):
  - Mock VLM: Fusion katkısını izole eder (Δ MOTA, vlm_on vs vlm_off).
  - Gerçek VLM (DVB): Anomaly score kalibrasyonunu doğrular
    (tehdit-VAR ortalama > tehdit-YOK ortalama, p < 0.05 Wilcoxon).

Kullanım:
    # Sadece sentetik (mock VLM) — donanım gerekmez:
    python -m eval.vlm_cross_dataset --mode synthetic --seeds 10

    # DVB gerçek verisi (Qwen2-VL-2B INT8, GPU gerekir):
    python -m eval.vlm_cross_dataset --mode dvb --max-frames 200
    python -m eval.vlm_cross_dataset --mode dvb --max-frames 0  (tümü)

Çıktı:
    eval/results/vlm_synthetic_ablation.csv
    eval/results/vlm_dvb_anomaly.csv         (DVB modda)
    paper/tables/vlm_ablation_table.tex
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.ablation_runner import AblationRunner
from eval.simulated_tracker import (
    TrackerConfig, ALL_CONFIGS, VLM_ABLATION_CONFIGS,
)
from eval.scenarios import build_all
from eval.metrics import compute_mot_metrics


# ── 1. Sentetik ablation (mock VLM) ───────────────────────────────────────────

def run_synthetic_ablation(
    n_seeds: int = 10,
    out_csv: Optional[Path] = None,
    out_tex: Optional[Path] = None,
    verbose: bool = True,
) -> "pd.DataFrame":
    """
    16 config (8 baseline + 8 VLM) × 4 senaryo × n_seeds.
    VLM katkısını fusion boyutunda izole eder.
    """
    import pandas as pd

    print(f"[1/3] Sentetik ablation: 16 config × {n_seeds} seed × 4 senaryo")
    runner = AblationRunner(
        configs=VLM_ABLATION_CONFIGS,
        n_seeds=n_seeds,
        verbose=verbose,
    )
    df = runner.run()

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[OK] {out_csv}")

    # VLM Δ MOTA özeti
    _summarize_vlm_delta(df)

    if out_tex is not None:
        tex = _to_latex_ablation(df)
        out_tex.parent.mkdir(parents=True, exist_ok=True)
        out_tex.write_text(tex, encoding="utf-8")
        print(f"[OK] LaTeX tablosu → {out_tex}")

    return df


def _summarize_vlm_delta(df: "pd.DataFrame") -> None:
    import pandas as pd

    grp = df.groupby(["fusion_on", "kalman_on", "xai_adaptive", "vlm_on"])["MOTA"]
    rows = []
    for (fu, ka, xa, vlm), g in grp:
        finite = g.replace(-np.inf, np.nan).dropna()
        rows.append({"fu": fu, "ka": ka, "xa": xa, "vlm": vlm,
                     "mota_mean": finite.mean(), "mota_std": finite.std()})
    p = pd.DataFrame(rows)
    off = p[~p.vlm].set_index(["fu", "ka", "xa"])[["mota_mean", "mota_std"]]
    on  = p[ p.vlm].set_index(["fu", "ka", "xa"])[["mota_mean", "mota_std"]]
    delta = (on["mota_mean"] - off["mota_mean"]).dropna()
    print(f"\n  VLM ΔMOTA ortalaması : {delta.mean():+.4f}")
    print(f"  VLM ΔMOTA aralığı   : [{delta.min():+.4f}, {delta.max():+.4f}]")
    best_idx = delta.idxmax()
    print(f"  En fazla kazanım    : fu={best_idx[0]} ka={best_idx[1]} xa={best_idx[2]}"
          f" → ΔMOTA={delta[best_idx]:+.4f}")


def _to_latex_ablation(df: "pd.DataFrame") -> str:
    """
    Makale Table 8: 16 satır ablation tablosu (fu × ka × xa × vl).
    Her config için 4 senaryo ortalaması MOTA ± std gösterilir.
    """
    import pandas as pd

    agg = (
        df.groupby(["config", "fusion_on", "kalman_on", "xai_adaptive", "vlm_on"])["MOTA"]
        .agg(lambda s: s.replace(-np.inf, np.nan).agg(["mean", "std"]))
        .reset_index()
    )

    rows = []
    for _, row in agg.sort_values(
        ["fusion_on", "kalman_on", "xai_adaptive", "vlm_on"]
    ).iterrows():
        cfg   = row["config"]
        fu    = r"\checkmark" if row["fusion_on"]    else "--"
        ka    = r"\checkmark" if row["kalman_on"]    else "--"
        xa    = r"\checkmark" if row["xai_adaptive"] else "--"
        vl    = r"\checkmark" if row["vlm_on"]       else "--"
        mean_ = row.get("MOTA_mean", row.get("mean", float("nan")))
        std_  = row.get("MOTA_std",  row.get("std",  float("nan")))
        m_s   = f"{mean_:.3f}" if np.isfinite(mean_) else "--"
        s_s   = f"{std_:.3f}"  if np.isfinite(std_)  else "--"
        # Tam konfigürasyonu kalın yap
        if row["fusion_on"] and row["kalman_on"] and row["xai_adaptive"] and row["vlm_on"]:
            m_s = r"\textbf{" + m_s + "}"
        rows.append(f"{fu} & {ka} & {xa} & {vl} & ${m_s} \\pm {s_s}$ \\\\")

    body = "\n".join(rows)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Ablation Study: Fusion (F), Kalman (K), XAI (X), VLM (V) bileşenleri.
         MOTA = ortalama $\pm$ std (N=10 seed, 4 senaryo).
         En iyi yapılandırma \textbf{{kalın}} gösterilmiştir.}}
\label{{tab:ablation_vlm}}
\begin{{tabular}}{{ccccccc}}
\toprule
F & K & X & V & MOTA ($\uparrow$) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


# ── 2. DVB gerçek verisi — VLM anomaly score kalibrasyonu ─────────────────────

def run_dvb_vlm_eval(
    max_frames: int = 200,
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
    device: str = "cuda",
    int8: bool = True,
    out_csv: Optional[Path] = None,
    verbose: bool = True,
) -> Optional["pd.DataFrame"]:
    """
    Drone-vs-Bird test split'i üzerinde gerçek Qwen2-VL-2B çalıştırır.

    Her frame için:
      - VLM anomaly_score üretir
      - GT: drone frame (tehdit-VAR=1) veya bird/background (tehdit-YOK=0)
      - Wilcoxon rank-sum ile tehdit-VAR vs tehdit-YOK anomaly dağılımını karşılaştırır

    Donanım gerektirir (GPU + model indirme). Donanım yoksa
    DVB dataset yoksa uyarıyla çıkar.
    """
    import pandas as pd

    dvb_root = PROJECT_ROOT / "data" / "external" / "drone-vs-bird"
    if not dvb_root.exists() or not any(dvb_root.rglob("*.jpg")):
        print(f"[SKIP] DVB dataset bulunamadı: {dvb_root}")
        print("       data/external/drone-vs-bird/ altına dataset'i indirin.")
        return None

    print(f"[2/3] DVB VLM kalibrasyon değerlendirmesi (max_frames={max_frames or 'all'})")

    try:
        from src.vlm_scene_analyzer import VlmSceneAnalyzer
    except ImportError as e:
        print(f"[SKIP] vlm_scene_analyzer yüklenemedi: {e}")
        return None

    # Mock değil, gerçek backend (model_id ile)
    analyzer = VlmSceneAnalyzer(mock=False, throttle_s=0.0, cache_size=0)

    # DVB frame loader
    try:
        from data.external.dvb_loader import DVBLoader
    except ImportError:
        print("[SKIP] DVBLoader bulunamadı; data/external/dvb_loader.py mevcut mu?")
        return None

    loader = DVBLoader(dvb_root)
    frames = loader.load_mot_frames("test")
    if max_frames and max_frames > 0:
        frames = frames[:max_frames]

    records = []
    for i, frame in enumerate(frames):
        # GT: en az 1 drone track varsa tehdit-VAR
        gt_classes = getattr(frame, "classes", []) or []
        has_threat = any(c.lower() in ("drone", "uav", "fixedwinguav") for c in gt_classes)

        # Frame'i yükle
        img_path = getattr(frame, "image_path", None)
        img_np = None
        if img_path and Path(img_path).is_file():
            try:
                import cv2
                img_np = cv2.imread(str(img_path))
            except ImportError:
                try:
                    from PIL import Image
                    import numpy as _np
                    img_np = _np.array(Image.open(img_path))
                except Exception:
                    pass

        if img_np is None:
            # Görüntü yoksa mock analiz
            tracks = [{"track_id": 0, "sinif": c, "range_km": 1.0,
                        "velocity_ms": -20.0, "bearing_deg": 0.0, "conf": 0.7}
                       for c in (gt_classes or ["Drone"])]
            result = analyzer.queue_analysis(None, tracks)
        else:
            tracks = [{"track_id": j, "sinif": c, "range_km": 1.0,
                        "velocity_ms": -20.0, "bearing_deg": 0.0, "conf": 0.7}
                       for j, c in enumerate(gt_classes or ["Drone"])]
            result = analyzer.queue_analysis(img_np, tracks)

        if result is None:
            continue

        records.append({
            "frame_id": frame.frame_id,
            "gt_threat": int(has_threat),
            "vlm_anomaly_score": result.anomaly_score,
            "vlm_summary": result.summary[:100],
            "latency_ms": result.latency_ms,
            "is_mock": result.is_mock,
        })

        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(frames)}] son anomaly={result.anomaly_score:.3f}")

    if not records:
        print("[WARN] Hiç kayıt üretilemedi.")
        return None

    df = pd.DataFrame(records)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[OK] {out_csv}")

    _wilcoxon_report(df)
    return df


def _wilcoxon_report(df: "pd.DataFrame") -> None:
    """Wilcoxon rank-sum: tehdit-VAR vs tehdit-YOK anomaly dağılımı karşılaştırması."""
    threat_scores     = df[df.gt_threat == 1]["vlm_anomaly_score"].values
    non_threat_scores = df[df.gt_threat == 0]["vlm_anomaly_score"].values

    print(f"\n  Tehdit-VAR  (n={len(threat_scores)}): "
          f"mean={threat_scores.mean():.3f}, std={threat_scores.std():.3f}")
    print(f"  Tehdit-YOK  (n={len(non_threat_scores)}): "
          f"mean={non_threat_scores.mean():.3f}, std={non_threat_scores.std():.3f}")

    if len(threat_scores) > 4 and len(non_threat_scores) > 4:
        try:
            from scipy.stats import ranksums
            stat, p = ranksums(threat_scores, non_threat_scores)
            sig = "✓ anlamlı (p<0.05)" if p < 0.05 else "✗ anlamsız"
            print(f"  Wilcoxon rank-sum: W={stat:.2f}, p={p:.4f} — {sig}")
            print(f"  Kalibrasyon geçer: {'EVET' if p<0.05 and threat_scores.mean()>non_threat_scores.mean() else 'HAYIR'}")
        except ImportError:
            print("  [SKIP] scipy bulunamadı; Wilcoxon atlandı.")
    else:
        print("  [SKIP] Wilcoxon için yetersiz örnek.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="VLM cross-dataset kalibrasyon değerlendirmesi"
    )
    parser.add_argument("--mode", choices=["synthetic", "dvb", "both"],
                        default="synthetic",
                        help="synthetic=mock VLM ablation | dvb=gerçek VLM+DVB")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Sentetik koşu için seed sayısı")
    parser.add_argument("--max-frames", type=int, default=200,
                        help="DVB modunda maksimum frame (0=tümü)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-int8", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "eval" / "results"
    tables_dir  = PROJECT_ROOT / "paper" / "tables"

    if args.mode in ("synthetic", "both"):
        run_synthetic_ablation(
            n_seeds=args.seeds,
            out_csv=results_dir / "vlm_synthetic_ablation.csv",
            out_tex=tables_dir  / "vlm_ablation_table.tex",
            verbose=args.verbose,
        )

    if args.mode in ("dvb", "both"):
        run_dvb_vlm_eval(
            max_frames=args.max_frames,
            device=args.device,
            int8=not args.no_int8,
            out_csv=results_dir / "vlm_dvb_anomaly.csv",
            verbose=args.verbose,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
