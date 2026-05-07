"""
eval/cross_dataset_test.py — Sentetik → Gerçek generalization testi.

Pipeline:
  1. Synthetic-only eğitim (simulated_tracker, Faz 1-3 senaryoları)
  2. Drone-vs-Bird (gerçek) test verisi üzerinde değerlendirme
  3. Domain gap metrikleri: MOTA düşüşü, FP/FN kayması

Makale katkısı: Tablo 4 — cross-dataset generalization.

Sonuç beklentisi (Q1_HAZIRLIK_PLANI.md'den):
  Sentetik MOTA > Hybrid MOTA > DVB-only MOTA
  (domain gap nicelleştirilmiş olacak)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eval.metrics import compute_mot_metrics
from eval.scenarios import ALL_SCENARIOS
from eval.simulated_tracker import SimulatedTracker, TrackerConfig
from eval.statistics import _safe_mean, _safe_std


def _find_dvb_root() -> Path:
    """Dataset klasörünü iç içe yapıda da bulur."""
    base = PROJECT_ROOT / "data" / "external" / "drone-vs-bird"
    if not base.exists():
        return base
    # Mendeley ZIP'i açınca iç içe uzun isimli klasör oluyor
    for sub in base.iterdir():
        if sub.is_dir() and (sub / "Dataset").exists():
            return sub / "Dataset"
        if sub.is_dir() and any((sub / s / "images").exists()
                                for s in ("train", "test", "valid")):
            return sub
    return base


DVB_ROOT = _find_dvb_root()


@dataclass
class CrossDatasetResult:
    condition: str          # "synthetic", "dvb_test", "hybrid"
    scenario_or_split: str
    mota: float
    motp: float
    idf1: float
    idsw: int
    fp: int
    fn: int
    n_frames: int
    n_gt: int


def evaluate_synthetic(
    n_seeds: int = 10,
    tracker_cfg: Optional[TrackerConfig] = None,
) -> list[CrossDatasetResult]:
    """
    Sentetik senaryolarda tracker değerlendirmesi (baseline — domain içi).
    """
    if tracker_cfg is None:
        tracker_cfg = TrackerConfig(fusion_on=True, kalman_on=True)

    results = []
    for builder in ALL_SCENARIOS:
        mota_vals = []
        for seed in range(n_seeds):
            scenario = builder(seed=seed + 200)
            frames = scenario["frames"]
            tracker = SimulatedTracker(tracker_cfg, seed=seed + 200)
            hyp = tracker.process(frames)
            gt = [{"frame_id": f.frame_id, "ids": f.track_ids, "boxes": f.bboxes}
                  for f in frames]
            hyp_fmt = [{"frame_id": h["frame_id"], "ids": h["ids"], "boxes": h["boxes"]}
                       for h in hyp]
            m = compute_mot_metrics(gt, hyp_fmt)
            mota_vals.append(m.get("MOTA", float("nan")))

        results.append(CrossDatasetResult(
            condition="synthetic",
            scenario_or_split=builder.__name__,
            mota=_safe_mean(mota_vals),
            motp=float("nan"),
            idf1=float("nan"),
            idsw=0,
            fp=0,
            fn=0,
            n_frames=0,
            n_gt=0,
        ))
        print(f"  [Syntetik] {builder.__name__}: MOTA={_safe_mean(mota_vals):.3f}")

    return results


def evaluate_dvb(
    split: str = "test",
    tracker_cfg: Optional[TrackerConfig] = None,
    max_frames: Optional[int] = None,
) -> CrossDatasetResult:
    """
    Drone-vs-Bird gerçek verisi üzerinde tracker değerlendirmesi.
    Dataset indirilmemişse uyarı verir.
    """
    if not DVB_ROOT.exists() or not any(DVB_ROOT.rglob("*.jpg")):
        print(f"UYARI: DVB dataset bulunamadi -> {DVB_ROOT}")
        print("       data/external/drone-vs-bird/ altına ZIP'i çıkarın.")
        return CrossDatasetResult(
            condition="dvb_test", scenario_or_split=split,
            mota=float("nan"), motp=float("nan"), idf1=float("nan"),
            idsw=0, fp=0, fn=0, n_frames=0, n_gt=0,
        )

    from data.external.dvb_loader import DVBLoader
    from data.external.dvb_to_synthetic_radar import DVBRadarSynthesizer

    if tracker_cfg is None:
        tracker_cfg = TrackerConfig(
            fusion_on=True, kalman_on=True,
            camera_only_range_m=500.0,   # DVB drone'lar genellikle yakın
        )

    loader = DVBLoader(DVB_ROOT)
    dvb_frames = loader.load_mot_frames(split)
    if max_frames:
        dvb_frames = dvb_frames[:max_frames]

    # Sentetik radar ekle
    synth = DVBRadarSynthesizer()
    radar_frames = synth.synthesize(dvb_frames)

    # GT formatı
    gt_mot = loader.to_mot_format(dvb_frames)

    # DVBFrame → simulated_tracker uyumlu fake GTFrame
    from eval.ground_truth import GTFrame
    fake_gt_frames = []
    for dvb_f, radar_f in zip(dvb_frames, radar_frames):
        fake_gt_frames.append(GTFrame(
            frame_id=dvb_f.frame_id,
            timestamp_s=dvb_f.frame_id * 0.04,
            track_ids=dvb_f.track_ids,
            bboxes=dvb_f.bboxes,
            ranges_m=[m.range_m for m in radar_f],
            radar_detections=[m.detected for m in radar_f],
            classes=dvb_f.classes,
            radar_available=True,
        ))

    tracker = SimulatedTracker(tracker_cfg, seed=0)
    hyp = tracker.process(fake_gt_frames)
    hyp_fmt = [{"frame_id": h["frame_id"], "ids": h["ids"], "boxes": h["boxes"]}
               for h in hyp]

    m = compute_mot_metrics(gt_mot, hyp_fmt)
    mota = m.get("MOTA", float("nan"))
    print(f"  [DVB {split}] MOTA={mota:.3f}, IDSW={m.get('IDSW',0)}, "
          f"FP={m.get('FP',0)}, FN={m.get('FN',0)}")

    return CrossDatasetResult(
        condition="dvb_test",
        scenario_or_split=split,
        mota=float(mota),
        motp=float(m.get("MOTP", float("nan"))),
        idf1=float(m.get("IDF1", float("nan"))),
        idsw=int(m.get("IDSW", 0)),
        fp=int(m.get("FP", 0)),
        fn=int(m.get("FN", 0)),
        n_frames=len(dvb_frames),
        n_gt=sum(len(f.track_ids) for f in dvb_frames),
    )


def domain_gap_table(
    synth_results: list[CrossDatasetResult],
    dvb_result: CrossDatasetResult,
) -> str:
    """
    Makale Tablo 4: domain gap özeti.
    """
    synth_mean = _safe_mean([r.mota for r in synth_results if np.isfinite(r.mota)])
    dvb_mota = dvb_result.mota
    gap = synth_mean - dvb_mota if np.isfinite(dvb_mota) else float("nan")

    lines = [
        "Domain Gap Analysis — Synthetic vs Drone-vs-Bird",
        "=" * 55,
        f"  Synthetic MOTA (mean):  {synth_mean:+.3f}",
        f"  DVB Real MOTA:          {dvb_mota:+.3f}" if np.isfinite(dvb_mota) else
        f"  DVB Real MOTA:          N/A (dataset yuklenmedi)",
        f"  Domain gap (delta):     {gap:+.3f}" if np.isfinite(gap) else
        f"  Domain gap (delta):     N/A",
        "",
        "Senaryo bazlı:",
    ]
    for r in synth_results:
        lines.append(f"  {r.scenario_or_split:<20}: MOTA={r.mota:+.3f}")

    return "\n".join(lines)


def to_latex(
    synth_results: list[CrossDatasetResult],
    dvb_result: CrossDatasetResult,
) -> str:
    """Makale Tablo 4 LaTeX."""
    synth_mean = _safe_mean([r.mota for r in synth_results if np.isfinite(r.mota)])
    dvb_mota = dvb_result.mota
    gap = synth_mean - dvb_mota if np.isfinite(dvb_mota) else float("nan")

    rows = []
    for r in synth_results:
        m = f"{r.mota:.3f}" if np.isfinite(r.mota) else "--"
        rows.append(f"Synthetic & {r.scenario_or_split} & {m} & -- \\\\")

    dvb_m = f"{dvb_mota:.3f}" if np.isfinite(dvb_mota) else "--"
    gap_s = f"{gap:.3f}" if np.isfinite(gap) else "--"
    rows.append(f"Drone-vs-Bird & test & {dvb_m} & {gap_s} \\\\")

    body = "\n".join(rows)
    return rf"""
\begin{{table}}[ht]
\centering
\caption{{Cross-Dataset Generalization (Synthetic $\to$ Drone-vs-Bird).
         Domain gap = $\Delta$MOTA between synthetic in-distribution
         and real out-of-distribution evaluation.}}
\label{{tab:cross_dataset}}
\begin{{tabular}}{{llcc}}
\toprule
Condition & Scenario/Split & MOTA & $\Delta$MOTA \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip()


def run(
    n_seeds: int = 10,
    dvb_split: str = "test",
    max_dvb_frames: Optional[int] = None,
    save_latex: bool = True,
) -> None:
    print("Cross-dataset generalization testi baslatiliyor...\n")

    print("[1/2] Sentetik degerlendirme...")
    synth = evaluate_synthetic(n_seeds=n_seeds)

    print("\n[2/2] Drone-vs-Bird degerlendirme...")
    dvb = evaluate_dvb(split=dvb_split, max_frames=max_dvb_frames)

    print("\n" + domain_gap_table(synth, dvb))

    if save_latex:
        tex = to_latex(synth, dvb)
        out = PROJECT_ROOT / "paper" / "tables" / "cross_dataset_table.tex"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex, encoding="utf-8")
        print(f"\nLaTeX tablosu: {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    run(n_seeds=args.seeds, dvb_split=args.split,
        max_dvb_frames=args.max_frames)
