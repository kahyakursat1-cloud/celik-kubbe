"""
eval/comparison_runner.py — Baseline tracker karşılaştırma koşusu.

Aynı senaryoda tüm tracker'ları çalıştırır, MOT metriklerini hesaplar.
Çıktı: Drones makalesi Tablo 3 — tracker × metrik × senaryo.

Kullanım:
    from eval.comparison_runner import ComparisonRunner
    runner = ComparisonRunner(n_seeds=5)
    df = runner.run()
    runner.print_table(df)

    # veya CLI:
    python -m eval.comparison_runner --seeds 5 --out results/comparison.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import compute_mot_metrics
from eval.scenarios import build_all
from eval.simulated_tracker import SimulatedTracker, TrackerConfig
from eval.baselines.sort_wrapper import SORTWrapper, SORT_AVAILABLE
from eval.baselines.deepsort_wrapper import DeepSORTWrapper, DEEPSORT_AVAILABLE
from eval.baselines.bytetrack_wrapper import ByteTrackWrapper
from eval.baselines.imm_kalman import IMMKalmanTracker


# Çelik Kubbe'nin en iyi config'i (ablasyon sonucu: fu1_ka1_xa0)
CELIK_KUBBE_CONFIG = TrackerConfig(fusion_on=True, kalman_on=True, xai_adaptive=False)


def _get_trackers() -> list:
    trackers = []
    trackers.append(("Celik-Kubbe", None))  # Özel: SimulatedTracker ile
    trackers.append(("ByteTrack", ByteTrackWrapper()))
    trackers.append(("IMM-Kalman", IMMKalmanTracker()))
    if SORT_AVAILABLE:
        trackers.append(("SORT", SORTWrapper()))
    if DEEPSORT_AVAILABLE:
        trackers.append(("DeepSORT", DeepSORTWrapper()))
    return trackers


def _run_tracker(
    tracker_name: str,
    tracker_obj,
    scenario: dict,
    seed: int,
) -> dict[str, float]:
    """
    Bir tracker'ı bir senaryoda çalıştır:
    1. Ortak raw detections üret (fusion_on=True, kalman_on=False → ham tespitler)
    2. Tracker bu detections'ı işleyerek track ID'leri üretir
    3. GT ile karşılaştırılır
    """
    # Ortak raw detections (Kalman olmadan, sadece detection)
    raw_cfg = TrackerConfig(fusion_on=True, kalman_on=False, xai_adaptive=False)
    raw_tracker = SimulatedTracker(config=raw_cfg, seed=seed)
    det_frames = raw_tracker.process(scenario["frames"])

    if tracker_name == "Celik-Kubbe":
        # Çelik Kubbe kendi tam pipeline'ını çalıştırır (fusion + Kalman)
        ck_tracker = SimulatedTracker(config=CELIK_KUBBE_CONFIG, seed=seed)
        hyp_frames = ck_tracker.process(scenario["frames"])
    else:
        # Baseline tracker raw detections'ı alır
        hyp_frames = tracker_obj.process(det_frames)

    try:
        result = compute_mot_metrics(scenario["gt_mot"], hyp_frames)
    except Exception as e:
        result = {
            "MOTA": float("nan"), "MOTP_IoU": float("nan"),
            "IDF1": float("nan"), "IDSW": 0, "FP": 0, "FN": 0,
        }
    return result


class ComparisonRunner:
    """
    N_seeds × senaryolar × tracker'lar → pandas DataFrame.
    """

    def __init__(self, n_seeds: int = 5, base_seed: int = 0, verbose: bool = True):
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.verbose = verbose

    def run(self) -> pd.DataFrame:
        trackers = _get_trackers()
        records = []
        total = self.n_seeds * 4 * len(trackers)
        done = 0

        for seed_offset in range(self.n_seeds):
            seed = self.base_seed + seed_offset
            scenarios = build_all(seed=seed)
            for sc in scenarios:
                for tracker_name, tracker_obj in trackers:
                    metrics = _run_tracker(tracker_name, tracker_obj, sc, seed)
                    records.append({
                        "tracker": tracker_name,
                        "scenario": sc["name"],
                        "seed": seed,
                        **{k: v for k, v in metrics.items()
                           if k in ("MOTA", "MOTP_IoU", "IDF1", "IDSW", "FP", "FN")},
                    })
                    done += 1
                    if self.verbose:
                        print(f"  [{done}/{total}] {tracker_name:<14} | "
                              f"{sc['name']:<16} | seed={seed} "
                              f"-> MOTA={metrics.get('MOTA', float('nan')):.3f}")

        return pd.DataFrame(records)

    @staticmethod
    def aggregate(df: pd.DataFrame) -> pd.DataFrame:
        metrics = ["MOTA", "MOTP_IoU", "IDF1", "IDSW", "FP", "FN"]

        def safe_mean(s):
            return s.replace(-np.inf, np.nan).replace(np.inf, np.nan).mean()

        def safe_std(s):
            return s.replace(-np.inf, np.nan).replace(np.inf, np.nan).std()

        agg = (
            df.groupby(["tracker", "scenario"])[metrics]
            .agg([safe_mean, safe_std])
            .reset_index()
        )
        agg.columns = [
            "_".join(c).strip("_")
            .replace("safe_mean", "mean").replace("safe_std", "std")
            if c[1] else c[0]
            for c in agg.columns
        ]
        return agg

    @staticmethod
    def print_table(df: pd.DataFrame) -> None:
        agg = ComparisonRunner.aggregate(df)
        for sc_name in ["single_threat", "multi_threat", "sensor_dropout", "low_snr"]:
            sub = agg[agg["scenario"] == sc_name].sort_values("MOTA_mean", ascending=False)
            print(f"\n{'='*60}")
            print(f"  Senaryo: {sc_name}")
            print(f"{'='*60}")
            print(f"  {'Tracker':<16} {'MOTA':>7} {'IDF1':>7} {'MOTP':>7} {'IDSW':>5}")
            print(f"  {'-'*50}")
            for _, row in sub.iterrows():
                m = row["MOTA_mean"]
                idf1 = row["IDF1_mean"]
                motp = row["MOTP_IoU_mean"]
                idsw = row["IDSW_mean"]
                marker = " <-- Ours" if row["tracker"] == "Celik-Kubbe" else ""
                print(f"  {row['tracker']:<16} {m:>7.3f} {idf1:>7.3f} "
                      f"{motp:>7.3f} {idsw:>5.1f}{marker}")

    @staticmethod
    def to_latex(df: pd.DataFrame, out_path: str | Path) -> None:
        """paper-ready Tablo 3 LaTeX çıktısı."""
        agg = ComparisonRunner.aggregate(df)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        scenarios_order = ["single_threat", "multi_threat", "sensor_dropout", "low_snr"]
        scenario_labels = {
            "single_threat": "Single Threat",
            "multi_threat": "Multi Threat",
            "sensor_dropout": "Sensor Dropout",
            "low_snr": "Low SNR",
        }
        trackers_order = ["Celik-Kubbe", "SORT", "DeepSORT", "ByteTrack", "IMM-Kalman"]

        sc_headers = " & ".join(
            r"\multicolumn{2}{c}{" + scenario_labels[s] + "}"
            for s in scenarios_order
        )
        sub_headers = " & ".join(
            r"\textbf{MOTA} & \textbf{IDF1}"
            for _ in scenarios_order
        )

        rows = []
        for tr in trackers_order:
            sub = agg[agg["tracker"] == tr]
            if sub.empty:
                continue
            is_ours = tr == "Celik-Kubbe"
            label = r"\textbf{Çelik Kubbe (Ours)}" if is_ours else tr.replace("-", r"\textendash ")
            cells = []
            for sc in scenarios_order:
                row = sub[sub["scenario"] == sc]
                if row.empty:
                    cells += ["—", "—"]
                    continue
                mota = row["MOTA_mean"].values[0]
                idf1 = row["IDF1_mean"].values[0]
                mota_s = f"\\textbf{{{mota:.3f}}}" if is_ours else f"{mota:.3f}"
                idf1_s = f"\\textbf{{{idf1:.3f}}}" if is_ours else f"{idf1:.3f}"
                cells += [mota_s, idf1_s]
            rows.append(f"  {label} & " + " & ".join(cells) + r" \\")

        n_sc = len(scenarios_order)
        col_spec = "l" + "cc" * n_sc

        latex = rf"""% comparison_table.tex — auto-generated by eval/comparison_runner.py
\begin{{table}}[htbp]
\centering
\caption{{Baseline tracker comparison: MOTA and IDF1 (mean over 5 seeds).
  Çelik Kubbe uses full sensor fusion + Kalman; baselines receive same raw detections.}}
\label{{tab:comparison}}
\setlength{{\tabcolsep}}{{5pt}}
\begin{{tabular}}{{{col_spec}}}
\toprule
\textbf{{Method}} & {sc_headers} \\
 & {sub_headers} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
        out.write_text(latex, encoding="utf-8")
        print(f"[OK] Karsilastirma LaTeX tablosu -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline tracker karsilastirma")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("docs/figures/comparison_raw.csv"))
    parser.add_argument("--latex", type=Path,
                        default=Path("docs/figures/comparison_table.tex"))
    args = parser.parse_args()

    print(f"Karsilastirma: {len(_get_trackers())} tracker x {args.seeds} seed x 4 senaryo")
    runner = ComparisonRunner(n_seeds=args.seeds, base_seed=args.base_seed)
    df = runner.run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[OK] Ham sonuclar -> {args.out}")
    runner.print_table(df)
    runner.to_latex(df, args.latex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
