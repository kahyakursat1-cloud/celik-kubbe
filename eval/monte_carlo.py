"""
eval/monte_carlo.py — N=50 Monte Carlo değerlendirme.

Her iterasyonda:
  - Rastgele seed, parametre pertürbasyon
  - 4 senaryo × 5 tracker
  - MOTA/MOTP/IDF1/IDSW metrikleri

Çıktı: CSV + pickle (eval/results/monte_carlo_results.csv)
"""

from __future__ import annotations

import csv
import itertools
import pickle
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

from eval.scenarios import single_threat, multi_threat, sensor_dropout, low_snr
from eval.simulated_tracker import SimulatedTracker, TrackerConfig
from eval.metrics import compute_mot_metrics
from eval.baselines.sort_wrapper import SORTWrapper
from eval.baselines.deepsort_wrapper import DeepSORTWrapper
from eval.baselines.bytetrack_wrapper import ByteTrackWrapper
from eval.baselines.imm_kalman import IMMKalmanTracker as IMMKalmanWrapper


RESULTS_DIR = Path(__file__).parent.parent / "eval" / "results"
SCENARIO_BUILDERS = {
    "single_threat": single_threat,
    "multi_threat": multi_threat,
    "sensor_dropout": sensor_dropout,
    "low_snr": low_snr,
}


@dataclass
class MCResult:
    mc_seed: int
    scenario: str
    tracker: str
    mota: float
    motp: float
    idf1: float
    idsw: int
    fp: int
    fn: int
    n_gt: int
    elapsed_s: float


def _run_celik_kubbe(scenario: dict, mc_seed: int) -> dict:
    frames = scenario["frames"]
    cfg = TrackerConfig(fusion_on=True, kalman_on=True, xai_adaptive=False)
    tracker = SimulatedTracker(cfg, seed=mc_seed)
    hyp = tracker.process(frames)
    gt = [{"frame_id": f.frame_id, "ids": f.track_ids, "boxes": f.bboxes}
          for f in frames]
    hyp_fmt = [{"frame_id": f["frame_id"], "ids": f["ids"], "boxes": f["boxes"]}
               for f in hyp]
    return compute_mot_metrics(gt, hyp_fmt)


def _run_tracker_safe(name: str, scenario: dict, mc_seed: int) -> dict:
    try:
        frames = scenario["frames"]
        gt_frames = [{"frame_id": f.frame_id, "ids": f.track_ids, "boxes": f.bboxes}
                     for f in frames]
        cfg_raw = TrackerConfig(fusion_on=True, kalman_on=False, xai_adaptive=False)
        raw_tracker = SimulatedTracker(cfg_raw, seed=mc_seed)
        det_frames = raw_tracker.process(frames)

        if name == "SORT":
            wrapper = SORTWrapper()
        elif name == "DeepSORT":
            wrapper = DeepSORTWrapper()
        elif name == "ByteTrack":
            wrapper = ByteTrackWrapper()
        elif name == "IMM-Kalman":
            wrapper = IMMKalmanWrapper()
        else:
            raise ValueError(f"Bilinmeyen tracker: {name}")

        hyp = wrapper.process(det_frames)
        hyp_fmt = [{"frame_id": f["frame_id"], "ids": f["ids"], "boxes": f["boxes"]}
                   for f in hyp]
        return compute_mot_metrics(gt_frames, hyp_fmt)
    except Exception as e:
        return {"MOTA": float("nan"), "MOTP": float("nan"), "IDF1": float("nan"),
                "IDSW": 0, "FP": 0, "FN": 0, "num_objects": 0, "_error": str(e)}


def run_monte_carlo(
    n_iterations: int = 50,
    seed_offset: int = 1000,
    verbose: bool = True,
) -> list[MCResult]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    trackers = {
        "CelikKubbe": None,
        "SORT": SORTWrapper,
        "DeepSORT": DeepSORTWrapper,
        "ByteTrack": ByteTrackWrapper,
        "IMM-Kalman": IMMKalmanWrapper,
    }

    results = []
    total = n_iterations * len(SCENARIO_BUILDERS) * len(trackers)
    done = 0

    for mc_iter in range(n_iterations):
        mc_seed = seed_offset + mc_iter

        for scenario_name, builder in SCENARIO_BUILDERS.items():
            scenario = builder(seed=mc_seed)

            for tracker_name in trackers:
                t0 = time.perf_counter()
                if tracker_name == "CelikKubbe":
                    m = _run_celik_kubbe(scenario, mc_seed)
                else:
                    m = _run_tracker_safe(tracker_name, scenario, mc_seed)
                elapsed = time.perf_counter() - t0

                mota = m.get("MOTA", float("nan"))
                if mota != mota:
                    mota = float("nan")

                results.append(MCResult(
                    mc_seed=mc_seed,
                    scenario=scenario_name,
                    tracker=tracker_name,
                    mota=float(mota) if np.isfinite(mota) else float("nan"),
                    motp=float(m.get("MOTP", float("nan"))),
                    idf1=float(m.get("IDF1", float("nan"))),
                    idsw=int(m.get("IDSW", 0)),
                    fp=int(m.get("FP", 0)),
                    fn=int(m.get("FN", 0)),
                    n_gt=int(m.get("num_objects", 0)),
                    elapsed_s=elapsed,
                ))

                done += 1
                if verbose and done % 20 == 0:
                    pct = 100 * done / total
                    print(f"  [{pct:5.1f}%] iter={mc_iter+1}/{n_iterations} "
                          f"{scenario_name}/{tracker_name}: MOTA={mota:.3f}")

    _save_results(results)
    return results


def _save_results(results: list[MCResult]) -> None:
    csv_path = RESULTS_DIR / "monte_carlo_results.csv"
    pkl_path = RESULTS_DIR / "monte_carlo_results.pkl"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if not results:
            return
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)

    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    print(f"  Sonuclar: {csv_path}")
    print(f"  Pickle  : {pkl_path}")


def load_results(path: Optional[Path] = None) -> list[MCResult]:
    """Pickle'dan yükle; başarısız olursa CSV'ye düş."""
    p = path or (RESULTS_DIR / "monte_carlo_results.pkl")
    try:
        import pickle as _pkl

        class _Unpickler(_pkl.Unpickler):
            def find_class(self, module, name):
                if name == "MCResult":
                    return MCResult
                return super().find_class(module, name)

        with open(p, "rb") as f:
            return _Unpickler(f).load()
    except Exception:
        return load_results_csv()


def load_results_csv(path: Optional[Path] = None) -> list[MCResult]:
    """CSV'den MCResult listesi oluştur."""
    import csv
    p = path or (RESULTS_DIR / "monte_carlo_results.csv")
    results = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results.append(MCResult(
                mc_seed=int(row["mc_seed"]),
                scenario=row["scenario"],
                tracker=row["tracker"],
                mota=float(row["mota"]) if row["mota"] not in ("nan", "") else float("nan"),
                motp=float(row["motp"]) if row["motp"] not in ("nan", "") else float("nan"),
                idf1=float(row["idf1"]) if row["idf1"] not in ("nan", "") else float("nan"),
                idsw=int(float(row["idsw"])),
                fp=int(float(row["fp"])),
                fn=int(float(row["fn"])),
                n_gt=int(float(row["n_gt"])),
                elapsed_s=float(row["elapsed_s"]),
            ))
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=1000)
    args = parser.parse_args()

    print(f"Monte Carlo N={args.n} baslatiliyor...")
    results = run_monte_carlo(n_iterations=args.n, seed_offset=args.seed_offset)
    print(f"\nToplam {len(results)} sonuc kaydedildi.")
