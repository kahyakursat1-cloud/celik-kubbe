"""
eval/ablation_runner.py — 8 config × N senaryo ablasyon koşusu.

fusion_on × kalman_on × xai_adaptive = 8 hücre.
Her hücre için MOTA/MOTP/IDF1 hesaplanır.

Kullanım:
    from eval.ablation_runner import AblationRunner
    runner = AblationRunner(n_seeds=5)
    table = runner.run()
    print(table.to_string())

    # veya CLI:
    python -m eval.ablation_runner --seeds 10 --out results/ablation.csv
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

from eval.simulated_tracker import (
    SimulatedTracker, TrackerConfig, ALL_CONFIGS, VLM_ABLATION_CONFIGS,
)
from eval.scenarios import build_all
from eval.metrics import compute_mot_metrics


def _run_one(config: TrackerConfig, scenario: dict, seed: int) -> dict[str, float]:
    tracker = SimulatedTracker(config=config, seed=seed)
    hyp_frames = tracker.process(scenario["frames"])
    try:
        result = compute_mot_metrics(scenario["gt_mot"], hyp_frames)
    except Exception:
        result = {"MOTA": float("nan"), "MOTP_IoU": float("nan"), "IDF1": float("nan"),
                  "IDSW": 0, "FP": 0, "FN": 0}
    return result


class AblationRunner:
    """
    8 TrackerConfig (vlm_on=False) veya 16 TrackerConfig (--enable-vlm)
    × tüm senaryolar × n_seeds → pandas DataFrame.

    Sütunlar: config, scenario, seed, MOTA, MOTP_IoU, IDF1, IDSW, FP, FN, vlm_on
    """

    def __init__(
        self,
        configs: list[TrackerConfig] | None = None,
        n_seeds: int = 5,
        base_seed: int = 0,
        verbose: bool = True,
        enable_vlm: bool = False,
    ):
        if configs is not None:
            self.configs = configs
        elif enable_vlm:
            self.configs = VLM_ABLATION_CONFIGS   # 16 config
        else:
            self.configs = ALL_CONFIGS             # 8 config
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.verbose = verbose

    def run(self) -> pd.DataFrame:
        records = []
        total = len(self.configs) * self.n_seeds * 4  # 4 senaryo
        done = 0
        for seed_offset in range(self.n_seeds):
            seed = self.base_seed + seed_offset
            scenarios = build_all(seed=seed)
            for sc in scenarios:
                for cfg in self.configs:
                    metrics = _run_one(cfg, sc, seed)
                    records.append({
                        "config": cfg.label(),
                        "fusion_on": cfg.fusion_on,
                        "kalman_on": cfg.kalman_on,
                        "xai_adaptive": cfg.xai_adaptive,
                        "vlm_on": cfg.vlm_on,
                        "scenario": sc["name"],
                        "seed": seed,
                        **{k: v for k, v in metrics.items()
                           if k in ("MOTA", "MOTP_IoU", "IDF1", "IDSW", "FP", "FN")},
                    })
                    done += 1
                    if self.verbose:
                        print(f"  [{done}/{total}] {cfg.label()} | {sc['name']} | seed={seed} "
                              f"-> MOTA={metrics.get('MOTA', float('nan')):.3f}")
        return pd.DataFrame(records)

    @staticmethod
    def aggregate(df: pd.DataFrame) -> pd.DataFrame:
        """
        seed boyutunu ortalayıp std hesaplar (-inf / NaN değerleri hariç tutar).
        Döndürür: config × scenario → mean ± std
        """
        import numpy as np
        metrics = ["MOTA", "MOTP_IoU", "IDF1", "IDSW", "FP", "FN"]

        def safe_mean(s):
            finite = s.replace(-np.inf, np.nan).replace(np.inf, np.nan)
            return finite.mean()

        def safe_std(s):
            finite = s.replace(-np.inf, np.nan).replace(np.inf, np.nan)
            return finite.std()

        agg = (
            df.groupby(["config", "fusion_on", "kalman_on", "xai_adaptive", "scenario"])[metrics]
            .agg([safe_mean, safe_std])
            .reset_index()
        )
        agg.columns = [
            "_".join(c).strip("_").replace("safe_mean", "mean").replace("safe_std", "std")
            if c[1] else c[0]
            for c in agg.columns
        ]
        return agg

    @staticmethod
    def pivot_mota(agg: pd.DataFrame) -> pd.DataFrame:
        """config × scenario MOTA pivot tablosu."""
        return agg.pivot_table(
            index="config",
            columns="scenario",
            values="MOTA_mean",
            aggfunc="first",
        ).round(3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablation koşusu")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/ablation_raw.csv"))
    parser.add_argument("--agg-out", type=Path, default=Path("results/ablation_agg.csv"))
    parser.add_argument("--enable-vlm", action="store_true",
                        help="VLM eksenini ekle: 8 → 16 config (vlm_on ∈ {0,1})")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    n_configs = 16 if args.enable_vlm else 8
    print(f"Ablation koşusu: {n_configs} config × {args.seeds} seed × 4 senaryo "
          f"= {n_configs * args.seeds * 4} değerlendirme"
          + (" [VLM ekseni aktif]" if args.enable_vlm else ""))
    runner = AblationRunner(
        n_seeds=args.seeds, base_seed=args.base_seed,
        enable_vlm=args.enable_vlm,
    )
    df = runner.run()
    df.to_csv(args.out, index=False)
    print(f"\n[OK] Ham sonuçlar → {args.out}")

    agg = AblationRunner.aggregate(df)
    agg.to_csv(args.agg_out, index=False)
    print(f"[OK] Özet tablo → {args.agg_out}")

    pivot = AblationRunner.pivot_mota(agg)
    print("\nMOTA Özet (config × senaryo):")
    print(pivot.to_string())

    # VLM etki özeti
    if args.enable_vlm and "vlm_on" in df.columns:
        _print_vlm_delta(df)

    return 0


def _print_vlm_delta(df: pd.DataFrame) -> None:
    """VLM açık/kapalı arasındaki ortalama ΔMOTA farkını yazdır."""
    import numpy as np
    grp = df.groupby(["fusion_on", "kalman_on", "xai_adaptive", "vlm_on"])["MOTA"]
    pairs = []
    for (fu, ka, xa, vlm), grp_data in grp:
        mean = grp_data.replace(-np.inf, np.nan).mean()
        pairs.append({"fu": fu, "ka": ka, "xa": xa, "vlm": vlm, "mota": mean})
    import pandas as _pd
    p = _pd.DataFrame(pairs)
    off = p[p.vlm == False].set_index(["fu", "ka", "xa"])["mota"]
    on  = p[p.vlm == True ].set_index(["fu", "ka", "xa"])["mota"]
    delta = (on - off).dropna()
    print(f"\nVLM ΔMOTA (ortalama): {delta.mean():+.4f} "
          f"[min {delta.min():+.4f}, max {delta.max():+.4f}]")


if __name__ == "__main__":
    sys.exit(main())
