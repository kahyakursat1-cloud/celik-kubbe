"""
eval/results_aggregator.py — Ablasyon sonuçlarını tablo + figür olarak çıkarır.

Kullanım:
    from eval.results_aggregator import ResultsAggregator
    agg = ResultsAggregator(raw_df)
    agg.save_figures("docs/figures/")
    agg.save_latex("docs/figures/ablation_table.tex")
    agg.print_summary()
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_LABELS = {
    "MOTA_mean": "MOTA",
    "MOTP_IoU_mean": "MOTP (IoU)",
    "IDF1_mean": "IDF1",
    "IDSW_mean": "ID-SW",
}

SCENARIO_ORDER = ["single_threat", "multi_threat", "sensor_dropout", "low_snr"]
SCENARIO_LABELS = {
    "single_threat": "Single Threat",
    "multi_threat": "Multi Threat",
    "sensor_dropout": "Sensor Dropout",
    "low_snr": "Low SNR",
}

CONFIG_LABELS = {
    "fu0_ka0_xa0": "No Fusion, No Kalman, Fixed XAI",
    "fu0_ka0_xa1": "No Fusion, No Kalman, Adaptive XAI",
    "fu0_ka1_xa0": "No Fusion, Kalman, Fixed XAI",
    "fu0_ka1_xa1": "No Fusion, Kalman, Adaptive XAI",
    "fu1_ka0_xa0": "Fusion, No Kalman, Fixed XAI",
    "fu1_ka0_xa1": "Fusion, No Kalman, Adaptive XAI",
    "fu1_ka1_xa0": "Fusion, Kalman, Fixed XAI (Ours)",
    "fu1_ka1_xa1": "Fusion, Kalman, Adaptive XAI (Ours+)",
}


class ResultsAggregator:

    def __init__(self, raw_df: pd.DataFrame):
        self.raw = raw_df
        self.agg = self._aggregate()

    def _aggregate(self) -> pd.DataFrame:
        from eval.ablation_runner import AblationRunner
        return AblationRunner.aggregate(self.raw)

    def save_figures(self, out_dir: str | Path, dpi: int = 300) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._fig_heatmap(out, dpi)
        self._fig_boxplot(out, dpi)

    def _fig_heatmap(self, out: Path, dpi: int) -> None:
        pivot = self.agg.pivot_table(
            index="config", columns="scenario",
            values="MOTA_mean", aggfunc="first",
        )
        configs_ordered = [c for c in [
            "fu0_ka0_xa0", "fu0_ka0_xa1", "fu0_ka1_xa0", "fu0_ka1_xa1",
            "fu1_ka0_xa0", "fu1_ka0_xa1", "fu1_ka1_xa0", "fu1_ka1_xa1",
        ] if c in pivot.index]
        scenarios_ordered = [s for s in SCENARIO_ORDER if s in pivot.columns]
        pivot = pivot.reindex(index=configs_ordered, columns=scenarios_ordered)

        short_labels = [c.replace("fu", "F").replace("ka", "K").replace("xa", "X")
                        for c in configs_ordered]
        sc_labels = [SCENARIO_LABELS.get(s, s) for s in scenarios_ordered]

        fig, ax = plt.subplots(figsize=(8, 5))
        data = pivot.values.astype(float)
        im = ax.imshow(data, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        plt.colorbar(im, ax=ax, label="MOTA")

        ax.set_xticks(range(len(sc_labels)))
        ax.set_xticklabels(sc_labels, rotation=20, ha="right", fontsize=9)
        ax.set_yticks(range(len(short_labels)))
        ax.set_yticklabels(short_labels, fontsize=8)
        ax.set_title("Ablation Study — MOTA Heatmap\n"
                     "(F=Fusion, K=Kalman, X=Adaptive XAI; 1=On, 0=Off)", fontsize=10)

        for i in range(len(configs_ordered)):
            for j in range(len(scenarios_ordered)):
                val = data[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color="black" if 0.3 < val < 0.8 else "white")

        fig.tight_layout()
        path = out / "ablation_heatmap.pdf"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Heatmap → {path}")

    def _fig_boxplot(self, out: Path, dpi: int) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        metrics = ["MOTA", "MOTP_IoU", "IDF1"]
        titles = ["MOTA", "MOTP (IoU)", "IDF1"]

        for ax, metric, title in zip(axes, metrics, titles):
            data_by_config = []
            labels = []
            configs_ordered = [
                "fu0_ka0_xa0", "fu0_ka0_xa1", "fu0_ka1_xa0", "fu0_ka1_xa1",
                "fu1_ka0_xa0", "fu1_ka0_xa1", "fu1_ka1_xa0", "fu1_ka1_xa1",
            ]
            for cfg_label in configs_ordered:
                subset = self.raw[self.raw["config"] == cfg_label][metric].dropna()
                if len(subset) > 0:
                    data_by_config.append(subset.values)
                    labels.append(cfg_label.replace("fu", "F").replace("ka", "K")
                                  .replace("xa", "X"))

            if data_by_config:
                bp = ax.boxplot(data_by_config, patch_artist=True, notch=False)
                colors = ["#d32f2f" if "fu0" in labels[i] else "#1976d2"
                          for i in range(len(labels))]
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.set_title(title, fontsize=10)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3, axis="y")
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

        axes[0].set_ylabel("Metrik değeri")
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#d32f2f", alpha=0.7, label="No Fusion"),
            Patch(facecolor="#1976d2", alpha=0.7, label="With Fusion"),
        ]
        fig.legend(handles=legend_elements, loc="upper center",
                   ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=9)
        fig.suptitle("Tracker Ablation Study — MOT Metrics Distribution", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        path = out / "ablation_boxplot.pdf"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Boxplot → {path}")

    def save_latex(self, out_path: str | Path) -> None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        pivot_mota = self.agg.pivot_table(
            index="config", columns="scenario",
            values="MOTA_mean", aggfunc="first",
        )
        pivot_std = self.agg.pivot_table(
            index="config", columns="scenario",
            values="MOTA_std", aggfunc="first",
        )

        configs_ordered = [
            "fu0_ka0_xa0", "fu0_ka0_xa1", "fu0_ka1_xa0", "fu0_ka1_xa1",
            "fu1_ka0_xa0", "fu1_ka0_xa1", "fu1_ka1_xa0", "fu1_ka1_xa1",
        ]
        scenarios_ordered = [s for s in SCENARIO_ORDER if s in pivot_mota.columns]
        pivot_mota = pivot_mota.reindex(index=configs_ordered, columns=scenarios_ordered)
        pivot_std = pivot_std.reindex(index=configs_ordered, columns=scenarios_ordered)

        sc_headers = " & ".join(
            SCENARIO_LABELS.get(s, s).replace("_", r"\_") for s in scenarios_ordered
        )

        rows = []
        for cfg in configs_ordered:
            if cfg not in pivot_mota.index:
                continue
            row_label = CONFIG_LABELS.get(cfg, cfg).replace("_", r"\_")
            is_ours = "Ours" in CONFIG_LABELS.get(cfg, "")
            cells = []
            for sc in scenarios_ordered:
                mean_v = pivot_mota.loc[cfg, sc] if sc in pivot_mota.columns else float("nan")
                std_v = pivot_std.loc[cfg, sc] if sc in pivot_std.columns else float("nan")
                if np.isnan(mean_v):
                    cells.append("—")
                else:
                    std_str = f"{{\\tiny$\\pm${std_v:.2f}}}" if not np.isnan(std_v) else ""
                    cell = f"{mean_v:.3f}{std_str}"
                    if is_ours:
                        cell = r"\textbf{" + cell + "}"
                    cells.append(cell)
            row_str = f"  {row_label} & " + " & ".join(cells) + r" \\"
            if is_ours:
                rows.append(r"  \midrule")
            rows.append(row_str)

        n_cols = 1 + len(scenarios_ordered)
        col_spec = "l" + "c" * len(scenarios_ordered)

        latex = rf"""% ablation_table.tex — auto-generated by eval/results_aggregator.py
% Include in paper: \input{{figures/ablation_table.tex}}
\begin{{table}}[htbp]
\centering
\caption{{Ablation study: MOTA (mean$\pm$std over 5 seeds, 4 scenarios).
  F=Fusion, K=Kalman filter, X=Adaptive XAI.
  Bold rows are our proposed configurations.}}
\label{{tab:ablation}}
\setlength{{\tabcolsep}}{{6pt}}
\begin{{tabular}}{{{col_spec}}}
\toprule
\textbf{{Configuration}} & {sc_headers} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
        out.write_text(latex, encoding="utf-8")
        print(f"[OK] LaTeX tablo → {out}")

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print(" Ablation Özeti — MOTA (tüm senaryolar ortalama)")
        print("=" * 60)
        mean_per_config = (
            self.raw.groupby("config")["MOTA"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
        )
        for cfg, row in mean_per_config.iterrows():
            bar_len = int(row["mean"] * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            label = CONFIG_LABELS.get(str(cfg), str(cfg))
            print(f"  {str(cfg):<18} {bar} {row['mean']:.3f} ± {row['std']:.3f}  {label}")
        print()
        best = mean_per_config["mean"].idxmax()
        print(f"  → En iyi config: {best} ({CONFIG_LABELS.get(str(best), '')})")
