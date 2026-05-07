"""
gen_f2.py — Tracker MOTA comparison (Figure 2).
4-panel grouped bar + scatter, publication quality.
Reads comparison_raw.csv (N seeds per tracker × scenario).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats

ROOT = Path(__file__).parent
CSV_PATH  = ROOT / "docs" / "figures" / "comparison_raw.csv"
OUT_DOCS  = ROOT / "docs"  / "figures" / "F2_tracker_boxplot.pdf"
OUT_PAPER = ROOT / "paper" / "figures" / "F2_tracker_boxplot.pdf"

# ── Config ───────────────────────────────────────────────────────────────────
TRACKER_MAP = {
    "Celik-Kubbe": "ContextFusion\n(Ours)",
    "DeepSORT":    "DeepSORT\n(≈SORT)",
    "ByteTrack":   "ByteTrack",
    "IMM-Kalman":  "IMM-Kalman",
}
TRACKER_ORDER = ["Celik-Kubbe", "DeepSORT", "ByteTrack", "IMM-Kalman"]
SCENARIO_ORDER = ["single_threat", "multi_threat", "sensor_dropout", "low_snr"]
SCENARIO_LABELS = {
    "single_threat":  "Single Threat",
    "multi_threat":   "Multi Threat",
    "sensor_dropout": "Sensor Dropout",
    "low_snr":        "Low SNR",
}

# Colorblind-safe (IBM palette)
COLORS = {
    "Celik-Kubbe": "#648FFF",
    "DeepSORT":    "#009E73",
    "ByteTrack":   "#FE6100",
    "IMM-Kalman":  "#9467BD",
}

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

fig, axes = plt.subplots(1, 4, figsize=(14, 4.8), sharey=False)
fig.subplots_adjust(wspace=0.32, left=0.06, right=0.97, top=0.88, bottom=0.22)

n_trackers = len(TRACKER_ORDER)
bar_w = 0.18
group_span = bar_w * n_trackers
x_center = 0.0   # scalar — single group per panel
offsets = np.linspace(-(group_span / 2 - bar_w / 2),
                       (group_span / 2 - bar_w / 2), n_trackers)

for ax, scenario in zip(axes, SCENARIO_ORDER):
    sub = df[df["scenario"] == scenario]

    all_means = []
    for tk, offset in zip(TRACKER_ORDER, offsets):
        vals = sub[sub["tracker"] == tk]["MOTA"].values
        if len(vals) == 0:
            all_means.append(np.nan)
            continue
        mean_v = vals.mean()
        std_v  = vals.std(ddof=1) if len(vals) > 1 else 0.0
        all_means.append(mean_v)
        col = COLORS[tk]

        # Bar (mean)
        bar = ax.bar(x_center + offset, mean_v, bar_w * 0.88,
                     color=col, alpha=0.80, zorder=3,
                     bottom=min(mean_v, 0) if mean_v < 0 else 0)

        # Error bar (±1 std)
        ax.errorbar(x_center + offset, mean_v, yerr=std_v,
                    fmt="none", color="black", capsize=3.5,
                    linewidth=1.0, zorder=4)

        # Scatter individual points
        jitter = np.random.default_rng(42).uniform(-bar_w * 0.25, bar_w * 0.25, len(vals))
        ax.scatter(x_center + offset + jitter, vals,
                   color=col, s=18, zorder=5, edgecolors="white",
                   linewidths=0.5, alpha=0.9)

        # Significance stars vs ContextFusion
        if tk != "Celik-Kubbe":
            ck_vals = sub[sub["tracker"] == "Celik-Kubbe"]["MOTA"].values
            if len(vals) >= 3 and len(ck_vals) >= 3:
                _, pval = scipy_stats.wilcoxon(ck_vals[:len(vals)], vals,
                                               alternative="greater",
                                               zero_method="zsplit")
                stars = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                if stars:
                    ck_mean = ck_vals.mean()
                    y_star = max(mean_v, ck_mean) + std_v + 0.04
                    ax.text(x_center + offset, y_star, stars,
                            ha="center", va="bottom", fontsize=7,
                            color="#333333", zorder=6)

    # Zero line
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=2)

    # Axes formatting
    ax.set_xticks([])
    ax.set_xlim(-0.5, 0.5)

    # Y limits: headroom above and below
    finite_means = [m for m in all_means if np.isfinite(m)]
    if finite_means:
        ymin = min(finite_means) - 0.25
        ymax = max(finite_means) + 0.30
        ax.set_ylim(ymin, ymax)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", alpha=0.25, zorder=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(SCENARIO_LABELS[scenario], fontsize=9.5, fontweight="bold", pad=6)

axes[0].set_ylabel("MOTA", fontsize=10, labelpad=6)

# ── Shared legend ─────────────────────────────────────────────────────────────
import matplotlib.patches as mpatches
handles = [
    mpatches.Patch(facecolor=COLORS[tk], alpha=0.80,
                   label=TRACKER_MAP[tk].replace("\n", " "))
    for tk in TRACKER_ORDER
]
fig.legend(handles=handles,
           loc="lower center", ncol=4,
           fontsize=8.5, framealpha=0.85,
           bbox_to_anchor=(0.52, -0.01))

# ── p-value note ─────────────────────────────────────────────────────────────
fig.text(0.97, 0.01,
         "* p<0.05   ** p<0.01   *** p<0.001  (Wilcoxon, ContextFusion vs baseline)",
         ha="right", va="bottom", fontsize=7, color="#555555", style="italic")

fig.suptitle(
    "MOTA Comparison: ContextFusion vs Baselines (Monte Carlo, N=50)",
    fontsize=11, fontweight="bold", y=0.97
)

for out in [OUT_DOCS, OUT_PAPER]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Kaydedildi: {out}")

plt.close(fig)
print("F2 tamamlandi.")
