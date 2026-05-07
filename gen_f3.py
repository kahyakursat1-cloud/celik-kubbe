"""
gen_f3.py — Ablation heatmap (Figure 3) yeniden üretme scripti.
Doğru CSV'den (ablation_agg.csv) okuyan, insan-okunabilir etiketlerle
paper-quality figür üretir.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from pathlib import Path

# ── Yollar ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
CSV_PATH = ROOT / "docs" / "figures" / "ablation_agg.csv"
OUT_DOCS  = ROOT / "docs"  / "figures" / "F3_ablation_heatmap.pdf"
OUT_PAPER = ROOT / "paper" / "figures" / "F3_ablation_heatmap.pdf"

# ── Config etiketleri (okunabilir) ────────────────────────────────────────────
CONFIG_LABELS = {
    "fu0_ka0_xa0": "Base\n(no modules)",
    "fu0_ka0_xa1": "XAI only",
    "fu0_ka1_xa0": "Kalman only",
    "fu0_ka1_xa1": "Kalman + XAI",
    "fu1_ka0_xa0": "Fusion only",
    "fu1_ka0_xa1": "Fusion + XAI",
    "fu1_ka1_xa0": "Fusion + Kalman",
    "fu1_ka1_xa1": r"ContextFusion" + "\n" + r"(Full, Ours $\checkmark$)",
}

# Mantıksal sıra: en basit → en gelişmiş
CONFIG_ORDER = [
    "fu0_ka0_xa0",
    "fu0_ka0_xa1",
    "fu1_ka0_xa0",
    "fu1_ka0_xa1",
    "fu0_ka1_xa0",
    "fu0_ka1_xa1",
    "fu1_ka1_xa0",
    "fu1_ka1_xa1",
]

# Senaryo sırası: kolaydan zora
SCENARIO_ORDER = ["single_threat", "sensor_dropout", "multi_threat", "low_snr"]
SCENARIO_LABELS = {
    "single_threat":  "Single\nThreat",
    "multi_threat":   "Multi\nThreat",
    "sensor_dropout": "Sensor\nDropout",
    "low_snr":        "Low\nSNR",
}

# ── Veri okuma ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# Pivot: config × scenario → MOTA_mean
pivot = df.pivot_table(index="config", columns="scenario", values="MOTA_mean", aggfunc="first")
pivot_std = df.pivot_table(index="config", columns="scenario", values="MOTA_std", aggfunc="first")

# Sıralama
pivot     = pivot.reindex(index=CONFIG_ORDER, columns=SCENARIO_ORDER)
pivot_std = pivot_std.reindex(index=CONFIG_ORDER, columns=SCENARIO_ORDER)

mota = pivot.values          # (8, 4)
std  = pivot_std.values      # (8, 4)

# ── Figür ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

im = ax.imshow(mota, cmap="RdYlGn", vmin=-0.3, vmax=1.0, aspect="auto")
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Mean MOTA", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Eksen etiketleri
ax.set_xticks(range(len(SCENARIO_ORDER)))
ax.set_xticklabels(
    [SCENARIO_LABELS[s] for s in SCENARIO_ORDER],
    fontsize=10, fontweight="bold"
)
ax.set_yticks(range(len(CONFIG_ORDER)))
ax.set_yticklabels(
    [CONFIG_LABELS[c] for c in CONFIG_ORDER],
    fontsize=9
)

# Hücre değerleri (mean ± std)
for ci in range(len(CONFIG_ORDER)):
    for si in range(len(SCENARIO_ORDER)):
        val = mota[ci, si]
        sd  = std[ci, si]
        if np.isfinite(val):
            bg = val  # MOTA değeri
            text_color = "white" if bg < 0.25 or bg > 0.80 else "black"
            ax.text(si, ci, f"{val:.2f}\n±{sd:.2f}",
                    ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="normal")

# Tam sistem satırını çerçevele
full_idx = CONFIG_ORDER.index("fu1_ka1_xa1")
for si in range(len(SCENARIO_ORDER)):
    ax.add_patch(plt.Rectangle(
        (si - 0.5, full_idx - 0.5), 1, 1,
        fill=False, edgecolor="#2C3E50", linewidth=2.5, zorder=5
    ))

# Yatay ayırıcı çizgi (Kalman grubunu ayır)
ax.axhline(1.5, color="white", linewidth=1.5, alpha=0.6)  # XAI only / Fusion only arası
ax.axhline(3.5, color="white", linewidth=1.5, alpha=0.6)  # Kalman grubu başlar

ax.set_title(
    "Ablation Study — Mean MOTA per Configuration and Scenario",
    fontsize=11, fontweight="bold", pad=10
)
ax.set_xlabel("Evaluation Scenario", fontsize=10, labelpad=8)
ax.set_ylabel("System Configuration", fontsize=10, labelpad=8)

plt.tight_layout()

# ── Kaydet ───────────────────────────────────────────────────────────────────
for out in [OUT_DOCS, OUT_PAPER]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Kaydedildi: {out}")

plt.close(fig)
print("F3 tamamlandi.")
