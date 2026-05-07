"""
gen_f1.py — ContextFusion system architecture diagram (Figure 1).
Publication-quality, IEEE-style pipeline layout.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

ROOT = Path(__file__).parent
OUT_DOCS  = ROOT / "docs"  / "figures" / "F1_architecture.pdf"
OUT_PAPER = ROOT / "paper" / "figures" / "F1_architecture.pdf"

# ── Colour palette (IBM Colorblind-safe) ─────────────────────────────────────
C = {
    "sense":    "#648FFF",   # blue  — sensors
    "detect":   "#FFB000",   # amber — detection
    "fuse":     "#FE6100",   # orange — fusion/tracking
    "reason":   "#DC267F",   # magenta — reasoning
    "output":   "#009E73",   # green — output
    "arrow":    "#404040",
    "divider":  "#CCCCCC",
    "bg":       "#FAFAFA",
    "text_dark":"#1A1A2E",
}

ALPHA_BOX = 0.18
ALPHA_LAYER = 0.07

fig, ax = plt.subplots(figsize=(14, 5.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5.5)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Layer bands (background) ──────────────────────────────────────────────────
LAYERS = [
    (0.10,  2.50, C["sense"],  "Sensing"),
    (2.65,  2.50, C["detect"], "Detection"),
    (5.30,  2.50, C["fuse"],   "Fusion & Tracking"),
    (7.95,  2.50, C["reason"], "Reasoning"),
    (10.60, 3.10, C["output"], "Output"),
]
LAYER_Y, LAYER_H = 0.45, 4.30

for x, w, col, lbl in LAYERS:
    bg = mpatches.FancyBboxPatch(
        (x, LAYER_Y), w, LAYER_H,
        boxstyle="round,pad=0.08",
        facecolor=col, alpha=ALPHA_LAYER,
        edgecolor=col, linewidth=1.2, zorder=0
    )
    ax.add_patch(bg)
    ax.text(x + w / 2, LAYER_Y + LAYER_H + 0.10, lbl,
            ha="center", va="bottom", fontsize=8.5,
            color=col, fontweight="bold", zorder=3)

# ── Component boxes ──────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, sublabel, color, zorder=2):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.10",
        facecolor=color, alpha=0.85,
        edgecolor=color, linewidth=1.6, zorder=zorder
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2 + (0.18 if sublabel else 0),
            label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=zorder + 1)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 - 0.22,
                sublabel, ha="center", va="center",
                fontsize=7.5, color="white", alpha=0.92, zorder=zorder + 1,
                style="italic")

# Sensing
box(ax, 0.20, 2.85, 2.28, 1.15, "Camera", "YOLOv11m · 25 Hz", C["sense"])
box(ax, 0.20, 0.75, 2.28, 1.75, "AERIS-10 Radar",
    "FMCW X-band\nfₓ=10 GHz · 10 Hz\nCA-CFAR detection", C["sense"])

# Detection
box(ax, 2.80, 2.85, 2.25, 1.15, "Object\nDetector",
    "BBox · conf · class", C["detect"])
box(ax, 2.80, 0.75, 2.25, 1.75, "Radar\nProcessor",
    "Range · bearing\nSNR · Doppler\nSwerling RCS", C["detect"])

# Fusion + Tracking
box(ax, 5.45, 2.85, 2.25, 1.15, "MAP Fusion",
    "IoU + conf + SNR", C["fuse"])
box(ax, 5.45, 0.75, 2.25, 1.75, "Kalman\nTracker",
    "CV model · Δt=0.5 s\n4-state [x,y,ẋ,ẏ]\ngating θ≤15°", C["fuse"])

# Reasoning
box(ax, 8.10, 2.85, 2.25, 1.15, "XAI Threat\nScoring",
    "r·v·c factors\nSHAP attribution", C["reason"])
box(ax, 8.10, 0.75, 2.25, 1.75, "WTA\nOptimizer",
    "Hungarian alg.\nτᵢ × Pₖⁱʲ cost\nbattery constraints", C["reason"])

# Output
box(ax, 10.75, 2.85, 2.70, 1.15, "Operator\nDisplay",
    "SHAP bars · track list\nthreat ranking", C["output"])
box(ax, 10.75, 0.75, 2.70, 1.75, "Gimbal\nControl",
    "MAVLink\nbearing → motor\n18 FPS end-to-end", C["output"])

# ── Arrows ───────────────────────────────────────────────────────────────────
arrowkw = dict(arrowstyle="-|>", color=C["arrow"],
               lw=1.6, mutation_scale=14)

def arr(ax, x1, y1, x2, y2, **kw):
    kw_full = {**arrowkw, **kw}
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(**kw_full))

# camera → detector
arr(ax, 2.48, 3.42,  2.80, 3.42)
# radar → radar processor
arr(ax, 2.48, 1.62,  2.80, 1.62)

# detector → fusion
arr(ax, 5.05, 3.42,  5.45, 3.42)
# radar proc → fusion
arr(ax, 5.05, 1.62,  5.45, 2.00)
# radar proc → kalman (direct radar measurement)
arr(ax, 5.05, 1.40,  5.45, 1.40, linestyle="dashed", alpha=0.5)

# fusion → XAI
arr(ax, 7.70, 3.42,  8.10, 3.42)
# kalman → XAI (track data)
arr(ax, 7.70, 1.62,  8.10, 3.10)
# kalman → WTA
arr(ax, 7.70, 1.40,  8.10, 1.40)

# XAI → operator display
arr(ax, 10.35, 3.42,  10.75, 3.42)
# WTA → gimbal
arr(ax, 10.35, 1.62,  10.75, 1.62)

# XAI → WTA (threat scores)
arr(ax, 9.22, 2.85,  9.22, 2.50, color=C["reason"])

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(7.0, 5.28,
        "ContextFusion — Multi-Sensor Counter-UAV System Architecture",
        ha="center", va="center", fontsize=11.5, fontweight="bold",
        color=C["text_dark"])

# ── Legend for dashed arrow ──────────────────────────────────────────────────
solid_line = mpatches.Patch(facecolor=C["arrow"], label="Primary data flow")
dash_line  = plt.Line2D([0], [0], linestyle="--", color=C["arrow"],
                        alpha=0.55, label="Auxiliary measurement path")
ax.legend(handles=[solid_line, dash_line],
          loc="lower left", bbox_to_anchor=(0.01, -0.02),
          fontsize=7.5, framealpha=0.7, ncol=2)

plt.tight_layout(pad=0.4)

for out in [OUT_DOCS, OUT_PAPER]:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Kaydedildi: {out}")

plt.close(fig)
print("F1 tamamlandi.")
