"""
eval/figures.py — Paper-ready figürler (300 DPI, PDF/PNG).

F1: Mimari blok diyagramı (Matplotlib patch)
F2: Tracker performans box-plot (MOTA per scenario)
F3: Ablation heatmap (fusion × kalman × scenario)
F4: ROC eğrileri (detection probability vs PFA)
F5: Örnek track görselleştirmesi (qualitative)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGURES_DIR = Path(__file__).parent.parent / "docs" / "figures"
DPI = 300
TRACKER_COLORS = {
    "CelikKubbe": "#1f77b4",
    "SORT":        "#ff7f0e",
    "DeepSORT":    "#2ca02c",
    "ByteTrack":   "#d62728",
    "IMM-Kalman":  "#9467bd",
}


def _save(fig: plt.Figure, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Kaydedildi: {path}")
    return path


# ─── F1: Mimari Diyagramı ─────────────────────────────────────────────────────

def figure_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, color="#AED6F1", fontsize=9):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor="#2C3E50", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))

    box(0.2, 3.8, 1.6, 0.9, "Camera\nYOLOv11m", "#AED6F1")
    box(0.2, 2.4, 1.6, 0.9, "AERIS-10\nRadar", "#A9DFBF")
    box(2.2, 3.1, 1.8, 1.0, "Detection\nPipeline", "#F9E79F")
    box(4.4, 3.1, 1.8, 1.0, "Sensor\nFusion", "#FAD7A0")
    box(6.6, 3.8, 1.8, 0.9, "Kalman\nTracker", "#D7BDE2")
    box(6.6, 2.4, 1.8, 0.9, "XAI Threat\nScoring", "#F1948A")
    box(8.8, 3.1, 1.8, 1.0, "WTA\nOptimizer", "#ABEBC6")
    box(10.8, 3.1, 1.0, 1.0, "Gimbal\nControl", "#85C1E9")

    arrow(1.8, 4.25, 2.2, 3.8)
    arrow(1.8, 2.85, 2.2, 3.4)
    arrow(4.0, 3.6, 4.4, 3.6)
    arrow(6.2, 3.6, 6.6, 4.25)
    arrow(6.2, 3.6, 6.6, 2.85)
    arrow(8.4, 4.25, 8.8, 3.8)
    arrow(8.4, 2.85, 8.8, 3.4)
    arrow(10.6, 3.6, 10.8, 3.6)

    ax.text(6.0, 5.6, "Çelik Kubbe — Multi-Sensor Counter-UAV System",
            ha="center", va="center", fontsize=13, fontweight="bold")

    return _save(fig, "F1_architecture.pdf")


# ─── F2: Tracker Box-Plot ─────────────────────────────────────────────────────

def figure_tracker_boxplot(mc_results) -> Path:
    from eval.monte_carlo import MCResult

    scenarios = ["single_threat", "multi_threat", "sensor_dropout", "low_snr"]
    trackers = list(TRACKER_COLORS.keys())

    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 5), sharey=True)
    fig.suptitle("MOTA Distribution — Monte Carlo (N=50)", fontsize=12, fontweight="bold")

    for ax, scenario in zip(axes, scenarios):
        data = []
        labels = []
        colors = []
        for tracker in trackers:
            vals = [r.mota for r in mc_results
                    if r.scenario == scenario and r.tracker == tracker
                    and np.isfinite(r.mota)]
            data.append(vals)
            labels.append(tracker.replace("CelikKubbe", "Ours"))
            colors.append(TRACKER_COLORS[tracker])

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(scenario.replace("_", "\n"), fontsize=9)
        ax.set_xticks(range(1, len(trackers) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_ylim(-1.0, 1.1)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("MOTA", fontsize=10)
    plt.tight_layout()
    return _save(fig, "F2_tracker_boxplot.pdf")


# ─── F3: Ablation Heatmap ─────────────────────────────────────────────────────

def figure_ablation_heatmap(ablation_csv_path: Optional[Path] = None) -> Path:
    import csv

    csv_path = ablation_csv_path or (
        FIGURES_DIR.parent / "figures" / "ablation_results.csv"
    )
    if not csv_path.exists():
        csv_path = FIGURES_DIR / "ablation_results.csv"

    rows = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "Ablation CSV bulunamadi\n(eval/ablation_runner.py ile uret)",
                ha="center", va="center", fontsize=12, color="red")
        return _save(fig, "F3_ablation_heatmap.pdf")

    scenarios = sorted(set(r["scenario"] for r in rows))
    configs = sorted(set(r["config"] for r in rows))

    mota_grid = np.full((len(configs), len(scenarios)), float("nan"))
    for r in rows:
        ci = configs.index(r["config"])
        si = scenarios.index(r["scenario"])
        try:
            mota_grid[ci, si] = float(r.get("mota_mean", r.get("MOTA_mean", "nan")))
        except (ValueError, KeyError):
            pass

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(mota_grid, cmap="RdYlGn", vmin=-0.5, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="MOTA (mean)")

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=9)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=8)
    ax.set_title("Ablation Study — MOTA per Config × Scenario", fontsize=11, fontweight="bold")

    for ci in range(len(configs)):
        for si in range(len(scenarios)):
            val = mota_grid[ci, si]
            if np.isfinite(val):
                ax.text(si, ci, f"{val:.2f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    return _save(fig, "F3_ablation_heatmap.pdf")


# ─── F4: ROC Eğrileri ────────────────────────────────────────────────────────

def figure_roc_curves(mc_results=None) -> Path:
    """
    Pseudo-ROC: CFAR PFA sweep → detection probability.
    PhysicsRadarSimulator ile üretilir.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from radar_simulator import PhysicsRadarSimulator, RadarParams, TargetProfile

    pfa_values = np.logspace(-6, -1, 40)
    scenarios = {
        "Drone@3km":   ("Drone", 3000.0, 5.0),
        "Helikopter@5km": ("Helikopter", 5000.0, 10.0),
        "Jet@8km":     ("Jet", 8000.0, 15.0),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    colors_roc = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for (label, (sinif, range_m, el)), color in zip(scenarios.items(), colors_roc):
        pd_vals = []
        for pfa in pfa_values:
            params = RadarParams(pfa=float(pfa))
            sim = PhysicsRadarSimulator(params, seed=0)
            n_trials = 200
            detected = 0
            for trial in range(n_trials):
                sim._rng.seed(trial)
                sim._np_rng = np.random.default_rng(trial)
                dets = sim.detect_targets([
                    TargetProfile(range_m=range_m, bearing_deg=0,
                                  elevation_deg=el, velocity_ms=-50, sinif=sinif)
                ])
                if dets and dets[0].detected:
                    detected += 1
            pd_vals.append(detected / n_trials)

        ax.semilogx(pfa_values, pd_vals, color=color, label=label, linewidth=2)

    ax.set_xlabel("Probability of False Alarm (PFA)", fontsize=10)
    ax.set_ylabel("Probability of Detection ($P_d$)", fontsize=10)
    ax.set_title("ROC Curves — Physics-Based Radar Simulator", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return _save(fig, "F4_roc_curves.pdf")


# ─── F5: Örnek Track Görselleştirmesi ────────────────────────────────────────

def figure_track_example() -> Path:
    """
    Qualitative: 3 hedef, 100 frame, CelikKubbe vs ByteTrack track yolları.
    """
    from eval.scenarios import multi_threat
    from eval.simulated_tracker import SimulatedTracker, TrackerConfig
    from eval.baselines.bytetrack_wrapper import ByteTrackWrapper

    scenario = multi_threat(seed=42)
    gt_frames = scenario["frames"]

    cfg = TrackerConfig(fusion_on=True, kalman_on=True)
    ours = SimulatedTracker(cfg, seed=42)
    hyp_ours = ours.process(gt_frames)

    cfg_raw = TrackerConfig(fusion_on=False, kalman_on=False)
    raw = SimulatedTracker(cfg_raw, seed=42)
    det_frames = raw.process(gt_frames)
    bt = ByteTrackWrapper()
    hyp_bt = bt.process(det_frames)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Qualitative Track Examples — Multi-Threat Scenario",
                 fontsize=11, fontweight="bold")

    def _plot_tracks(ax, hyp_frames, title, gt_f):
        track_paths: dict[int, list[tuple[float, float]]] = {}
        for f in hyp_frames:
            for tid, box in zip(f["ids"], f["boxes"]):
                cx = box[0] + box[2] / 2
                cy = box[1] + box[3] / 2
                track_paths.setdefault(tid, []).append((cx, cy))

        gt_paths: dict[int, list[tuple[float, float]]] = {}
        for f in gt_f:
            for tid, box in zip(f.track_ids, f.bboxes):
                cx = box[0] + box[2] / 2
                cy = box[1] + box[3] / 2
                gt_paths.setdefault(tid, []).append((cx, cy))

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(gt_paths), 1)))
        for (tid, pts), col in zip(gt_paths.items(), colors):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "--", color=col, alpha=0.5, linewidth=1, label=f"GT {tid}")

        for (tid, pts), col in zip(track_paths.items(), colors[:len(track_paths)]):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-", color=col, linewidth=1.5)
            ax.plot(xs[0], ys[0], "o", color=col, markersize=5)

        ax.set_xlim(0, 1280)
        ax.set_ylim(720, 0)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.grid(alpha=0.2)

    _plot_tracks(ax1, hyp_ours, "Çelik Kubbe (Fusion+Kalman)", gt_frames)
    _plot_tracks(ax2, hyp_bt, "ByteTrack (Baseline)", gt_frames)

    gt_line = plt.Line2D([0], [0], linestyle="--", color="gray", label="Ground Truth")
    hyp_line = plt.Line2D([0], [0], linestyle="-", color="gray", label="Hypothesis")
    fig.legend(handles=[gt_line, hyp_line], loc="lower center", ncol=2, fontsize=9)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    return _save(fig, "F5_track_example.pdf")


# ─── Ana çağrı ───────────────────────────────────────────────────────────────

def generate_all(mc_results=None) -> None:
    print("Figurler uretiliyor...")
    figure_architecture()
    if mc_results:
        figure_tracker_boxplot(mc_results)
    figure_ablation_heatmap()
    print("  F4 ROC hesaplaniyor (200 trial × 40 PFA)...")
    figure_roc_curves()
    figure_track_example()
    print("Tum figurler tamamlandi.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc-pkl", type=Path, default=None,
                        help="Monte Carlo pickle dosyasi (F2 icin)")
    args = parser.parse_args()

    mc_results = None
    if args.mc_pkl and args.mc_pkl.exists():
        from eval.monte_carlo import load_results
        mc_results = load_results(args.mc_pkl)

    generate_all(mc_results)
