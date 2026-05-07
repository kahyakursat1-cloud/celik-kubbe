"""
analyze_mission.py — Görev sonrası kara kutu analiz scripti.

logs/blackbox/tracks_*.csv ve events_*.csv dosyalarını yükleyip:
  • Polar plot: tehdit izleri (range vs bearing, zaman renkli)
  • Tehdit seviyesi dağılımı, sınıf pie chart, kaynak dağılımı
  • Range zaman serisi (per-track)
  • Olay zaman çizgisi
  • summary.txt: metrik özeti

Kullanım:
    python deployment/analyze_mission.py logs/blackbox/tracks_20260505_110608.csv
    python deployment/analyze_mission.py --latest
    python deployment/analyze_mission.py logs/blackbox/tracks_*.csv  # birden fazla

Çıktı: logs/analysis/<timestamp>/dashboard.png + summary.txt
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LEVEL_ORDER = ["DÜŞÜK", "ORTA", "YÜKSEK", "KRİTİK"]
LEVEL_COLORS = {"DÜŞÜK": "#4caf50", "ORTA": "#ffc107",
                "YÜKSEK": "#ff9800", "KRİTİK": "#f44336"}


def load_tracks(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    r["Range_km"] = float(r["Range_km"])
                    r["Bearing_deg"] = float(r["Bearing_deg"])
                    r["Velocity_ms"] = float(r["Velocity_ms"])
                    r["Altitude_m"] = float(r["Altitude_m"])
                    r["_ts"] = datetime.fromisoformat(r["Timestamp"])
                    r["_source_file"] = p.name
                except (ValueError, KeyError):
                    continue
                rows.append(r)
    return rows


def load_events(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    r["_ts"] = datetime.fromisoformat(r["Timestamp"])
                except (ValueError, KeyError):
                    continue
                rows.append(r)
    return rows


def panel_polar_tracks(ax, rows):
    if not rows:
        ax.text(0.5, 0.5, "Tehdit izi yok", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Polar İz Haritası")
        return

    by_id: dict[str, list] = defaultdict(list)
    for r in rows:
        by_id[r["Threat_ID"]].append(r)

    # En aktif 30 tehdidi göster (kalabalık olmasın)
    top = sorted(by_id.items(), key=lambda kv: -len(kv[1]))[:30]

    for tid, samples in top:
        thetas = np.radians([s["Bearing_deg"] for s in samples])
        rs = [s["Range_km"] for s in samples]
        ax.plot(thetas, rs, "-", linewidth=0.8, alpha=0.6)
        ax.plot(thetas[-1:], rs[-1:], "o", markersize=3)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Polar İz Haritası — top {len(top)}/{len(by_id)} hedef")


def panel_threat_level(ax, rows):
    if not rows:
        ax.set_title("Tehdit Seviyesi (veri yok)")
        return
    by_id: dict[str, str] = {}
    for r in rows:
        by_id[r["Threat_ID"]] = r["Threat_Level"]  # son seviye
    counts = Counter(by_id.values())
    levels = [l for l in LEVEL_ORDER if l in counts]
    values = [counts[l] for l in levels]
    colors = [LEVEL_COLORS.get(l, "#888") for l in levels]
    ax.bar(levels, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Tehdit Seviyesi — toplam {sum(values)} hedef")
    ax.set_ylabel("Hedef sayısı")


def panel_class_pie(ax, rows):
    if not rows:
        ax.set_title("Sınıf Dağılımı (veri yok)")
        return
    by_id: dict[str, str] = {}
    for r in rows:
        by_id[r["Threat_ID"]] = r["Class"]
    counts = Counter(by_id.values()).most_common(8)
    labels = [c for c, _ in counts]
    values = [v for _, v in counts]
    ax.pie(values, labels=labels, autopct="%1.0f%%", startangle=90,
           textprops={"fontsize": 8})
    ax.set_title("Sınıf Dağılımı")


def panel_source(ax, rows):
    if not rows:
        ax.set_title("Kaynak Dağılımı (veri yok)")
        return
    counts = Counter(r["Source"] for r in rows)
    sources = list(counts.keys())
    values = [counts[s] for s in sources]
    ax.barh(sources, values, color="#1976d2")
    for i, v in enumerate(values):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    ax.set_title("Kaynak Dağılımı (örnek bazlı)")
    ax.set_xlabel("Örnek sayısı")


def panel_range_time(ax, rows):
    if not rows:
        ax.set_title("Range Zaman Serisi (veri yok)")
        return
    t0 = min(r["_ts"] for r in rows)
    by_id: dict[str, list] = defaultdict(list)
    for r in rows:
        elapsed = (r["_ts"] - t0).total_seconds()
        by_id[r["Threat_ID"]].append((elapsed, r["Range_km"]))

    top = sorted(by_id.items(), key=lambda kv: -len(kv[1]))[:15]
    for tid, samples in top:
        ts = [s[0] for s in samples]
        rs = [s[1] for s in samples]
        ax.plot(ts, rs, "-", linewidth=0.8, alpha=0.7, label=tid[:10])

    ax.set_xlabel("Görev süresi (s)")
    ax.set_ylabel("Mesafe (km)")
    ax.set_title(f"Range vs Zaman — top {len(top)} hedef")
    ax.grid(True, alpha=0.3)
    if len(top) <= 8:
        ax.legend(fontsize=7, loc="upper right")


def panel_events(ax, events):
    if not events:
        ax.text(0.5, 0.5, "Olay yok", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Olay Zaman Çizgisi")
        return
    t0 = min(e["_ts"] for e in events)
    types = sorted(set(e["Event_Type"] for e in events))
    color_map = {t: plt.cm.tab10(i % 10) for i, t in enumerate(types)}
    for e in events:
        elapsed = (e["_ts"] - t0).total_seconds()
        y = types.index(e["Event_Type"])
        ax.scatter(elapsed, y, c=[color_map[e["Event_Type"]]], s=30, alpha=0.7)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=8)
    ax.set_xlabel("Görev süresi (s)")
    ax.set_title(f"Olay Zaman Çizgisi — {len(events)} olay")
    ax.grid(True, alpha=0.3, axis="x")


def write_summary(out_path: Path, tracks: list[dict], events: list[dict],
                  track_files: list[Path], event_files: list[Path]) -> None:
    lines = ["Çelik Kubbe — Görev Analiz Özeti",
             "=" * 50, ""]

    if tracks:
        t_start = min(r["_ts"] for r in tracks)
        t_end = max(r["_ts"] for r in tracks)
        duration = (t_end - t_start).total_seconds()
        unique_ids = {r["Threat_ID"] for r in tracks}

        lines.append(f"Görev başlangıcı : {t_start.isoformat(timespec='seconds')}")
        lines.append(f"Görev bitişi     : {t_end.isoformat(timespec='seconds')}")
        lines.append(f"Süre             : {duration:.1f} s ({duration/60:.1f} dk)")
        lines.append(f"Tespit kaydı     : {len(tracks)}")
        lines.append(f"Benzersiz hedef  : {len(unique_ids)}")
        lines.append("")

        # Sınıf dağılımı (son sınıflandırma)
        last_class = {}
        for r in tracks:
            last_class[r["Threat_ID"]] = r["Class"]
        lines.append("Sınıf dağılımı (benzersiz):")
        for cls, n in Counter(last_class.values()).most_common():
            lines.append(f"  {cls:25s} {n:5d}")
        lines.append("")

        # Tehdit seviyesi dağılımı (son seviye)
        last_level = {}
        for r in tracks:
            last_level[r["Threat_ID"]] = r["Threat_Level"]
        lines.append("Tehdit seviyesi (benzersiz):")
        for lvl in LEVEL_ORDER:
            n = sum(1 for v in last_level.values() if v == lvl)
            if n:
                lines.append(f"  {lvl:10s} {n:5d}")
        lines.append("")

        # Kaynak dağılımı (örnek bazlı)
        lines.append("Kaynak dağılımı (örnek):")
        for src, n in Counter(r["Source"] for r in tracks).most_common():
            lines.append(f"  {src:20s} {n:7d}")
        lines.append("")

        # Mesafe istatistikleri
        ranges = [r["Range_km"] for r in tracks]
        lines.append(f"Mesafe (km):     min={min(ranges):.2f}  "
                     f"medyan={np.median(ranges):.2f}  max={max(ranges):.2f}")
        velocities = [abs(r["Velocity_ms"]) for r in tracks if r["Velocity_ms"] != 0]
        if velocities:
            lines.append(f"Hız |v| (m/s):   min={min(velocities):.1f}  "
                         f"medyan={np.median(velocities):.1f}  max={max(velocities):.1f}")
        lines.append("")
    else:
        lines.append("(Tehdit izi kaydı yok)")
        lines.append("")

    if events:
        lines.append(f"Olay sayısı      : {len(events)}")
        for et, n in Counter(e["Event_Type"] for e in events).most_common():
            lines.append(f"  {et:20s} {n:5d}")
        lines.append("")

    lines.append("Kaynak dosyalar:")
    for p in track_files:
        lines.append(f"  tracks: {p}")
    for p in event_files:
        lines.append(f"  events: {p}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_latest_tracks() -> list[Path]:
    bb = PROJECT_ROOT / "logs" / "blackbox"
    files = sorted(bb.glob("tracks_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        return []
    return [files[-1]]


def matched_event_files(track_files: list[Path]) -> list[Path]:
    """tracks_YYYYMMDD_HHMMSS.csv → events_YYYYMMDD_HHMMSS.csv eşleştir."""
    out = []
    for tf in track_files:
        ef = tf.with_name(tf.name.replace("tracks_", "events_", 1))
        if ef.is_file():
            out.append(ef)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Çelik Kubbe görev analiz")
    parser.add_argument("inputs", nargs="*", type=Path,
                        help="tracks_*.csv dosyaları (yoksa --latest gerekli)")
    parser.add_argument("--latest", action="store_true",
                        help="logs/blackbox/ içindeki en son kaydı kullan")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Çıktı dizini (varsayılan: logs/analysis/<timestamp>)")
    parser.add_argument("--no-events", action="store_true",
                        help="Olay (events_*.csv) dosyalarını yükleme")
    args = parser.parse_args()

    if args.latest:
        track_files = find_latest_tracks()
    else:
        track_files = [p for p in args.inputs if p.is_file()]

    if not track_files:
        print("[FAIL] Yüklenecek tracks_*.csv bulunamadı. --latest veya path verin.",
              file=sys.stderr)
        return 1

    event_files: list[Path] = []
    if not args.no_events:
        event_files = matched_event_files(track_files)

    print(f"Yükleniyor: {len(track_files)} tracks, {len(event_files)} events...")
    tracks = load_tracks(track_files)
    events = load_events(event_files)
    print(f"  → {len(tracks)} tespit, {len(events)} olay")

    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = PROJECT_ROOT / "logs" / "analysis" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Çelik Kubbe Görev Analizi — {len(tracks)} tespit, "
                 f"{len({r['Threat_ID'] for r in tracks})} hedef",
                 fontsize=14, fontweight="bold")

    ax_polar = fig.add_subplot(2, 3, 1, projection="polar")
    ax_level = fig.add_subplot(2, 3, 2)
    ax_pie = fig.add_subplot(2, 3, 3)
    ax_src = fig.add_subplot(2, 3, 4)
    ax_range = fig.add_subplot(2, 3, 5)
    ax_events = fig.add_subplot(2, 3, 6)

    panel_polar_tracks(ax_polar, tracks)
    panel_threat_level(ax_level, tracks)
    panel_class_pie(ax_pie, tracks)
    panel_source(ax_src, tracks)
    panel_range_time(ax_range, tracks)
    panel_events(ax_events, events)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    dashboard_path = out_dir / "dashboard.png"
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary_path = out_dir / "summary.txt"
    write_summary(summary_path, tracks, events, track_files, event_files)

    print(f"\n[OK] Dashboard : {dashboard_path}")
    print(f"[OK] Özet     : {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
