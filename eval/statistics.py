"""
eval/statistics.py — Pairwise istatistiksel anlamlılık testleri.

Yöntemler:
  - Wilcoxon signed-rank test (eşleşik): aynı senaryolarda tracker karşılaştırması
  - Mann-Whitney U test (bağımsız): genel dağılım karşılaştırması
  - Bonferroni düzeltmesi: çoklu karşılaştırma için FWER kontrolü

Referans: Demsar, "Statistical Comparisons of Classifiers", JMLR 2006.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class PairwiseResult:
    tracker_a: str
    tracker_b: str
    scenario: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    statistic: float
    p_value: float
    p_value_corrected: float
    significant: bool
    test_name: str
    effect_size: float


def _safe_mean(x: list[float]) -> float:
    arr = np.array([v for v in x if np.isfinite(v)])
    return float(np.mean(arr)) if len(arr) > 0 else float("nan")


def _safe_std(x: list[float]) -> float:
    arr = np.array([v for v in x if np.isfinite(v)])
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's Delta: non-parametrik etki büyüklüğü, [-1, 1].
    |d| < 0.147 küçük, < 0.33 orta, >= 0.474 büyük.
    """
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    greater = sum(1 for ai in a for bi in b if ai > bi)
    less = sum(1 for ai in a for bi in b if ai < bi)
    return (greater - less) / (len(a) * len(b))


def pairwise_wilcoxon(
    results_by_tracker: dict[str, dict[str, list[float]]],
    alpha: float = 0.05,
    use_bonferroni: bool = True,
) -> list[PairwiseResult]:
    """
    Tracker çiftleri arasında Wilcoxon signed-rank test.

    Parametreler
    -----------
    results_by_tracker: {tracker_name: {scenario: [mota_values...]}}
    alpha: anlamlılık eşiği (Bonferroni öncesi)
    use_bonferroni: Bonferroni düzeltmesi uygula

    Döndürür
    --------
    list[PairwiseResult]
    """
    trackers = list(results_by_tracker.keys())
    pairs = list(itertools.combinations(trackers, 2))
    scenarios = list(next(iter(results_by_tracker.values())).keys())

    all_results = []
    n_comparisons = len(pairs) * len(scenarios)
    alpha_corrected = alpha / n_comparisons if use_bonferroni else alpha

    for t_a, t_b in pairs:
        for scenario in scenarios:
            vals_a = results_by_tracker[t_a].get(scenario, [])
            vals_b = results_by_tracker[t_b].get(scenario, [])

            fin_a = np.array([v for v in vals_a if np.isfinite(v)])
            fin_b = np.array([v for v in vals_b if np.isfinite(v)])

            n = min(len(fin_a), len(fin_b))
            if n < 5:
                stat, p = float("nan"), float("nan")
                test_name = "wilcoxon_skipped"
            else:
                try:
                    stat, p = stats.wilcoxon(fin_a[:n], fin_b[:n],
                                             alternative="two-sided",
                                             zero_method="wilcox")
                    test_name = "wilcoxon"
                except Exception:
                    stat, p = float("nan"), float("nan")
                    test_name = "wilcoxon_error"

            effect = _cliffs_delta(fin_a, fin_b)
            p_corr = min(float(p) * n_comparisons, 1.0) if np.isfinite(p) else float("nan")
            sig = bool(np.isfinite(p_corr) and p_corr < alpha)

            all_results.append(PairwiseResult(
                tracker_a=t_a,
                tracker_b=t_b,
                scenario=scenario,
                n_a=len(fin_a),
                n_b=len(fin_b),
                mean_a=_safe_mean(list(fin_a)),
                mean_b=_safe_mean(list(fin_b)),
                std_a=_safe_std(list(fin_a)),
                std_b=_safe_std(list(fin_b)),
                statistic=float(stat),
                p_value=float(p),
                p_value_corrected=p_corr,
                significant=sig,
                test_name=test_name,
                effect_size=effect,
            ))

    return all_results


def pairwise_mannwhitney(
    results_by_tracker: dict[str, dict[str, list[float]]],
    alpha: float = 0.05,
    use_bonferroni: bool = True,
) -> list[PairwiseResult]:
    """
    Bağımsız gruplar için Mann-Whitney U testi.
    Wilcoxon'ın eşleşik olmayan versiyonu.
    """
    trackers = list(results_by_tracker.keys())
    pairs = list(itertools.combinations(trackers, 2))
    scenarios = list(next(iter(results_by_tracker.values())).keys())

    all_results = []
    n_comparisons = len(pairs) * len(scenarios)
    alpha_corrected = alpha / n_comparisons if use_bonferroni else alpha

    for t_a, t_b in pairs:
        for scenario in scenarios:
            vals_a = results_by_tracker[t_a].get(scenario, [])
            vals_b = results_by_tracker[t_b].get(scenario, [])

            fin_a = np.array([v for v in vals_a if np.isfinite(v)])
            fin_b = np.array([v for v in vals_b if np.isfinite(v)])

            if len(fin_a) < 3 or len(fin_b) < 3:
                stat, p = float("nan"), float("nan")
                test_name = "mannwhitney_skipped"
            else:
                try:
                    stat, p = stats.mannwhitneyu(fin_a, fin_b, alternative="two-sided")
                    test_name = "mannwhitney"
                except Exception:
                    stat, p = float("nan"), float("nan")
                    test_name = "mannwhitney_error"

            effect = _cliffs_delta(fin_a, fin_b)
            p_corr = min(float(p) * n_comparisons, 1.0) if np.isfinite(p) else float("nan")
            sig = bool(np.isfinite(p_corr) and p_corr < alpha)

            all_results.append(PairwiseResult(
                tracker_a=t_a,
                tracker_b=t_b,
                scenario=scenario,
                n_a=len(fin_a),
                n_b=len(fin_b),
                mean_a=_safe_mean(list(fin_a)),
                mean_b=_safe_mean(list(fin_b)),
                std_a=_safe_std(list(fin_a)),
                std_b=_safe_std(list(fin_b)),
                statistic=float(stat),
                p_value=float(p),
                p_value_corrected=p_corr,
                significant=sig,
                test_name=test_name,
                effect_size=effect,
            ))

    return all_results


def aggregate_by_tracker_scenario(
    mc_results,
    metric: str = "mota",
) -> dict[str, dict[str, list[float]]]:
    """
    MCResult listesini {tracker: {scenario: [metric_values]}} sözlüğüne dönüştürür.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for r in mc_results:
        val = getattr(r, metric, None)
        if val is None:
            continue
        out.setdefault(r.tracker, {}).setdefault(r.scenario, []).append(float(val))
    return out


def significance_summary(pairwise: list[PairwiseResult]) -> str:
    """LaTeX / makale için özet tablo."""
    lines = ["Pairwise Statistical Significance (Wilcoxon + Bonferroni)"]
    lines.append(f"{'Tracker A':<18} vs {'Tracker B':<18} {'Scenario':<16} "
                 f"{'p_corr':>8} {'Sig':>4} {'d_cliff':>8}")
    lines.append("-" * 80)
    for r in pairwise:
        sig_str = "YES" if r.significant else "no"
        p_str = f"{r.p_value_corrected:.4f}" if np.isfinite(r.p_value_corrected) else "N/A"
        d_str = f"{r.effect_size:+.3f}" if np.isfinite(r.effect_size) else "N/A"
        lines.append(f"{r.tracker_a:<18}    {r.tracker_b:<18} {r.scenario:<16} "
                     f"{p_str:>8} {sig_str:>4} {d_str:>8}")
    return "\n".join(lines)


def to_latex(pairwise: list[PairwiseResult], caption: str = "") -> str:
    """Makale tablosu: p-value + etki büyüklüğü."""
    rows = []
    for r in pairwise:
        sig_str = r"$\checkmark$" if r.significant else "--"
        p_str = f"{r.p_value_corrected:.4f}" if np.isfinite(r.p_value_corrected) else "N/A"
        d_str = f"{r.effect_size:+.3f}" if np.isfinite(r.effect_size) else "N/A"
        rows.append(
            f"{r.tracker_a} vs {r.tracker_b} & {r.scenario} & "
            f"{r.mean_a:.3f} & {r.mean_b:.3f} & "
            f"{p_str} & {sig_str} & {d_str} \\\\"
        )

    body = "\n".join(rows)
    cap = caption or "Pairwise significance test results (Wilcoxon + Bonferroni)."
    return rf"""
\begin{{table}}[ht]
\centering
\caption{{{cap}}}
\label{{tab:significance}}
\small
\begin{{tabular}}{{llccccr}}
\toprule
Comparison & Scenario & $\overline{{MOTA}}_A$ & $\overline{{MOTA}}_B$ & $p_{{corr}}$ & Sig. & Cliff's $\delta$ \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
""".strip()


def save_significance_table(
    pairwise: list[PairwiseResult],
    out_dir: Path,
    caption: str = "",
) -> None:
    """Wilcoxon tablosunu CSV + LaTeX olarak kaydeder."""
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "significance_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fields = ["tracker_a", "tracker_b", "scenario", "n_a", "n_b",
                  "mean_a", "mean_b", "std_a", "std_b",
                  "p_value", "p_value_corrected", "significant",
                  "effect_size", "test_name"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        from dataclasses import asdict
        for r in pairwise:
            d = asdict(r)
            writer.writerow({k: d[k] for k in fields})

    tex_path = out_dir / "significance_table.tex"
    tex_path.write_text(to_latex(pairwise, caption), encoding="utf-8")
    print(f"  Significance CSV: {csv_path}")
    print(f"  Significance TeX: {tex_path}")


if __name__ == "__main__":
    from pathlib import Path
    from eval.monte_carlo import load_results, RESULTS_DIR
    results = load_results()
    by_tracker = aggregate_by_tracker_scenario(results, "mota")
    pw = pairwise_wilcoxon(by_tracker)
    print(significance_summary(pw))
    save_significance_table(pw, RESULTS_DIR, "Pairwise Wilcoxon + Bonferroni (N=50).")
