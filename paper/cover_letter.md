# Cover Letter — Drones (MDPI)

**To:** Editors-in-Chief, Drones (MDPI)

**Re:** Submission — *"Adaptive Explainable Threat Scoring with Cross-Modality
Uncertainty Quantification for Multi-Sensor Counter-UAV Systems:
A Synthetic-to-Real Evaluation Framework"*

---

Dear Editors,

We are pleased to submit the above manuscript for consideration in *Drones* (MDPI).

## Why *Drones*?

This work directly addresses the journal's core scope: novel methodologies for
UAV/counter-UAV systems. Recent *Drones* publications on sensor fusion
(doi:10.3390/drones6100274), explainable AI for UAV applications
(doi:10.3390/drones8010004), and deep learning detection (doi:10.3390/drones7030150)
confirm the journal's receptivity to simulation-based evaluation frameworks
when physical hardware testing is not yet possible.

## Three Novel Contributions

**1. Adaptive XAI Threat Weights.**
Unlike existing fixed-weight fusion approaches, our system dynamically adjusts
threat scoring factors ($r$-, $v$-, $c$-factors) based on scene complexity,
sensor confidence, and environmental context. SHAP-based attribution provides
operator-interpretable explanations for every engagement decision.

**2. Cross-Modality Uncertainty Quantification.**
We introduce a Bayesian framework combining FMCW radar SNR and YOLOv11m
detection confidence into calibrated fusion decisions with explicit uncertainty
bounds. This directly addresses the reviewer concern "how is explainability
measured?" — uncertainty quantifies when the system should defer to the operator.

**3. Physics-Based Synthetic Evaluation Framework.**
A high-fidelity radar simulator implementing the full radar range equation,
Swerling Case 1 fluctuation model, flat-earth multipath, sinc² antenna pattern,
and CA-CFAR detection is released with the paper. This enables rigorous
evaluation without physical hardware access, with Monte Carlo statistical
validation ($N = 50$, 1000 tracker evaluations, Bonferroni-corrected Wilcoxon
tests).

## Key Results

| Metric | Çelik Kubbe (Ours) | Best Baseline (DeepSORT) |
|--------|-------------------|--------------------------|
| Mean MOTA | **0.645 ± 0.18** | −0.038 |
| Multi-threat MOTA | **0.655** | −0.008 |
| Significance | — | p < 0.0001 (corrected) |
| Effect size (Cliff's δ) | — | 0.947 (large) |

## Simulation-Only Strategy

We acknowledge the absence of physical hardware validation and address this
transparently:
- The paper explicitly discusses the simulation-to-real gap (Section 7.1)
  and rates the current implementation at TRL 4.
- The physics-based simulator is validated against closed-form predictions
  ($R^4$ law, CA-CFAR PFA), not just empirically tuned.
- Planned future work includes Drone-vs-Bird cross-dataset evaluation and
  Jetson Orin field deployment, described in Section 7.2.

*Drones* has published simulation-based C-UAV research; we believe this work
meets the standard for such contributions through its statistical rigor,
code reproducibility, and transparent limitation disclosure.

## Reproducibility

Complete source code, synthetic datasets, Docker environment, and parameter
seeds are released with the paper (GitHub link upon acceptance).
All experiments are deterministically reproducible via `random.seed()` controls.

## Conflict of Interest

The authors declare no conflicts of interest. This work was not funded by any
defense contractor or government weapons program.

We believe this manuscript makes a valuable contribution to the counter-UAV
research community and look forward to the reviewers' assessment.

Sincerely,

**Kürşat Kahya**  
Çukurova Science and Art Center, Adana, Turkey  
kahyakursat1@gmail.com

---

*Word count: ~250 (abstract) + ~5000 (body) = within Drones 10-15 page guideline.*
