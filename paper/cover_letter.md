# Cover Letter — Drones (MDPI)

**To:** Editors-in-Chief, *Drones* (MDPI)

**Re:** Original Research Submission —
*"Context-Sensitive Explainable Threat Scoring with Confidence-Weighted
Cross-Modal Fusion for Multi-Sensor Counter-UAV Systems:
A Physics-Based Simulation Framework"*

---

Dear Editors,

I am pleased to submit the above manuscript for consideration as an original
research article in *Drones* (MDPI).

## Scope Alignment

This work directly addresses *Drones*' core scope on counter-UAV (C-UAV) system
methodologies. It presents **ContextFusion**: a context-sensitive explainable AI
(XAI) framework for multi-sensor threat tracking that combines X-band FMCW radar
with YOLOv11m camera detections. The manuscript contributes (i) a novel
confidence-weighted log-linear MAP fusion model, (ii) context-sensitive threat
weights with SHAP-based decision-layer attribution, and (iii) a reproducible
physics-based simulation suite enabling rigorous evaluation without physical
hardware access.

## Three Novel Contributions

**1. Context-Sensitive XAI Threat Scoring.**
Threat weights ($r$-, $v$-, $c$-factors) adapt dynamically to scene complexity,
multi-threat density, and sensor dropout conditions. SHAP attribution surfaces
the dominant scoring driver (*range*, *velocity*, or *classification confidence*)
at each timestep, moving XAI from the perception layer to the
**decision layer** — enabling real-time operator trust calibration in
high-stakes C-UAV engagement scenarios.

**2. Confidence-Weighted Cross-Modal Fusion.**
A log-linear MAP fusion model propagates YOLOv11m detection confidence and
radar CA-CFAR SNR into a composite association score with explicit theoretical
grounding. The framework is designed to be extended with learned fusion weights
(attention-based or Bayesian network) once real paired radar-camera data becomes
available.

**3. Physics-Based Reproducible Evaluation Framework.**
A high-fidelity radar simulator implementing the radar range equation, Swerling
Case 1 RCS fluctuations, flat-earth multipath, and CA-CFAR detection enables
Monte Carlo evaluation ($N = 50$ runs per scenario, $N_\text{total} = 200$
pooled) without physical hardware. The simulator is validated against closed-form
predictions ($R^4$ range law, CA-CFAR $P_\text{FA} = 10^{-6}$).

## Key Results

| Metric | ContextFusion (Ours) | DeepSORT (best baseline) |
|--------|---------------------|--------------------------|
| Mean MOTA (all scenarios) | **0.645** [95% CI: 0.618–0.672] | −0.038 |
| Sensor Dropout MOTA | **0.569** | −0.068 |
| Low-SNR MOTA | **0.675** | −0.048 |
| Significant comparisons | — | 12 of 16 (Bonferroni-corrected Wilcoxon) |
| Effect size | — | Cliff's δ > 0.70 (large) |

## Simulation-Only Transparency

I acknowledge the absence of physical hardware validation and address it
explicitly in Section 7 (Limitations):

- The system is assessed at **TRL 4** (component validation in laboratory
  environment), with a concrete three-phase field validation programme
  (AERIS-10 static-range RCS characterisation → DJI Matrice 300 dynamic
  flight trials → Jetson Orin INT8 pipeline benchmarking) described as
  future work.
- Monte Carlo results reflect *within-simulator variability*; the paper
  explicitly states they should not be interpreted as guarantees of
  real-world generalisation.
- A cross-dataset evaluation on the Drone-vs-Bird benchmark (889 frames,
  real imagery) provides a partial, independent generalisation check
  (Section 6.4).

*Drones* has published simulation-based C-UAV research where physical
testbeds were unavailable; this work meets that standard through statistical
rigour, physics-validated simulation, and transparent limitation disclosure.

## Reproducibility

Complete source code, evaluation scripts, and configuration are available at:
**https://github.com/kahyakursat1-cloud/celik-kubbe**
(reviewer access enabled; will be made fully public upon acceptance).
All Monte Carlo experiments are deterministically reproducible via
`random.seed()` controls specified in `config.yaml`.

## Conflict of Interest

The author declares no conflicts of interest. This research received no
external funding. The proposed framework is designed exclusively for
defensive civil airspace protection applications.

I believe this manuscript makes a substantive contribution to the C-UAV
research community through its combination of context-aware XAI,
physics-principled fusion, and reproducible open-source evaluation.
I look forward to the reviewers' assessment.

Sincerely,

**Kürşat Kahya**
Çukurova Science and Art Center, Adana, Turkey
kahyakursat1@gmail.com
