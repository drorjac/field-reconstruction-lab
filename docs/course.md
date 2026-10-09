# Teaching path

Prerequisites: Python/NumPy, basic linear algebra, derivatives, elementary
probability. Work through the notebooks in order. Every lesson has theory,
a runnable experiment, a figure, an exercise, and a checkable outcome.

| Lesson | Question | Main experiment | Expected understanding |
|---|---|---|---|
| 01 Pixels and sampling | What is a pixel? | Aliasing, FFT, blur | Sampling loses information; smoothing changes resolution |
| 02 Forward operators | What does each sensor observe? | Point, line, footprint, adjoint | Geometry belongs in A; averages preserve constants |
| 03 Classical reconstruction | What can regularization recover? | IDW, L2, TV, heat | Prior assumptions change reconstructions |
| 04 Bayesian fields | Where should we trust the image? | GP means and standard deviations | Conditioning on linear support; uncertainty depends on assumptions |
| 05 Neural reconstruction | Can examples teach a prior? | Train a CNN on independent fields | Training distribution, fixed geometry, generalization |
| 06 Generative diffusion | Can we generate multiple plausible fields? | Train and sample conditional DDPM | Noise prediction, reverse sampling, uncalibrated spread |
| 07 Real observations | How do examples transfer? | NASA 1D series and real 2D photograph | Units, provenance, masking, out-of-distribution errors |
| 08 Projections and 3D | Why is tomography hard? | Radon/FBP and 3D column sums | Nullspaces, view diversity, discretization |
| 09 Sensor fusion and CML | How do sensors differ? | Heterogeneous noise and support | Weighting and nonlinear calibration |
| 11 Real physical map | What does a real analyzed field show? | NASA regional temperature map | Analysis provenance, physical units, simulated sensors |
| 12 Physical heat fields | How does a PDE define a field? | Analytic 1D/2D heat modes | Boundary conditions and equilibrium priors |
| 10 Evaluation | What makes a result convincing? | Independent multi-seed benchmarks | Same data budgets, held-out measurements, leakage |

Suggested schedule: one lesson per week, with two sessions for diffusion.
CPU quick runs are deliberately small. Use longer runs as experiments, not
as presumed improvements. Tutors should require uncertainty calibration and
measurement-unit explanations before accepting “better” model claims.

Final assignment: reconstruct a temperature or rainfall field from points,
line measurements, and area measurements. Compare five estimators on identical
sensor budgets, tune only on validation cases, include an out-of-distribution
field, and explain at least one failure with an operator nullspace.
