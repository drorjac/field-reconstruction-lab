# From pixels to physical fields

## 1. A pixel is a measurement

A continuous field f(r,t) may describe temperature, rain rate, density, or optical
intensity. A stored image is a discretization. A camera pixel integrates light
through optics, over an area, and over exposure time; it is not generally an
exact point sample. We use nodal grids as a transparent approximation.
All normalized coordinates use array-axis order: in 2D `(row, column)`.

A point measurement is f(r_i). A line average is
`y_i = (1/L_i) ∫_link f(r) ds + ε_i`. A footprint measurement is a spatial
weighted average. A projection is an integral through a field. These are
linear only after appropriate calibration. A common discrete form is

`y = A x + ε`, `ε ~ N(0,R)`.

Rows of A describe support. The adjoint Aᵀ spreads measurements back to their
support; it is not an inverse. Two links crossing at a point do not measure
the field at that intersection. Collapsing a link to its midpoint discards
information and changes the likelihood.

## 2. Sampling and signal processing

The discrete Fourier transform expresses a sampled field in sinusoidal modes.
Sampling above twice the highest frequency avoids aliasing only under ideal
band-limited assumptions. Finite pixels add a low-pass transfer function.
Blur attenuates high frequencies; deconvolution amplifies noise where that
transfer function is small. Interpolation is therefore a prior-dependent
inverse problem, not a way to create missing information.

Convolution and Fourier multiplication describe shift-invariant optics.
Our point/line/footprint matrices need not be shift invariant. The gradient
matrix D uses spacing 1/(n−1), so derivative penalties have domain-aware scale.
It uses forward differences with no wraparound; boundaries are not periodic.

## 3. Identifiability

If A has fewer independent rows than columns, any h with Ah=0 can be added to
x without changing the observations. A prior selects among these possibilities.
A single 2D projection of a 3D volume cannot identify the volume. Show two
different volumes with the same column sums before interpreting an image.
Dense operators and kernels here are limited to teaching grids: A needs O(mn)
storage, a dense GP needs O(n²), and dense regularized solves need O(n³).

## 4. Deterministic estimators

**IDW:** weighted average of point observations with weights 1/dᵖ. Useful
baseline, no physical observation model and no uncertainty distribution.

**Tikhonov:** minimize
`½ ||R^(-1/2)(Ax-y)||² + (α/2)||Dx||²`.
Normal equations are `(AᵀR⁻¹A + αDᵀD)x=AᵀR⁻¹y`.
The small numerical ridge prevents singular solves. Choosing α on the test
field is leakage. Use held-out sensors or separate simulated validation fields.

**TV:** replace the squared gradient by `α Σ sqrt((Dx)_j² + δ²)`.
This is smoothed anisotropic TV; it favors piecewise smooth/constant structure.
The optimizer reports its termination status. A finite iterate is not proof
of convergence. TV may erase small structures and create staircasing.

**Heat/Landweber:** gradient descent on data fidelity plus quadratic roughness.
The stable step is below 2/λ_max of the Hessian. Unconstrained fields can become
negative; rainfall applications need an explicit nonnegative likelihood/model
or constrained optimizer rather than silently clipping evaluation outputs.

## 5. Bayesian reconstruction

Let x ~ N(0,K) with an RBF spatial covariance and ε ~ N(0,R).
For arbitrary linear observations, Gaussian conditioning gives

`μ = K Aᵀ (A K Aᵀ + R)⁻¹ y`

`Σ = K − K Aᵀ (A K Aᵀ + R)⁻¹ A K`.

This handles line averages and heterogeneous sensors directly. The code uses
Cholesky solves. The returned standard deviation concerns the latent field;
future noisy sensors have additional measurement variance. Fixed length scale,
variance, and noise omit hyperparameter uncertainty. An incorrect kernel can
be confidently wrong. Zero prior mean is a teaching choice; center physical
data or provide a scientifically justified mean in research extensions.

The Tikhonov estimate can also be interpreted as a Gaussian-prior MAP estimate.
Thus “Bayesian” and “non-Bayesian” are different inferential treatments, not
necessarily completely different point estimates. We keep their objectives
and uncertainty claims explicit.

## 6. Supervised learning

The CNN maps normalized backprojection and sensor coverage to a field.
Targets come from independent seeded synthetic fields with smooth blobs,
fronts, and waves. Spatial train/test leakage is avoided by splitting entire
fields, not individual pixels. Geometry is fixed during this training demo;
the network is not certified for new layouts. Coverage and backprojection
are a compressed representation; different measurements can map to the same
condition. A geometry-aware network could retain A or graph structure.

## 7. Two meanings of diffusion

Heat diffusion smooths a field according to ∂u/∂t = κΔu. It is deterministic.
A DDPM corrupts examples through
`x_t = sqrt(ᾱ_t)x_0 + sqrt(1−ᾱ_t)ε`
and trains a neural network to predict ε given x_t, time, and conditioning.
The implemented reverse process uses the standard Gaussian DDPM posterior
coefficients and learned noise predictions. Conditioning here is sensor
backprojection and coverage. The optional affine correction is a heuristic
blend toward observed data; with noisy y it is not exact Bayesian posterior
sampling. A small teaching DDPM may underperform classical methods.

Sample spread measures variability of this trained sampler. It is not a
calibrated credible interval. Verify coverage on independent fields before
making uncertainty claims; uncertainty can collapse or be excessive.

## 8. Physical sensors: CMLs and beyond

For commercial microwave links, attenuation is related to rain approximately
through a power law k=a Rᵇ, integrated along the propagation path. Wet antennas,
dry baselines, frequency, polarization, and path heterogeneity matter.
Averaging R is not equivalent to inverting mean attenuation when b≠1.
Our line operator models calibrated line-average field measurements, not raw
radio attenuation. The old GMZ work is preserved in `Iterative/` for comparison.

Other examples: gauges sample near points; radar/satellite footprints average
volumes or areas; tomography measures rays; thermal cameras measure band
radiance. Project lat/lon to a metric coordinate system, preserve units and
support, quantify uncertainty, and model calibration before combining them.

## 9. Evaluation

Field RMSE/MAE require a full reference. Real sensor datasets often lack one.
Use held-out sensors as well as consistency on observed sensors; the latter
alone rewards overfitting. Report noise, sensor counts, field dimensions,
train/validation/test seeds, and calibration. Compare algorithms with the same
observations when claiming rankings. The default IDW demo uses points only
and is labeled accordingly. Benchmark tools support a controlled point-only
comparison. Repeat multiple fields and layouts, report distributions, and
separate tuning from final evaluation.

## Reading

- [Rasmussen & Williams, GPML, Chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf): Gaussian conditioning and regression.
- [Ho, Jain & Abbeel (2020), DDPM](https://arxiv.org/abs/2006.11239): generative diffusion and noise-prediction learning.
- [scikit-image Radon tutorial](https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html): sinograms and filtered backprojection.
- [NASA GISTEMP](https://data.giss.nasa.gov/gistemp/): real temperature anomalies, baseline and uncertainty products.
