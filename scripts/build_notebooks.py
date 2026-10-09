"""Build self-contained tutorial sources; do not overwrite verified notebooks casually."""

from pathlib import Path
import nbformat as nb

setup = """from pathlib import Path
import os
if Path.cwd().name == 'tutorials': os.chdir('..')
import numpy as np
import matplotlib.pyplot as plt
from fieldlab import Grid, points, lines, footprints, projections, combine
from fieldlab.fields import simulate, sensor_layout
from fieldlab.operators import observe
from fieldlab.models import idw, tikhonov, gaussian_process, total_variation, heat_reconstruction
"""
lessons = [
    (
        "01_pixels_sampling",
        "Pixels, sampling, and Fourier analysis",
        """A digital image represents samples of a continuous field. Point sampling and area averaging are different operators. The Nyquist condition requires assumptions about band limits; sharper edges violate them. A Fourier spectrum shows which scales a blur removes.

**Objective:** explain why high-frequency fields can look low-frequency after coarse sampling.""",
        """from scipy.ndimage import gaussian_filter
x=np.linspace(0,1,512,endpoint=False)
signal=np.sin(2*np.pi*37*x)
coarse=np.arange(0,512,32)
image=simulate(Grid((64,64)),seed=1,kind='front')
blur=gaussian_filter(image,2)
fig,axes=plt.subplots(1,3,figsize=(12,3))
axes[0].plot(x,signal); axes[0].plot(x[coarse],signal[coarse],'o-'); axes[0].set_title('Aliasing: 37 cycles, 16 samples')
axes[1].imshow(blur); axes[1].set_title('Blurred front')
axes[2].imshow(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))); axes[2].set_title('Log Fourier magnitude')
plt.show()
assert np.std(blur)<np.std(image)
""",
        """**Exercise:** replace 37 cycles by 3, then 8. Change blur scale from 2 to 4. Explain both the frequency ambiguity and the loss of edge resolution. Why does a finite pixel footprint not simply behave as a point?""",
    ),
    (
        "02_sensor_operators",
        "What point, line, and footprint sensors observe",
        """Write y=Ax+ε. Each row of A describes support. Line quadrature computes an average rather than treating a measurement as a midpoint value. The adjoint spreads residuals to their supports; it does not invert A.

**Objective:** verify conservation of constants and understand an adjoint.""",
        """grid=Grid((20,20)); f=simulate(grid,3)
p=np.array([[.2,.4],[.8,.6]])
e=np.array([[[.1,.1],[.9,.8]],[[.2,.9],[.8,.1]]])
A=combine(points(grid,p),lines(grid,e),footprints(grid,p,.15))
assert np.allclose(A@np.ones(grid.size),1)
rng=np.random.default_rng(9); x=rng.normal(size=grid.size); v=rng.normal(size=A.shape[0])
assert np.allclose((A@x)@v,x@(A.T@v))
fig,axes=plt.subplots(1,3,figsize=(10,3))
axes[0].imshow(f); axes[0].set_title('Field')
axes[1].imshow(A[2].reshape(grid.shape)); axes[1].set_title('One line support')
axes[2].imshow((A.T@(A@f.ravel())).reshape(grid.shape)); axes[2].set_title('Backprojection, not inverse')
plt.show()
print('Measurements:',A@f.ravel())
""",
        """**Exercise:** measure a linear field analytically; a line average equals the mean of its endpoint values. Construct a curved field for which this fails. Compare quadrature with 8 and 128 samples. Check that reversing endpoints preserves an average.""",
    ),
    (
        "03_classical_models",
        "Classical reconstruction and regularization",
        """IDW uses points; Tikhonov minimizes data residual plus squared gradient; TV penalizes absolute gradients; heat regularization performs stable iterative descent. For a fair comparison here, all models receive the same point measurements.

**Objective:** see the influence of regularization on a sharp front.""",
        """grid=Grid((16,16)); truth=simulate(grid,22,kind='front')
p=np.random.default_rng(3).uniform(0,1,(45,2)); A=points(grid,p); y=observe(A,truth,.03,4)
models={'IDW':idw(grid,p,y),'Tikhonov':tikhonov(grid,A,y),'TV':total_variation(grid,A,y),'Heat':heat_reconstruction(grid,A,y)}
fig,axes=plt.subplots(1,5,figsize=(14,3))
axes[0].imshow(truth,vmin=0,vmax=1); axes[0].set_title('Truth')
for ax,(name,r) in zip(axes[1:],models.items()):
    ax.imshow(r.mean,vmin=0,vmax=1); ax.set_title(name)
    print(name,'RMSE',np.sqrt(np.mean((r.mean-truth)**2)),r.info)
plt.show()
""",
        """**Exercise:** choose alpha on three separate validation fields, then evaluate on a fourth field. Never choose it using the final truth. Increase noise and explain changes in smoothness. Which estimate preserves a front best? Check TV convergence diagnostics.""",
    ),
    (
        "04_bayesian_fields",
        "Bayesian conditioning on linear measurements",
        """A Gaussian prior x~N(0,K) and measurement noise ε~N(0,R) imply a Gaussian posterior. For a general operator, μ=KAᵀ(AKAᵀ+R)⁻¹y. Its covariance shrinks where measurements constrain the field. A fixed kernel does not express all scientific uncertainty.

**Objective:** distinguish uncertainty from error.""",
        """grid=Grid((18,18)); truth=simulate(grid,33)
A,layout=sensor_layout(grid,15,12,4,seed=4); y=observe(A,truth,.04,5)
r=gaussian_process(grid,A,y,noise=.04,length_scale=.18)
fig,axes=plt.subplots(1,3,figsize=(10,3))
for ax,name,f in zip(axes,['Posterior mean','Conditional std','Absolute error'],[r.mean,r.std,np.abs(r.mean-truth)]):
    im=ax.imshow(f); ax.set_title(name); fig.colorbar(im,ax=ax)
plt.show()
print('Pointwise 95% interval coverage:',np.mean(np.abs(r.mean-truth)<=1.96*r.std))
assert np.all(r.std>=0)
""",
        """**Exercise:** repeat for length scales .04, .2, and .8. Is high confidence always justified? Add a sensor and check that posterior variance cannot increase under fixed assumptions. Explain why pointwise coverage differs from simultaneous field coverage.""",
    ),
    (
        "05_neural_reconstruction",
        "Learn a reconstruction prior with a CNN",
        """A CNN maps normalized backprojection and coverage to a field. Training examples are distinct whole fields, not pixels from the test image. This compact network uses a fixed sensor layout. Its compressed condition does not contain all measurement information.

**Objective:** actually train a supervised model, inspect its loss, and test on an independent seed.""",
        """from fieldlab.neural import train_cnn,predict_cnn
grid=Grid((16,16)); A,_=sensor_layout(grid,seed=6)
net,loss=train_cnn(grid,A,steps=100,count=96,seed=1000)
truth=simulate(grid,9000,kind='front'); y=observe(A,truth,.03,9001)
r=predict_cnn(net,grid,A,y)
fig,axes=plt.subplots(1,3,figsize=(10,3))
axes[0].plot(loss); axes[0].set_title('Training MSE')
axes[1].imshow(truth); axes[1].set_title('Independent test field')
axes[2].imshow(r.mean); axes[2].set_title('CNN estimate')
plt.show()
print('RMSE:',np.sqrt(np.mean((r.mean-truth)**2)))
""",
        """**Exercise:** test a new sensor layout and report degradation. Train with several layouts as an extension. Compare against a GP using exactly the same observations. Add validation loss rather than treating training loss as generalization evidence.""",
    ),
    (
        "06_diffusion",
        "Train and sample a conditional DDPM",
        """Generative diffusion corrupts examples with Gaussian noise and learns to predict that noise. It is different from heat smoothing. Our reverse sampler conditions on sensor backprojection and includes heuristic measurement correction. This is a teaching conditional generator, not an exact noisy posterior sampler.

**Objective:** train noise prediction and inspect multiple possible fields and their spread.""",
        """from fieldlab.neural import ConditionalDiffusion
grid=Grid((16,16)); A,_=sensor_layout(grid,seed=7)
ddpm=ConditionalDiffusion(timesteps=30)
loss=ddpm.train(grid,A,steps=160,count=96,seed=2000)
truth=simulate(grid,9100); y=observe(A,truth,.03,9101)
r,draws=ddpm.reconstruct(grid,A,y,samples=6,seed=3000)
fig,axes=plt.subplots(1,4,figsize=(12,3))
axes[0].plot(loss); axes[0].set_title('Noise prediction MSE')
for ax,name,f in zip(axes[1:],['One sample','Sample mean','Uncalibrated spread'],[draws[0],r.mean,r.std]): ax.imshow(f); ax.set_title(name)
plt.show()
assert np.isfinite(draws).all()
print('Mean RMSE:',np.sqrt(np.mean((r.mean-truth)**2)))
""",
        """**Exercise:** compare guidance strengths 0, .05, and .5, keeping the seed fixed. Does fitting noisy data improve field error? Increase training to 1000 steps and evaluate several independent seeds. Explain why sample spread is not automatically a credible interval.""",
    ),
    (
        "07_real_data",
        "Real observations: climate and image intensity",
        """The photograph is a real image; its sparse sensors are synthetic. NASA annual temperature anomalies are actual climate observations, with a simulated missing-year mask. This notebook uses a cached NASA snapshot when available and otherwise downloads it. Network failure is not hidden by a synthetic substitute.

**Objective:** preserve provenance and units while comparing reconstruction to a reference.""",
        """from fieldlab.data import camera_field,nasa_temperature
grid=Grid((16,16)); image=camera_field(grid.shape); A,_=sensor_layout(grid,seed=8)
y=observe(A,image,.03,9); r=tikhonov(grid,A,y)
fig,axes=plt.subplots(1,2,figsize=(7,3)); axes[0].imshow(image,cmap='gray'); axes[1].imshow(r.mean,cmap='gray'); plt.show()
years,temp,metadata=nasa_temperature()
grid1=Grid((len(years),)); idx=np.arange(0,len(years),2)
A1=points(grid1,(idx/(len(years)-1))[:,None]); r1=gaussian_process(grid1,A1,temp[idx],noise=.08,length_scale=.08)
plt.figure(figsize=(10,3)); plt.plot(years,temp,label='NASA full series'); plt.plot(years,r1.mean,label='GP from alternate years'); plt.ylabel('Temperature anomaly °C'); plt.legend(); plt.show()
print(metadata)
""",
        """**Exercise:** use a consecutive missing decade instead of alternate years. Does random-mask accuracy predict gap accuracy? For the photograph, compare a model trained on blobs against sharp image edges. Cite NASA and distinguish assumed noise from official error bars.""",
    ),
    (
        "08_projections",
        "Tomography and 3D projections",
        """A projection sums along a ray. Multiple angles enable tomography; a single view has a large nullspace. Filtered backprojection uses a frequency ramp filter and backprojection. A Shepp–Logan phantom is simulated, not a real medical scan.

**Objective:** reconstruct a 2D phantom and prove why one 3D view is ambiguous.""",
        """from skimage.data import shepp_logan_phantom
from skimage.transform import resize,radon,iradon
phantom=resize(shepp_logan_phantom(),(64,64),anti_aliasing=True); theta=np.linspace(0,180,45,endpoint=False)
sino=radon(phantom,theta=theta,circle=True); rec=iradon(sino,theta=theta,circle=True)
grid=Grid((8,8,8)); volume=simulate(grid,19); P=projections(grid,0)
alternative=volume[::-1].copy()
assert np.allclose(P@volume.ravel(),P@alternative.ravel())
fig,axes=plt.subplots(1,3,figsize=(10,3))
for ax,name,f in zip(axes,['Phantom','FBP','3D projection'],[phantom,rec,(P@volume.ravel()).reshape(8,8)]): ax.imshow(f,cmap='magma'); ax.set_title(name)
plt.show()
print('FBP RMSE',np.sqrt(np.mean((phantom-rec)**2)))
""",
        """**Exercise:** reduce views to 8, restrict angles to 0–90°, and add noise. Explain streak artifacts. Construct a nonzero vector h with Ph=0. Why is showing a projected volume different from recovering that volume?""",
    ),
    (
        "09_sensor_fusion",
        "Heterogeneous sensors and physical calibration",
        """Combine different supports without pretending they are identical points. Noise controls their relative influence. CML attenuation involves the path integral of a power law aRᵇ; averaging rain then applying the power law can be biased.

**Objective:** reconstruct with heterogeneous noise and expose nonlinear averaging.""",
        """grid=Grid((16,16)); truth=simulate(grid,41)
A,layout=sensor_layout(grid,20,15,8,seed=10)
noise=np.r_[np.full(20,.02),np.full(15,.05),np.full(8,.10)]
y=observe(A,truth,noise,seed=11); r=gaussian_process(grid,A,y,noise=noise)
fig,axes=plt.subplots(1,2,figsize=(7,3)); axes[0].imshow(truth); axes[1].imshow(r.mean); plt.show()
rain=np.array([1.,9.]); exponent=1.6
print('Mean rain power:',rain.mean()**exponent,'Mean attenuation factor:',np.mean(rain**exponent))
assert not np.isclose(rain.mean()**exponent,np.mean(rain**exponent))
""",
        """**Exercise:** corrupt one sensor family with bias; explain why larger independent noise does not fix systematic bias. Add a calibration parameter or quality-control step. Load your own CSV using the documented schema, retaining units and source metadata.""",
    ),
    (
        "10_evaluation",
        "Evaluation without leakage",
        """Compare identical observations; tune hyperparameters on separate validation fields. Field RMSE requires reference truth, while held-out sensor errors can be computed without a dense truth map. Training residual alone rewards overfitting. Multiple independent fields and layouts reveal variability.

**Objective:** run a controlled multi-seed comparison with held-out sensors.""",
        """grid=Grid((12,12)); rows=[]
for seed in range(3):
    truth=simulate(grid,10000+seed,kind='front'); rng=np.random.default_rng(11000+seed)
    p=rng.uniform(0,1,(30,2)); A=points(grid,p); y=observe(A,truth,.03,12000+seed)
    H=points(grid,rng.uniform(0,1,(20,2))); hy=observe(H,truth,.03,13000+seed)
    estimators={'IDW':idw(grid,p,y),'L2':tikhonov(grid,A,y),'GP':gaussian_process(grid,A,y)}
    for name,r in estimators.items(): rows.append((name,np.sqrt(np.mean((r.mean-truth)**2)),np.sqrt(np.mean((H@r.mean.ravel()-hy)**2))))
for name in estimators:
    values=np.array([row[1:] for row in rows if row[0]==name]); print(name,'mean [field RMSE, held-out sensor RMSE]',values.mean(0),'std',values.std(0))
fig,ax=plt.subplots(); ax.boxplot([[r[1] for r in rows if r[0]==name] for name in estimators],tick_labels=list(estimators)); ax.set_ylabel('Field RMSE'); plt.show()
""",
        """**Exercise:** expand to all seven models, maintain identical measurement budgets, and train neural models only once on separate training fields. Report training time, inference time, uncertainty coverage, and an out-of-distribution front or new geometry. Explain where your experiment lacks statistical power.""",
    ),
]

for filename, title, theory, code, exercise in lessons:
    notebook = nb.v4.new_notebook(
        cells=[
            nb.v4.new_markdown_cell("# " + title + "\n\n" + theory),
            nb.v4.new_code_cell(setup),
            nb.v4.new_code_cell(code),
            nb.v4.new_markdown_cell(
                exercise
                + "\n\nSee [theory notes](../docs/theory.md) for assumptions and equations."
            ),
        ]
    )
    notebook.metadata["kernelspec"] = {
        "display_name": "Field Reconstruction Lab",
        "language": "python",
        "name": "fieldlab",
    }
    notebook.metadata["language_info"] = {"name": "python"}
    nb.write(notebook, Path("tutorials") / (filename + ".ipynb"))
