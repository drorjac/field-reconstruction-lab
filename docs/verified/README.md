# Verified examples

Verified on 2026-10-09; Python 3.14.8.

All **15 tests passed**. All **12 lesson notebooks executed without errors**, including training and sampling.

Five independent test fields, identical 50-point measurements for all seven estimators, fixed sensor layout. CNN training: 200 steps; conditional DDPM: 400 steps. No test-set hyperparameter tuning.

| Model | Mean field RMSE | Across-field std | Mean held-out sensor RMSE |
|---|---:|---:|---:|
| IDW | 0.1376 | 0.0219 | 0.1428 |
| Tikhonov | 0.1183 | 0.0197 | 0.1268 |
| GP | 0.1251 | 0.0451 | 0.1195 |
| TV | 0.1184 | 0.0268 | 0.1208 |
| Heat | 0.2425 | 0.0483 | 0.2374 |
| CNN | 0.0960 | 0.0158 | 0.1037 |
| DDPM | 0.1051 | 0.0148 | 0.1164 |

These are teaching-scale results, not a general model ranking. The learned prior matches the synthetic training distribution. Geometry remains fixed. Diffusion spread is uncalibrated; the weakly regularized heat baseline illustrates how good observed-sensor fit can coexist with poor field error.

NASA annual-series and map provenance (URLs, hashes, units) are retained in the JSON reports. The map is a real spatial analysis with synthetic measurements. The photograph is real intensity with synthetic measurements. For an actual CML validation study, supply calibrated, independently sourced sensor data.

Numeric reports are in this directory. Reproduce them using the README commands. Inference timings are machine- and concurrency-dependent.
