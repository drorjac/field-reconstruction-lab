# Data and reproducibility

## Included or downloadable examples

- Synthetic 1D/2D/3D normalized Gaussian-blob, front, and wave fields. These are
  educational spatial examples, not numerical solutions of weather dynamics.
- Real NASA GISTEMP annual temperature anomalies: downloader caches raw CSV,
  SHA256, units, source URL, and record count. Network errors remain errors.
  The missing-year mask is simulated. Noise in the reconstruction is assumed,
  not NASA's published observational uncertainty. NASA requires citation of
  the dataset page and Lenssen et al. (2024), doi:10.1029/2023JD040179.
- Real NASA gridded temperature analysis: regional 2019–2023 mean, in °C
  anomalies. Underlying NASA analysis uses 1200 km smoothing. Its sparse
  point/line/footprint sensors are simulated. Cache includes raw data, hash,
  access date, region, period, and units. This demonstrates reconstruction of
  a real physical field, not performance on actual independent stations.
- Real scikit-image camera photograph: grayscale intensity. The measurements
  are simulated and the image is resized. It is not a calibrated physical
  rainfall dataset. See [image provenance](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.camera).
- Shepp–Logan phantom: synthetic tomography reference, never described as a
  clinical scan. 3D projections use a simulated density volume.

## Your own real multi-sensor data

CSV schema (2D, one observation per row):

```csv
kind,value,noise,x0,y0,x1,y1,radius
point,0.4,0.03,0.2,0.6,,,
line,0.5,0.05,0.1,0.2,0.9,0.7,
footprint,0.6,0.08,0.5,0.5,,,0.15
```

These numbers illustrate the schema; they are not a real dataset.
`x0,y0` mean array axes 0,1 despite conventional Cartesian names.
Coordinates are normalized to [0,1] after projecting geographic locations.
`value` and `noise` must share field units, and noise is a positive standard
deviation. Lines contain calibrated averages, not raw attenuation.
Use `fieldlab.data.load_sensor_csv(path, grid)` and pass its A,y,std into
`tikhonov`, `gaussian_process`, or `total_variation`.

For real CML work, retain link endpoints, frequency, polarization, baseline
subtraction method, antenna correction, timestamps, quality flags, calibration
coefficients, and co-located gauge/radar provenance. The software supplies an
adapter but does not pretend that generic line-average simulations validate
raw CML retrieval. No private student measurements are redistributed.

## Environment

Create a fresh virtual environment and install from pyproject.toml. The
`requirements-tested.txt` records the environment used for checked results;
it is a record, not a cross-platform promise. Generated outputs are ignored
by Git. Figures and metrics should always be regenerated together.
