"""Real observations and explicit synthetic-observation adapters."""

from pathlib import Path
import hashlib
import json
import urllib.request
import numpy as np
from skimage import data as images
from skimage.transform import resize

NASA_URL = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"


def nasa_temperature(cache="data/cache"):
    """Annual global temperature anomalies (deg C relative to 1951–1980).

    Downloads NASA GISTEMP v4 CSV; preserves the raw file and SHA256 provenance.
    Network failures raise rather than substituting simulations.
    """
    root = Path(cache)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "nasa_gistemp.csv"
    if not path.exists():
        with urllib.request.urlopen(NASA_URL, timeout=45) as response:
            raw = response.read()
        if b"Year" not in raw or b"J-D" not in raw:
            raise ValueError("Unexpected NASA response")
        path.write_bytes(raw)
    import csv, io

    text = path.read_text()
    start = text.index("Year,")
    years = []
    values = []
    for row in csv.DictReader(io.StringIO(text[start:])):
        try:
            year = int(row["Year"])
            value = float(row["J-D"])
        except (ValueError, KeyError):
            continue
        years.append(year)
        values.append(value)
    if not years:
        raise ValueError("No complete annual records")
    from datetime import date

    metadata = {
        "accessed": str(date.today()),
        "source": NASA_URL,
        "units": "deg C anomaly relative to 1951–1980",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "count": len(years),
        "note": "Real annual series; missing-year mask in tutorial is simulated.",
    }
    (root / "nasa_gistemp.provenance.json").write_text(json.dumps(metadata, indent=2))
    return np.array(years), np.array(values), metadata


def camera_field(shape=(24, 24)):
    """Real camera photograph; resized intensity, NOT rainfall or calibrated radiance."""
    return resize(images.camera().astype(float) / 255, shape, anti_aliasing=True)


def load_sensor_csv(path, grid):
    """CSV: kind,value,noise,x0,y0,x1,y1,radius for normalized 2D locations.

    kind = point, line (average), footprint. Independent noise in field units.
    Caller must calibrate nonlinear sensors and project geographic coordinates.
    """
    import csv
    from .operators import points, lines, footprints

    if len(grid.shape) != 2:
        raise ValueError("CSV adapter currently supports 2D")
    rows = []
    values = []
    std = []
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            p = np.array([[float(row["x0"]), float(row["y0"])]])
            if row["kind"] == "point":
                A = points(grid, p)
            elif row["kind"] == "line":
                A = lines(grid, [[p[0], [float(row["x1"]), float(row["y1"])]]])
            elif row["kind"] == "footprint":
                A = footprints(grid, p, float(row["radius"]))
            else:
                raise ValueError("Unknown sensor kind")
            rows.append(A)
            values.append(float(row["value"]))
            std.append(float(row["noise"]))
    if (
        not rows
        or not np.isfinite(values).all()
        or not np.isfinite(std).all()
        or np.any(np.array(std) <= 0)
    ):
        raise ValueError("Nonempty finite sensor data and positive noise required")
    return np.vstack(rows), np.array(values), np.array(std)


NASA_GRID_URL = (
    "https://data.giss.nasa.gov/pub/gistemp/gistemp1200_GHCNv4_ERSSTv5.nc.gz"
)


def nasa_temperature_map(cache="data/cache", shape=(18, 24)):
    """Real GISTEMP gridded analysis, regional 2019–2023 mean, NOT raw sensors.

    Uses a finite midlatitude patch to avoid treating a sphere as a flat image.
    Block averages to a small grid; raises if cells are missing. The underlying
    NASA product is already spatially analyzed (1200 km smoothing).
    """
    import gzip, io
    from scipy.io import netcdf_file
    from datetime import date

    root = Path(cache)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "gistemp_grid.nc.gz"
    if not path.exists():
        with urllib.request.urlopen(NASA_GRID_URL, timeout=60) as response:
            raw = response.read()
        if raw[:2] != b"\x1f\x8b":
            raise ValueError("Unexpected gridded NASA response")
        path.write_bytes(raw)
    with netcdf_file(io.BytesIO(gzip.decompress(path.read_bytes())), mmap=False) as ds:
        lat = ds.variables["lat"][:].copy()
        lon = ds.variables["lon"][:].copy()
        time = ds.variables["time"][:].copy()
        origin = ds.variables["time"].units.decode().split("since ")[1].split()[0]
        dates = np.datetime64(origin) + time.astype("timedelta64[D]")
        select = (dates >= np.datetime64("2019-01-01")) & (
            dates < np.datetime64("2024-01-01")
        )
        variable = ds.variables["tempanomaly"]
        values = variable[select].copy().astype(float)
        fill = getattr(variable, "_FillValue", getattr(variable, "missing_value", 9999))
        values = unpack_netcdf(
            values,
            fill,
            getattr(variable, "scale_factor", 1),
            getattr(variable, "add_offset", 0),
        )
        latitude = (lat >= 20) & (lat <= 54)
        longitude = (lon >= -130) & (lon <= -84)
        patch = np.nanmean(values[:, latitude][:, :, longitude], axis=0)
    if not np.isfinite(patch).all():
        raise ValueError("Region contains missing cells")
    field = resize(patch, shape, anti_aliasing=True, preserve_range=True)
    metadata = {
        "source": NASA_GRID_URL,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "accessed": str(date.today()),
        "period": "2019–2023 mean",
        "region": "20–54 N, 130–84 W",
        "units": "deg C anomaly relative to 1951–1980",
        "note": "NASA spatial analysis with 1200km smoothing; synthetic sparse sensors. Flat regional index-grid approximation, not spherical geostatistics.",
    }
    (root / "nasa_grid.provenance.json").write_text(json.dumps(metadata, indent=2))
    return field, metadata


def unpack_netcdf(values, fill_value, scale_factor=1, add_offset=0):
    """Decode packed numeric fields; mask BEFORE applying scale and offset."""
    values = np.asarray(values, float).copy()
    values[np.isclose(values, fill_value)] = np.nan
    return values * float(scale_factor) + float(add_offset)
