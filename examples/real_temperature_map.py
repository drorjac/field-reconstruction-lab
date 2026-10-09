"""Real 2D physical field, simulated point/line/area observations."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fieldlab import Grid, tikhonov, gaussian_process, total_variation
from fieldlab.data import nasa_temperature_map
from fieldlab.fields import sensor_layout
from fieldlab.operators import observe
from fieldlab.benchmark import metrics


def main():
    truth, metadata = nasa_temperature_map(shape=(18, 24))
    grid = Grid(truth.shape)
    A, _ = sensor_layout(grid, 45, 20, 10, seed=90)
    y = observe(A, truth, noise=0.08, seed=91)
    models = {
        "Tikhonov": tikhonov(grid, A, y, noise=0.08, alpha=0.1),
        "GP": gaussian_process(grid, A, y, noise=0.08, length_scale=0.2),
        "TV": total_variation(grid, A, y, noise=0.08),
    }
    root = Path("results/temperature_map")
    root.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for ax, (name, f) in zip(
        axes, [("NASA reference", truth)] + [(n, r.mean) for n, r in models.items()]
    ):
        im = ax.imshow(
            f,
            origin="lower",
            extent=(-130, -84, 20, 54),
            vmin=truth.min(),
            vmax=truth.max(),
            cmap="coolwarm",
        )
        ax.set_title(name)
        fig.colorbar(im, ax=ax, label="°C anomaly", shrink=0.6)
    fig.savefig(root / "temperature_map.png", dpi=150)
    report = {
        "provenance": metadata,
        "scores": {n: metrics(truth, r.mean, A, y) for n, r in models.items()},
        "warning": "Reconstruction of a real analyzed field with simulated sensors; not validation on actual independent stations/CMLs.",
    }
    (root / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
