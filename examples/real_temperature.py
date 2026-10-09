"""Run from repository root: python -m examples.real_temperature."""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fieldlab.data import nasa_temperature
from fieldlab import Grid, points, tikhonov, gaussian_process


def main():
    years, values, provenance = nasa_temperature()
    grid = Grid((len(years),))
    rng = np.random.default_rng(10)
    indices = np.sort(
        rng.choice(len(years), size=int(0.65 * len(years)), replace=False)
    )
    A = points(grid, indices[:, None] / (len(years) - 1))
    y = values[indices]
    # Noise below is assumed reconstruction noise, not NASA's uncertainty product.
    estimates = {
        "Tikhonov": tikhonov(grid, A, y, noise=0.08, alpha=0.01),
        "GP": gaussian_process(grid, A, y, noise=0.08, length_scale=0.08),
    }
    root = Path("results/temperature")
    root.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(years, values, label="NASA full reference")
    ax.scatter(years[indices], y, s=10, label="Retained records")
    held = np.ones(len(years), bool)
    held[indices] = False
    scores = {}
    for name, result in estimates.items():
        ax.plot(years, result.mean, label=name)
        scores[name] = float(np.sqrt(np.mean((result.mean[held] - values[held]) ** 2)))
        if result.std is not None:
            ax.fill_between(
                years,
                result.mean - 1.96 * result.std,
                result.mean + 1.96 * result.std,
                alpha=0.15,
            )
    ax.set_ylabel("Global temperature anomaly (°C)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "temperature.png", dpi=150)
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "heldout_rmse": scores,
                "provenance": provenance,
                "noise_note": "Assumed .08°C; not official NASA error bars",
            },
            indent=2,
        )
    )
    print(scores)


if __name__ == "__main__":
    main()
