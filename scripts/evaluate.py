"""Matched point-data benchmark; no tuning on evaluation truth."""

import argparse, json
from pathlib import Path
import numpy as np
from fieldlab import (
    Grid,
    points,
    idw,
    tikhonov,
    gaussian_process,
    total_variation,
    heat_reconstruction,
)
from fieldlab.fields import simulate
from fieldlab.operators import observe
from fieldlab.benchmark import metrics

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=5)
args = p.parse_args()
if args.seeds < 1:
    p.error("positive seeds required")
grid = Grid((16, 16))
records = []
for seed in range(args.seeds):
    truth = simulate(grid, 5000 + seed, kind=("smooth", "front", "wave")[seed % 3])
    rng = np.random.default_rng(6000 + seed)
    locations = rng.uniform(0, 1, (50, 2))
    A = points(grid, locations)
    y = observe(A, truth, 0.03, 7000 + seed)
    held = points(grid, rng.uniform(0, 1, (25, 2)))
    held_y = observe(held, truth, 0.03, 8000 + seed)
    models = {
        "IDW": idw(grid, locations, y),
        "Tikhonov": tikhonov(grid, A, y),
        "GP": gaussian_process(grid, A, y),
        "TV": total_variation(grid, A, y),
        "Heat": heat_reconstruction(grid, A, y),
    }
    for name, res in models.items():
        r = {
            "seed": seed,
            "model": name,
            **metrics(truth, res.mean, A, y),
            "heldout_sensor_rmse": float(
                np.sqrt(np.mean((held @ res.mean.ravel() - held_y) ** 2))
            ),
            "info": res.info,
        }
        if res.std is not None:
            r["pointwise_95_coverage"] = float(
                np.mean(np.abs(res.mean - truth) <= 1.96 * res.std)
            )
        records.append(r)
root = Path("results/evaluation")
root.mkdir(parents=True, exist_ok=True)
summary = {
    name: {
        key: {
            "mean": float(np.mean([r[key] for r in records if r["model"] == name])),
            "std": float(np.std([r[key] for r in records if r["model"] == name])),
        }
        for key in ["rmse", "mae", "heldout_sensor_rmse"]
    }
    for name in models
}
(root / "metrics.json").write_text(
    json.dumps(
        {
            "note": "All models use identical points; fixed hyperparameters, no tuning. Independent fields/layouts.",
            "records": records,
            "summary": summary,
        },
        indent=2,
    )
)
print(json.dumps(summary, indent=2))
