"""Seven models on identical point observations, independent test fields.

Train neural priors once for a fixed layout. Test fields differ from training
seeds. Hyperparameters are fixed before evaluation, not selected on test truth.
"""

import argparse, json, time
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
from fieldlab.neural import train_cnn, predict_cnn, ConditionalDiffusion
from fieldlab.benchmark import metrics

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=200)
p.add_argument("--seeds", type=int, default=5)
args = p.parse_args()
if args.steps < 1 or args.seeds < 1:
    p.error("steps and seeds must be positive")
grid = Grid((16, 16))
rng = np.random.default_rng(6000)
locations = rng.uniform(0, 1, (50, 2))
A = points(grid, locations)
start = time.perf_counter()
cnn, cnn_loss = train_cnn(grid, A, steps=args.steps)
cnn_seconds = time.perf_counter() - start
start = time.perf_counter()
ddpm = ConditionalDiffusion()
diffusion_loss = ddpm.train(grid, A, steps=args.steps * 2)
diffusion_seconds = time.perf_counter() - start
records = []
for seed in range(args.seeds):
    truth = simulate(grid, 5000 + seed, kind=("smooth", "front", "wave")[seed % 3])
    y = observe(A, truth, 0.03, 7000 + seed)
    held = points(grid, rng.uniform(0, 1, (25, 2)))
    held_y = observe(held, truth, 0.03, 8000 + seed)
    jobs = {
        "IDW": lambda: idw(grid, locations, y),
        "Tikhonov": lambda: tikhonov(grid, A, y),
        "GP": lambda: gaussian_process(grid, A, y),
        "TV": lambda: total_variation(grid, A, y),
        "Heat": lambda: heat_reconstruction(grid, A, y),
        "CNN": lambda: predict_cnn(cnn, grid, A, y),
        "DDPM": lambda: ddpm.reconstruct(grid, A, y, seed=9000 + seed)[0],
    }
    for name, fn in jobs.items():
        start = time.perf_counter()
        res = fn()
        row = {
            "seed": seed,
            "model": name,
            **metrics(truth, res.mean, A, y),
            "inference_seconds": time.perf_counter() - start,
            "heldout_sensor_rmse": float(
                np.sqrt(np.mean((held @ res.mean.ravel() - held_y) ** 2))
            ),
            "info": res.info,
        }
        if res.std is not None:
            row["pointwise_95_coverage"] = float(
                np.mean(np.abs(res.mean - truth) <= 1.96 * res.std)
            )
        records.append(row)
summary = {
    name: {
        key: {
            "mean": float(np.mean([r[key] for r in records if r["model"] == name])),
            "std": float(np.std([r[key] for r in records if r["model"] == name])),
        }
        for key in ["rmse", "mae", "heldout_sensor_rmse", "inference_seconds"]
    }
    for name in jobs
}
root = Path("results/learning_evaluation")
root.mkdir(parents=True, exist_ok=True)
report = {
    "note": "All seven estimators see identical points. Fixed layout; independent test fields; fixed hyperparameters. DDPM 95% coverage statistic is descriptive, not a calibrated credible interval.",
    "training_seconds": {"CNN": cnn_seconds, "DDPM": diffusion_seconds},
    "training_steps": args.steps,
    "records": records,
    "summary": summary,
}
(root / "metrics.json").write_text(json.dumps(report, indent=2))
print(json.dumps(summary, indent=2))
