"""Reproducible, honest small-grid benchmark and visual reports."""

from pathlib import Path
import json
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .operators import Grid, observe
from .fields import simulate, sensor_layout
from .models import (
    idw,
    tikhonov,
    gaussian_process,
    total_variation,
    heat_reconstruction,
)


def metrics(truth, pred, A, y):
    err = np.asarray(pred).ravel() - np.asarray(truth).ravel()
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "measurement_rmse": float(
            np.sqrt(np.mean((A @ np.asarray(pred).ravel() - y) ** 2))
        ),
    }


def run(output="results/demo", size=20, seed=42, ml_steps=0, real=False):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    grid = Grid((size, size))
    truth = simulate(grid, seed, kind="front")
    if real:
        from .data import camera_field

        truth = camera_field(grid.shape)
    A, layout = sensor_layout(grid, seed=seed + 1)
    noise = 0.03
    y = observe(A, truth, noise, seed + 2)
    n = len(layout["points"])
    jobs = {
        "IDW (points only)": lambda: idw(grid, layout["points"], y[:n]),
        "Tikhonov": lambda: tikhonov(grid, A, y, noise),
        "Gaussian process": lambda: gaussian_process(grid, A, y, noise),
        "Total variation": lambda: total_variation(grid, A, y, noise),
        "Heat regularization": lambda: heat_reconstruction(grid, A, y, noise),
    }
    results = {}
    scores = {}
    training = {}
    for name, fn in jobs.items():
        start = time.perf_counter()
        result = fn()
        results[name] = result
        scores[name] = {
            **metrics(truth, result.mean, A, y),
            "seconds": time.perf_counter() - start,
            "info": result.info,
        }
    if ml_steps:
        from .neural import train_cnn, predict_cnn, ConditionalDiffusion

        net, loss = train_cnn(grid, A, steps=ml_steps)
        results["CNN"] = predict_cnn(net, grid, A, y)
        training["cnn_loss"] = loss
        diffusion = ConditionalDiffusion()
        training["diffusion_loss"] = diffusion.train(grid, A, steps=ml_steps * 2)
        results["Conditional DDPM"], draws = diffusion.reconstruct(grid, A, y)
        np.save(root / "diffusion_draws.npy", draws)
        import torch

        torch.save(
            {
                "cnn": net.state_dict(),
                "diffusion": diffusion.net.state_dict(),
                "grid_shape": grid.shape,
                "beta": diffusion.beta,
                "A": A,
                "training_seeds": [1000, 2000],
            },
            root / "models.pt",
        )
        for name in ["CNN", "Conditional DDPM"]:
            scores[name] = {
                **metrics(truth, results[name].mean, A, y),
                "info": results[name].info,
            }
    panels = [("Truth", truth), ("Sensor coverage", A.sum(0).reshape(grid.shape))] + [
        (name, res.mean) for name, res in results.items()
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for ax, (name, f) in zip(axes.flat, panels):
        im = ax.imshow(
            f,
            origin="lower",
            extent=(0, 1, 0, 1),
            vmin=0,
            vmax=1 if name != "Sensor coverage" else None,
            cmap="viridis",
        )
        ax.set_title(name)
        fig.colorbar(im, ax=ax, shrink=0.7)
    for ax in list(axes.flat)[len(panels) :]:
        ax.axis("off")
    fig.savefig(root / "reconstructions.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(truth, origin="lower", extent=(0, 1, 0, 1))
    # imshow horizontal coordinate is array axis 1.
    ax.scatter(
        layout["points"][:, 1], layout["points"][:, 0], c="white", s=14, label="points"
    )
    for e in layout["lines"]:
        ax.plot(e[:, 1], e[:, 0], color="orange", alpha=0.6)
    for c in layout["footprints"]:
        ax.add_patch(plt.Circle((c[1], c[0]), 0.1, fill=False, color="cyan"))
    ax.set_title("Point, line-average, and footprint sensors")
    ax.legend()
    fig.savefig(root / "sensors.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key in zip(axes, ["Gaussian process", "Conditional DDPM"]):
        if key in results:
            im = ax.imshow(results[key].std, origin="lower")
            fig.colorbar(im, ax=ax)
            ax.set_title(key + " uncertainty")
        else:
            ax.axis("off")
    fig.savefig(root / "uncertainty.png", dpi=150)
    plt.close(fig)
    if training:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
        for ax, (key, loss) in zip(axes, training.items()):
            ax.plot(loss)
            ax.set_title(key)
            ax.set_xlabel("Training step")
        fig.savefig(root / "training.png", dpi=150)
        plt.close(fig)
    np.savez_compressed(
        root / "experiment.npz",
        truth=truth,
        A=A,
        y=y,
        **{name.replace(" ", "_"): r.mean for name, r in results.items()},
    )
    report = {
        "seed": seed,
        "grid": grid.shape,
        "noise_std": noise,
        "source": "Real photograph with simulated sensor observations"
        if real
        else "Simulated front field",
        "ml_training": "Synthetic fields; fixed sensor geometry; test seed separate. Real image is out of distribution.",
        "fairness": "IDW sees points only; other estimators see all sensors. Scores are a demo, not a tuned ranking.",
        "scores": scores,
        "training": training,
    }
    (root / "metrics.json").write_text(json.dumps(report, indent=2))
    html = (
        '<!doctype html><meta charset="utf-8"><title>Field Reconstruction Lab</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:20px;background:#f5f6f8;color:#172133}img{max-width:100%}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #bbb}</style><h1>Field Reconstruction Lab</h1><p>'
        + report["source"]
        + "</p><p>"
        + report["fairness"]
        + '</p><img src="sensors.png"><img src="reconstructions.png"><h2>Measured results</h2><table><tr><th>Model</th><th>Field RMSE</th><th>Measurement RMSE</th></tr>'
    )
    for name, m in scores.items():
        html += f"<tr><td>{name}</td><td>{m['rmse']:.4f}</td><td>{m['measurement_rmse']:.4f}</td></tr>"
    html += '</table><h2>Uncertainty</h2><p>GP: conditional standard deviation under fixed Gaussian assumptions. DDPM: uncalibrated sample spread.</p><img src="uncertainty.png">'
    if training:
        html += '<h2>Training</h2><img src="training.png">'
    (root / "index.html").write_text(html)
    return report
