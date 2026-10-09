"""Simulated fields are normalized examples, not calibrated rain simulations."""

import numpy as np
from scipy.ndimage import gaussian_filter


def simulate(grid, seed=0, kind="smooth"):
    rng = np.random.default_rng(seed)
    x = grid.coordinates
    f = np.zeros(grid.size)
    for _ in range(5):
        center = rng.uniform(0.1, 0.9, len(grid.shape))
        width = rng.uniform(0.06, 0.22)
        f += rng.uniform(0.2, 1) * np.exp(
            -np.sum((x - center) ** 2, axis=1) / (2 * width**2)
        )
    if kind == "front":
        f += 0.5 * (x[:, 0] > 0.55)
    elif kind == "wave":
        f += 0.3 * np.sin(6 * np.pi * x[:, 0])
    elif kind != "smooth":
        raise ValueError("kind must be smooth, front, or wave")
    f = f.reshape(grid.shape)
    return (f - f.min()) / (np.ptp(f) + 1e-12)


def heat_step(field, sigma=1):
    """Gaussian heat smoothing with reflecting boundary conditions."""
    return gaussian_filter(field, sigma=sigma, mode="reflect")


def sensor_layout(grid, n_points=35, n_lines=20, n_footprints=10, seed=0):
    from .operators import points, lines, footprints, combine

    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, (n_points, len(grid.shape)))
    e = rng.uniform(0, 1, (n_lines, 2, len(grid.shape)))
    c = rng.uniform(0, 1, (n_footprints, len(grid.shape)))
    return combine(points(grid, p), lines(grid, e), footprints(grid, c)), {
        "points": p,
        "lines": e,
        "footprints": c,
    }


def heat_field(grid, time=0, diffusivity=0.02):
    """Analytic Neumann-boundary heat-equation solution on the unit box.

    Temperature = 1 + sum 0.2*cos(2π r_axis)*exp(-κ(2π)² t).
    Unlike Gaussian-blob generators, this has explicit PDE and boundary meaning.
    """
    if time < 0 or diffusivity < 0:
        raise ValueError("Nonnegative time and diffusivity required")
    coordinates = grid.coordinates
    f = 1 + 0.2 * np.cos(2 * np.pi * coordinates).sum(1) * np.exp(
        -diffusivity * (2 * np.pi) ** 2 * time
    )
    return f.reshape(grid.shape)
