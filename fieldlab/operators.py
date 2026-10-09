"""Explicit linear operators: array axes and coordinates use the SAME order.

A row is a measurement's spatial support, not a pseudo-point at its midpoint.
Coordinates live in [0,1]^d. Dense matrices are intentional for teaching grids.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid:
    shape: tuple

    def __post_init__(self):
        if not 1 <= len(self.shape) <= 3 or any(n < 2 for n in self.shape):
            raise ValueError("Use 1–3 axes with at least two cells per axis")

    @property
    def coordinates(self):
        return np.stack(
            np.meshgrid(*(np.linspace(0, 1, n) for n in self.shape), indexing="ij"), -1
        ).reshape(-1, len(self.shape))

    @property
    def size(self):
        return int(np.prod(self.shape))


def _weights(grid, locations):
    p = np.asarray(locations, dtype=float)
    if (
        p.ndim != 2
        or p.shape[1] != len(grid.shape)
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("Locations must be finite (m,d) coordinates in [0,1]")
    scaled = p * (np.array(grid.shape) - 1)
    lo = np.minimum(np.floor(scaled).astype(int), np.array(grid.shape) - 2)
    frac = scaled - lo
    A = np.zeros((len(p), grid.size))
    for corner in np.ndindex(*(2,) * len(grid.shape)):
        idx = lo + corner
        w = np.prod(np.where(np.array(corner), frac, 1 - frac), axis=1)
        np.add.at(A, (np.arange(len(p)), np.ravel_multi_index(idx.T, grid.shape)), w)
    return A


def points(grid, locations):
    """Multilinear point samples, including endpoints."""
    return _weights(grid, locations)


def lines(grid, endpoints, samples=64, integral=False):
    """Trapezoidal line averages; integral=True returns length-weighted integrals.

    Length uses normalized-domain coordinates. Multiply by physical domain
    length for isotropic units; rescale endpoints for anisotropic physical units.
    """
    e = np.asarray(endpoints, float)
    if e.ndim != 3 or e.shape[1:] != (2, len(grid.shape)) or samples < 2:
        raise ValueError("Expected (m,2,d) endpoints and >=2 quadrature samples")
    t = np.linspace(0, 1, samples)
    q = np.ones(samples)
    q[[0, -1]] = 0.5
    q /= q.sum()
    rows = [q @ _weights(grid, a[None] + t[:, None] * (b - a)) for a, b in e]
    A = np.asarray(rows).reshape(len(e), grid.size)
    if integral:
        A *= np.linalg.norm(e[:, 1] - e[:, 0], axis=1)[:, None]
    return A


def footprints(grid, centers, radius=0.1):
    """Uniform footprint averages over grid nodes inside a radius (e.g. pixels)."""
    c = np.asarray(centers, float)
    _weights(grid, c)  # shared bounds validation
    if radius <= 0:
        raise ValueError("radius must be positive")
    distances = np.linalg.norm(grid.coordinates[None] - c[:, None], axis=-1)
    A = (distances <= radius).astype(float)
    empty = A.sum(1) == 0
    A[empty] = points(grid, c[empty])
    return A / A.sum(1, keepdims=True)


def projections(grid, axis=0):
    """Parallel discrete sums along an axis; multiply by spacing for integrals.

    For 3D this produces a 2D projection. One view cannot identify a volume.
    """
    if not 0 <= axis < len(grid.shape):
        raise ValueError("Invalid projection axis")
    remaining = tuple(n for i, n in enumerate(grid.shape) if i != axis)
    A = np.zeros((int(np.prod(remaining)), grid.size))
    for idx in np.ndindex(*grid.shape):
        out = tuple(v for i, v in enumerate(idx) if i != axis)
        row = np.ravel_multi_index(out, remaining) if remaining else 0
        A[row, np.ravel_multi_index(idx, grid.shape)] = 1
    return A


def combine(*operators):
    if not operators or len({a.shape[1] for a in operators}) != 1:
        raise ValueError("Operators must share a grid")
    return np.vstack(operators)


def observe(A, field, noise=0.03, seed=0):
    std = np.broadcast_to(np.asarray(noise, float), (A.shape[0],))
    if np.any(std < 0):
        raise ValueError("Noise standard deviation cannot be negative")
    return (
        A @ np.asarray(field).ravel()
        + np.random.default_rng(seed).normal(size=len(std)) * std
    )
