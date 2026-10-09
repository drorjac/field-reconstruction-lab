"""Five independent estimators; all except IDW honor the full measurement A."""

from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy import sparse


@dataclass
class Reconstruction:
    mean: np.ndarray
    std: np.ndarray | None = None
    info: dict = field(default_factory=dict)


def _inputs(A, y, noise):
    A = np.asarray(A, float)
    y = np.asarray(y, float)
    if (
        A.ndim != 2
        or y.shape != (A.shape[0],)
        or not np.isfinite(A).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Finite A (m,n) and y (m,) required")
    s = np.broadcast_to(np.asarray(noise, float), y.shape)
    if np.any(s <= 0) or not np.isfinite(s).all():
        raise ValueError("Noise standard deviations must be positive")
    return A / s[:, None], y / s, s


def gradient_matrix(grid):
    rows = []
    cols = []
    vals = []
    k = 0
    for idx in np.ndindex(*grid.shape):
        for axis, n in enumerate(grid.shape):
            if idx[axis] + 1 < n:
                nxt = list(idx)
                nxt[axis] += 1
                rows.extend([k, k])
                cols.extend(
                    [
                        np.ravel_multi_index(idx, grid.shape),
                        np.ravel_multi_index(tuple(nxt), grid.shape),
                    ]
                )
                vals.extend([-(n - 1), n - 1])
                k += 1
    return sparse.csr_matrix((vals, (rows, cols)), shape=(k, grid.size))


def idw(grid, locations, values, power=2):
    """Point-only inverse-distance weighting. Do not pass line midpoints."""
    from .operators import points

    points(grid, locations)
    v = np.asarray(values, float)
    if v.shape != (len(locations),) or len(v) == 0 or power <= 0:
        raise ValueError("Nonempty point values and positive power required")
    d = cdist(grid.coordinates, np.asarray(locations))
    w = np.maximum(d, 1e-12) ** (-power)
    return Reconstruction(((w @ v) / w.sum(1)).reshape(grid.shape))


def tikhonov(grid, A, y, noise=0.03, alpha=1):
    """Weighted least squares with first-gradient roughness regularization."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    B, b, _ = _inputs(A, y, noise)
    D = gradient_matrix(grid)
    H = B.T @ B + alpha * (D.T @ D).toarray() + 1e-8 * np.eye(grid.size)
    mean = cho_solve(cho_factor(H), B.T @ b)
    return Reconstruction(mean.reshape(grid.shape))


def gaussian_process(grid, A, y, noise=0.03, length_scale=0.2, variance=1):
    """Zero-mean RBF GP conditioned on arbitrary linear measurements."""
    _, _, s = _inputs(A, y, noise)
    if length_scale <= 0 or variance <= 0:
        raise ValueError("Positive kernel parameters required")
    K = variance * np.exp(
        -cdist(grid.coordinates, grid.coordinates, "sqeuclidean")
        / (2 * length_scale**2)
    )
    KA = K @ A.T
    C = A @ KA + np.diag(s * s) + 1e-9 * np.eye(len(y))
    factor = cho_factor(C)
    mean = KA @ cho_solve(factor, y)
    var = np.maximum(variance - np.sum(KA * cho_solve(factor, KA.T).T, axis=1), 0)
    return Reconstruction(
        mean.reshape(grid.shape),
        np.sqrt(var).reshape(grid.shape),
        {"uncertainty": "conditional latent-field standard deviation; fixed kernel"},
    )


def total_variation(grid, A, y, noise=0.03, alpha=0.5, maxiter=1500):
    """Smoothed anisotropic TV via L-BFGS; preserves jumps better than L2."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    B, b, _ = _inputs(A, y, noise)
    D = gradient_matrix(grid)
    eps = 1e-3

    def objective(x):
        r = B @ x - b
        g = D @ x
        smooth = np.sqrt(g * g + eps * eps)
        return 0.5 * r @ r + alpha * smooth.sum(), B.T @ r + alpha * (
            D.T @ (g / smooth)
        )

    opt = minimize(
        objective,
        np.zeros(grid.size),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-6, "gtol": 1e-4},
    )
    return Reconstruction(
        opt.x.reshape(grid.shape),
        info={
            "converged": bool(opt.success),
            "iterations": opt.nit,
            "message": str(opt.message),
        },
    )


def heat_reconstruction(grid, A, y, noise=0.03, steps=400, diffusivity=0.002):
    """Stable Landweber descent + Laplacian (heat) regularization.

    Solves a deterministic quadratic objective, not generative diffusion.
    """
    if steps < 1 or diffusivity < 0:
        raise ValueError("Invalid heat parameters")
    B, b, _ = _inputs(A, y, noise)
    D = gradient_matrix(grid)
    H = B.T @ B + diffusivity * (D.T @ D).toarray()
    dt = 0.95 / max(np.linalg.eigvalsh(H)[-1], 1e-12)
    x = np.zeros(grid.size)
    rhs = B.T @ b
    for _ in range(steps):
        x += dt * (rhs - H @ x)
    return Reconstruction(x.reshape(grid.shape), info={"step_size": float(dt)})
