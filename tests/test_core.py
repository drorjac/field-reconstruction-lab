import numpy as np
import pytest
from fieldlab import Grid, points, lines, footprints, projections, combine
from fieldlab.models import (
    tikhonov,
    gaussian_process,
    total_variation,
    heat_reconstruction,
)
from fieldlab.fields import simulate
from fieldlab.operators import observe


@pytest.mark.parametrize("shape", [(9,), (7, 8), (4, 5, 6)])
def test_interpolation_linear_field_and_adjoint(shape):
    g = Grid(shape)
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, (8, len(shape)))
    A = points(g, p)
    f = g.coordinates.sum(1)
    assert np.allclose(A @ f, p.sum(1))
    x = rng.normal(size=g.size)
    y = rng.normal(size=8)
    assert np.allclose((A @ x) @ y, x @ (A.T @ y))
    assert np.allclose(A.sum(1), 1)


def test_lines_affine_integral_reversal():
    g = Grid((9, 10))
    e = np.array([[[0.1, 0.2], [0.9, 0.8]]])
    f = g.coordinates.sum(1)
    assert np.allclose(lines(g, e) @ f, 1)
    assert np.allclose(lines(g, e), lines(g, e[:, ::-1]))
    assert np.allclose(
        lines(g, e, integral=True) @ f, np.linalg.norm(e[0, 1] - e[0, 0])
    )


def test_footprints_projection_and_nullspace():
    g = Grid((4, 5, 6))
    F = footprints(g, [[0.1, 0.2, 0.3]], 0.01)
    assert np.allclose(F.sum(1), 1)
    volume = simulate(g, 8)
    P = projections(g, 0)
    assert np.allclose(P @ volume.ravel(), volume.sum(0).ravel())
    assert np.allclose(P @ volume.ravel(), P @ volume[::-1].ravel())


def test_gp_matches_scalar_gaussian_posterior_and_variance_shrinks():
    g = Grid((5,))
    A = points(g, [[0.5]])
    y = np.array([2.0])
    noise = 0.2
    r = gaussian_process(g, A, y, noise)
    assert np.isclose(r.mean[2], 2 / (1 + noise**2))
    assert np.isclose(r.std[2] ** 2, noise**2 / (1 + noise**2))
    more = gaussian_process(
        g, combine(A, points(g, [[0.1]])), np.array([2.0, 0.0]), noise
    )
    assert np.all(more.std <= r.std + 1e-9)


@pytest.mark.parametrize("solver", [tikhonov, total_variation, heat_reconstruction])
def test_reconstruction_improves_on_zero(solver):
    g = Grid((10, 10))
    truth = simulate(g, 3)
    A = points(g, np.random.default_rng(0).uniform(0, 1, (40, 2)))
    y = observe(A, truth, 0.01, 2)
    r = solver(g, A, y, noise=0.01)
    assert np.isfinite(r.mean).all()
    assert np.mean((r.mean - truth) ** 2) < np.mean(truth**2)


def test_weighting_downweights_outlier():
    g = Grid((6,))
    A = points(g, [[0.5], [0.5]])
    y = np.array([1.0, 10.0])
    r = gaussian_process(g, A, y, noise=np.array([0.01, 100.0]))
    reference = gaussian_process(g, A[:1], y[:1], noise=0.01)
    assert np.allclose(r.mean, reference.mean, atol=0.001)


def test_invalid_locations_noise_and_dimensions():
    g = Grid((6, 6))
    with pytest.raises(ValueError):
        points(g, [[2.0, 0.0]])
    with pytest.raises(ValueError):
        points(g, [[np.nan, 0.0]])
    with pytest.raises(ValueError):
        lines(g, [[[0, 0], [1, 1]]], samples=1)
    with pytest.raises(ValueError):
        gaussian_process(g, points(g, [[0, 0]]), np.ones(1), noise=0)
    with pytest.raises(ValueError):
        Grid((1,))


def test_packed_physical_data_units_and_missing_values():
    from fieldlab.data import unpack_netcdf

    decoded = unpack_netcdf([100, 200, 32767], 32767, 0.01)
    assert np.allclose(decoded[:2], [1.0, 2.0])
    assert np.isnan(decoded[2])


def test_physical_heat_mode_decay_and_equilibrium():
    from fieldlab.fields import heat_field

    g = Grid((8, 8))
    assert np.ptp(heat_field(g, 5)) < np.ptp(heat_field(g, 0))
    assert np.allclose(heat_field(g, time=1000), 1)
