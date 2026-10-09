import numpy as np
import pytest

torch = pytest.importorskip("torch")
from fieldlab import Grid
from fieldlab.fields import simulate, sensor_layout
from fieldlab.operators import observe
from fieldlab.neural import train_cnn, predict_cnn, ConditionalDiffusion


def test_cnn_training_learns_and_predicts_independent_field():
    g = Grid((8, 8))
    A, _ = sensor_layout(g, 10, 5, 2, seed=2)
    net, loss = train_cnn(g, A, steps=35, count=40)
    assert np.mean(loss[-5:]) < np.mean(loss[:5])
    f = simulate(g, 9999)
    r = predict_cnn(net, g, A, observe(A, f, 0.03, 9998))
    assert r.mean.shape == g.shape and np.isfinite(r.mean).all()


def test_diffusion_training_sampling_and_seed_reproducibility():
    g = Grid((8, 8))
    A, _ = sensor_layout(g, 10, 5, 2, seed=3)
    ddpm = ConditionalDiffusion(timesteps=8)
    loss = ddpm.train(g, A, steps=35, count=40)
    assert np.mean(loss[-5:]) < np.mean(loss[:5])
    f = simulate(g, 9997)
    y = observe(A, f, 0.03, 9996)
    r, draws = ddpm.reconstruct(g, A, y, samples=3, seed=4)
    _, again = ddpm.reconstruct(g, A, y, samples=3, seed=4)
    assert np.allclose(
        draws, again, rtol=1e-5, atol=1e-6
    )  # floating-point BLAS tolerance
    assert r.mean.shape == g.shape and np.isfinite(draws).all()
    assert np.any(r.std > 0)
