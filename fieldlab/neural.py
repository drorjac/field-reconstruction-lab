"""CPU-sized supervised CNN and conditional DDPM, trained on distinct fields.

The condition is normalized backprojection plus coverage; it is lossy for
arbitrary geometries. Data-consistency correction uses the original A.
Conditional diffusion ensembles are NOT calibrated GP posterior intervals.
"""

import numpy as np
import torch
from torch import nn
from .models import Reconstruction
from .fields import simulate
from .operators import observe


def condition(A, y, shape):
    coverage = np.sum(A * A, axis=0)
    back = A.T @ y / np.maximum(A.sum(0), 1e-6)
    return np.stack(
        [back.reshape(shape), (coverage / (coverage.max() + 1e-8)).reshape(shape)]
    ).astype("float32")


class ConvNet(nn.Module):
    def __init__(self, channels, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


def training_data(grid, A, count=128, noise=0.03, seed=1000):
    if len(grid.shape) != 2:
        raise ValueError("CNN and DDPM require a 2D grid")
    fields = np.stack(
        [
            simulate(grid, seed + i, kind=("smooth", "front", "wave")[i % 3])
            for i in range(count)
        ]
    ).astype("float32")
    conditions = np.stack(
        [
            condition(A, observe(A, f, noise, seed + i + 100000), grid.shape)
            for i, f in enumerate(fields)
        ]
    )
    return torch.from_numpy(fields[:, None]), torch.from_numpy(conditions)


def train_cnn(grid, A, steps=200, count=128, noise=0.03, seed=1000):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    target, c = training_data(grid, A, count, noise, seed)
    net = ConvNet(2)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.002)
    losses = []
    for _ in range(steps):
        idx = torch.randint(count, (16,))
        loss = (net(c[idx]) - target[idx]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    net.eval()
    return net, losses


def predict_cnn(net, grid, A, y):
    with torch.no_grad():
        pred = net(torch.from_numpy(condition(A, y, grid.shape))[None])[0, 0].numpy()
    return Reconstruction(pred)


class ConditionalDiffusion:
    def __init__(self, timesteps=40, seed=2000):
        if timesteps < 2:
            raise ValueError("At least two timesteps required")
        torch.manual_seed(seed)
        self.net = ConvNet(4)
        self.beta = torch.linspace(0.0005, 0.12, timesteps)
        self.alpha = 1 - self.beta
        self.bar = torch.cumprod(self.alpha, 0)
        self.timesteps = timesteps

    def train(self, grid, A, steps=400, count=128, noise=0.03, seed=2000):
        torch.manual_seed(seed)
        torch.set_num_threads(2)
        x, c = training_data(grid, A, count, noise, seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=0.001)
        losses = []
        self.net.train()
        for _ in range(steps):
            idx = torch.randint(count, (16,))
            t = torch.randint(self.timesteps, (16,))
            eps = torch.randn_like(x[idx])
            bar = self.bar[t, None, None, None]
            noisy = bar.sqrt() * x[idx] + (1 - bar).sqrt() * eps
            time = (t.float() / (self.timesteps - 1))[:, None, None, None].expand(
                -1, 1, *grid.shape
            )
            estimate = self.net(torch.cat([noisy, c[idx], time], 1))
            loss = (estimate - eps).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        self.net.eval()
        return losses

    def reconstruct(self, grid, A, y, samples=8, seed=3000, consistency=0.05):
        if samples < 2 or not 0 <= consistency <= 1:
            raise ValueError("Use >=2 samples and consistency in [0,1]")
        torch.manual_seed(seed)
        c = torch.from_numpy(condition(A, y, grid.shape))[None].repeat(samples, 1, 1, 1)
        x = torch.randn(samples, 1, *grid.shape)
        # Exact noise-free affine projection blended with generated x0 estimates.
        # For noisy observations this is an explicitly heuristic guidance, not a posterior sampler.
        correction = A.T @ np.linalg.solve(
            A @ A.T + 1e-4 * np.eye(len(y)), np.eye(len(y))
        )
        A_t = torch.tensor(A, dtype=torch.float32)
        P = torch.tensor(correction, dtype=torch.float32)
        yy = torch.tensor(y, dtype=torch.float32)
        with torch.no_grad():
            for t in reversed(range(self.timesteps)):
                time = torch.full((samples, 1, *grid.shape), t / (self.timesteps - 1))
                eps = self.net(torch.cat([x, c, time], 1))
                x0 = (x - (1 - self.bar[t]).sqrt() * eps) / self.bar[t].sqrt()
                flat = x0.flatten(1)
                flat += consistency * ((yy - flat @ A_t.T) @ P.T)
                x0 = flat.reshape_as(x0)
                if t == 0:
                    x = x0
                    break
                posterior_var = self.beta[t] * (1 - self.bar[t - 1]) / (1 - self.bar[t])
                coef0 = self.beta[t] * self.bar[t - 1].sqrt() / (1 - self.bar[t])
                coeft = self.alpha[t].sqrt() * (1 - self.bar[t - 1]) / (1 - self.bar[t])
                x = coef0 * x0 + coeft * x + posterior_var.sqrt() * torch.randn_like(x)
        draws = x[:, 0].numpy()
        return Reconstruction(
            draws.mean(0),
            draws.std(0, ddof=1),
            {
                "uncertainty": "ensemble spread; not calibrated posterior",
                "samples": samples,
            },
        ), draws
