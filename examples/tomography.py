"""Classical 2D CT and a 3D physical-density projection example."""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize
from fieldlab import Grid, projections
from fieldlab.fields import simulate


def main():
    root = Path("results/tomography")
    root.mkdir(parents=True, exist_ok=True)
    phantom = resize(shepp_logan_phantom(), (64, 64), anti_aliasing=True)
    angles = np.linspace(0, 180, 45, endpoint=False)
    sino = radon(phantom, theta=angles, circle=True)
    reconstruction = iradon(sino, theta=angles, filter_name="ramp", circle=True)
    grid = Grid((12, 12, 12))
    volume = simulate(grid, seed=9)
    projected = (projections(grid, axis=0) @ volume.ravel()).reshape(12, 12) / (
        grid.shape[0] - 1
    )
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, name, x in zip(
        axes,
        [
            "Synthetic CT phantom",
            "Radon sinogram",
            "Filtered backprojection",
            "3D density → 2D projection",
        ],
        [phantom, sino, reconstruction, projected],
    ):
        ax.imshow(x, cmap="magma")
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(root / "projections.png", dpi=150)
    np.savez_compressed(
        root / "projection_data.npz",
        phantom=phantom,
        sinogram=sino,
        reconstruction=reconstruction,
        volume=volume,
        projection=projected,
    )
    print("FBP RMSE:", np.sqrt(np.mean((phantom - reconstruction) ** 2)))


if __name__ == "__main__":
    main()
