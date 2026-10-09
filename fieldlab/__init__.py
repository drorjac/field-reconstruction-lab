"""Small, inspectable inverse-problem implementations for teaching."""

from .operators import Grid, points, lines, footprints, projections, combine
from .models import (
    idw,
    tikhonov,
    gaussian_process,
    total_variation,
    heat_reconstruction,
)

__all__ = [
    "Grid",
    "points",
    "lines",
    "footprints",
    "projections",
    "combine",
    "idw",
    "tikhonov",
    "gaussian_process",
    "total_variation",
    "heat_reconstruction",
]
