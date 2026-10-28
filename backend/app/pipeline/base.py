from abc import ABC, abstractmethod
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

ModelName = Literal["gompertz", "logistic"]
FloatArray = NDArray[np.float64]
MaskArray = NDArray[np.bool_]


class MeasurementResult(TypedDict):
    volume_cm3: float
    surface_cm2: float
    sphericity: float


class GrowthResult(TypedDict):
    r: float
    K: float
    ci_r: tuple[float, float]
    ci_K: tuple[float, float]
    curve: list[dict[str, float]]


class TrajectoryPoint(TypedDict):
    fraction: int
    volume_cm3: float
    dose_gy: float
    bed_gy: float


class RadiotherapyResult(TypedDict):
    fixed: list[TrajectoryPoint]
    adaptive: list[TrajectoryPoint]


class PipelineBackend(ABC):
    @abstractmethod
    def segment(
        self,
        volume_flair: NDArray[np.float32],
        volume_t1ce: NDArray[np.float32] | None = None,
    ) -> tuple[MaskArray, dict[str, float | int | str]]:
        """Return a binary tumor mask and parameters used to produce it."""

    @abstractmethod
    def measure(
        self, mask: MaskArray, voxel_size: tuple[float, float, float]
    ) -> MeasurementResult:
        """Calculate physical volume, surface area, and sphericity."""

    @abstractmethod
    def fit_growth(
        self,
        volumes_by_day: list[dict[str, float]],
        model: ModelName,
    ) -> GrowthResult:
        """Fit a growth model to measured volumes and return its uncertainty."""

    @abstractmethod
    def simulate_rt(
        self,
        growth_params: dict[str, float],
        alpha: float,
        beta: float,
        bed_max: float,
        n_fractions: int,
    ) -> RadiotherapyResult:
        """Compare fixed and adaptive fractionation trajectories."""
