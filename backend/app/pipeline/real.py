import numpy as np
from numpy.typing import NDArray

from app.pipeline.base import (
    GrowthResult,
    MaskArray,
    MeasurementResult,
    ModelName,
    PipelineBackend,
    RadiotherapyResult,
)


class RealBackend(PipelineBackend):
    def segment(
        self,
        volume_flair: NDArray[np.float32],
        volume_t1ce: NDArray[np.float32] | None = None,
    ) -> tuple[MaskArray, dict[str, float | int | str]]:
        # TODO: Add FLAIR z-score thresholding, opening, closing, hole filling,
        # and size-times-brightness connected-component selection.
        raise NotImplementedError("Real segmentation has not been implemented.")

    def measure(
        self, mask: MaskArray, voxel_size: tuple[float, float, float]
    ) -> MeasurementResult:
        # TODO: Implement validated physical measurements for real segmentations.
        raise NotImplementedError("Real measurements have not been implemented.")

    def fit_growth(
        self,
        volumes_by_day: list[dict[str, float]],
        model: ModelName,
    ) -> GrowthResult:
        # TODO: Implement Gompertz/logistic fitting with parameter uncertainty.
        raise NotImplementedError("Real growth fitting has not been implemented.")

    def simulate_rt(
        self,
        growth_params: dict[str, float],
        alpha: float,
        beta: float,
        bed_max: float,
        n_fractions: int,
    ) -> RadiotherapyResult:
        # TODO: Implement the linear-quadratic model with SLSQP optimization.
        raise NotImplementedError(
            "Real radiotherapy simulation has not been implemented."
        )
