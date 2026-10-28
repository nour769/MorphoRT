from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from app.pipeline.base import (
    FloatArray,
    GrowthResult,
    MaskArray,
    MeasurementResult,
    ModelName,
    PipelineBackend,
    RadiotherapyResult,
    TrajectoryPoint,
)
from app.pipeline.metrics import linear_quadratic_survival


def _gompertz(
    day: FloatArray, r: float, carrying_capacity: float, initial: float
) -> FloatArray:
    return carrying_capacity * np.exp(
        np.log(initial / carrying_capacity) * np.exp(-r * day)
    )


def _logistic(
    day: FloatArray, r: float, carrying_capacity: float, initial: float
) -> FloatArray:
    return carrying_capacity / (
        1.0 + ((carrying_capacity - initial) / initial) * np.exp(-r * day)
    )


class MockBackend(PipelineBackend):
    def segment(
        self,
        volume_flair: NDArray[np.float32],
        volume_t1ce: NDArray[np.float32] | None = None,
    ) -> tuple[MaskArray, dict[str, float | int | str]]:
        if volume_flair.ndim != 3 or min(volume_flair.shape) < 2:
            raise ValueError(
                "FLAIR volume must be a 3D image with at least 2 voxels per axis."
            )
        shape = np.asarray(volume_flair.shape, dtype=float)
        center = (shape - 1.0) / 2.0
        radii = np.maximum(shape * np.array([0.19, 0.16, 0.14]), 1.0)
        axes = np.ogrid[tuple(slice(0, size) for size in volume_flair.shape)]
        distance_squared = sum(
            ((axis - coordinate) / radius) ** 2
            for axis, coordinate, radius in zip(axes, center, radii, strict=True)
        )
        mask = (
            gaussian_filter((distance_squared <= 1.0).astype(float), sigma=0.8) >= 0.35
        )
        return mask.astype(bool), {
            "source": "synthetic",
            "smoothing_sigma_voxels": 0.8,
            "threshold": 0.35,
        }

    def measure(
        self, mask: MaskArray, voxel_size: tuple[float, float, float]
    ) -> MeasurementResult:
        if mask.ndim != 3:
            raise ValueError("Segmentation mask must be 3D.")
        if len(voxel_size) != 3 or any(size <= 0 for size in voxel_size):
            raise ValueError(
                "Voxel dimensions must be three positive values in millimeters."
            )
        voxel_volume_cm3 = float(np.prod(voxel_size)) / 1000.0
        volume_cm3 = float(mask.sum()) * voxel_volume_cm3
        if not np.any(mask):
            return {"volume_cm3": 0.0, "surface_cm2": 0.0, "sphericity": 0.0}

        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        surface_cm2 = 0.0
        voxel_area_cm2 = (
            voxel_size[1] * voxel_size[2] / 100.0,
            voxel_size[0] * voxel_size[2] / 100.0,
            voxel_size[0] * voxel_size[1] / 100.0,
        )
        for axis, face_area in enumerate(voxel_area_cm2):
            exposed = np.diff(padded.astype(np.int8), axis=axis)
            surface_cm2 += float(np.count_nonzero(exposed)) * face_area
        sphericity = (
            float(
                np.pi ** (1.0 / 3.0) * (6.0 * volume_cm3) ** (2.0 / 3.0) / surface_cm2
            )
            if surface_cm2 > 0
            else 0.0
        )
        return {
            "volume_cm3": volume_cm3,
            "surface_cm2": surface_cm2,
            "sphericity": sphericity,
        }

    def fit_growth(
        self,
        volumes_by_day: list[dict[str, float]],
        model: ModelName,
    ) -> GrowthResult:
        if len(volumes_by_day) < 3:
            raise ValueError(
                "At least 3 time points are required to fit a growth model."
            )
        days = np.asarray([point["day"] for point in volumes_by_day], dtype=float)
        volumes = np.asarray(
            [point["volume_cm3"] for point in volumes_by_day], dtype=float
        )
        if np.any(volumes <= 0) or not np.all(np.isfinite(volumes)):
            raise ValueError("Volumes must be finite positive values.")
        if not np.all(np.isfinite(days)) or len(np.unique(days)) != len(days):
            raise ValueError("Days must be finite and unique.")
        order = np.argsort(days)
        days = days[order]
        volumes = volumes[order]
        days_from_first = days - days[0]
        model_function: Callable[[FloatArray, float, float, float], FloatArray]
        model_function = _gompertz if model == "gompertz" else _logistic
        initial_guess = (0.05, float(max(volumes) * 2.0), float(volumes[0]))
        parameters, covariance = curve_fit(
            lambda day, r, capacity: model_function(
                day, r, capacity, float(volumes[0])
            ),
            days_from_first,
            volumes,
            p0=initial_guess[:2],
            bounds=([1e-8, max(volumes) * 1.001], [10.0, max(volumes) * 1e6]),
            maxfev=20_000,
        )
        r, carrying_capacity = (float(value) for value in parameters)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        confidence_r = (
            max(0.0, r - 1.96 * float(standard_errors[0])),
            r + 1.96 * float(standard_errors[0]),
        )
        confidence_capacity = (
            max(0.0, carrying_capacity - 1.96 * float(standard_errors[1])),
            carrying_capacity + 1.96 * float(standard_errors[1]),
        )
        curve_days = np.linspace(days[0], days[-1], 100)
        fitted_volumes = model_function(
            curve_days - days[0], r, carrying_capacity, float(volumes[0])
        )
        epsilon_r = max(abs(r) * 1e-5, 1e-7)
        epsilon_capacity = max(abs(carrying_capacity) * 1e-5, 1e-7)
        derivative_r = (
            model_function(
                curve_days - days[0],
                r + epsilon_r,
                carrying_capacity,
                float(volumes[0]),
            )
            - model_function(
                curve_days - days[0],
                r - epsilon_r,
                carrying_capacity,
                float(volumes[0]),
            )
        ) / (2.0 * epsilon_r)
        derivative_capacity = (
            model_function(
                curve_days - days[0],
                r,
                carrying_capacity + epsilon_capacity,
                float(volumes[0]),
            )
            - model_function(
                curve_days - days[0],
                r,
                carrying_capacity - epsilon_capacity,
                float(volumes[0]),
            )
        ) / (2.0 * epsilon_capacity)
        jacobian = np.column_stack((derivative_r, derivative_capacity))
        variances = np.einsum("ij,jk,ik->i", jacobian, covariance, jacobian)
        spreads = 1.96 * np.sqrt(np.maximum(variances, 0.0))
        curve: list[dict[str, float]] = []
        for day, fitted_volume, spread in zip(
            curve_days, fitted_volumes, spreads, strict=True
        ):
            curve.append(
                {
                    "day": float(day),
                    "volume_cm3": float(fitted_volume),
                    "lower_cm3": max(0.0, float(fitted_volume - spread)),
                    "upper_cm3": float(fitted_volume + spread),
                }
            )
        return {
            "r": r,
            "K": carrying_capacity,
            "ci_r": confidence_r,
            "ci_K": confidence_capacity,
            "curve": curve,
        }

    def simulate_rt(
        self,
        growth_params: dict[str, float],
        alpha: float,
        beta: float,
        bed_max: float,
        n_fractions: int,
    ) -> RadiotherapyResult:
        if alpha <= 0 or beta <= 0 or bed_max <= 0 or n_fractions < 1:
            raise ValueError(
                "Alpha, beta, BED limit, and fraction count must be positive."
            )
        initial_volume = growth_params.get("volume_cm3", 0.0)
        r = growth_params.get("r", 0.0)
        carrying_capacity = growth_params.get("K", initial_volume)
        if (
            initial_volume <= 0
            or r < 0
            or carrying_capacity <= 0
            or initial_volume > carrying_capacity
        ):
            raise ValueError(
                "Growth parameters must contain valid volume, r, and K values."
            )
        alpha_beta = alpha / beta
        dose = (
            -alpha_beta
            + np.sqrt(alpha_beta**2 + 4.0 * alpha_beta * bed_max / n_fractions)
        ) / 2.0
        fixed: list[TrajectoryPoint] = []
        adaptive: list[TrajectoryPoint] = []
        adaptive_bed = 0.0
        fixed_volume = initial_volume
        adaptive_volume = initial_volume
        for fraction in range(n_fractions + 1):
            if fraction > 0:
                fixed_volume = self._advance_volume(fixed_volume, r, carrying_capacity)
                adaptive_volume = self._advance_volume(
                    adaptive_volume, r, carrying_capacity
                )
                adaptive_dose = dose * min(
                    1.0,
                    max(0.25, 0.8 * (adaptive_volume / initial_volume) ** 0.25),
                )
                fixed_volume *= linear_quadratic_survival(alpha, beta, dose)
                adaptive_volume *= linear_quadratic_survival(alpha, beta, adaptive_dose)
                adaptive_bed += adaptive_dose * (1.0 + adaptive_dose / alpha_beta)
            else:
                adaptive_dose = dose
            fixed.append(
                {
                    "fraction": fraction,
                    "volume_cm3": fixed_volume,
                    "dose_gy": dose if fraction else 0.0,
                    "bed_gy": float(fraction * dose * (1.0 + dose / alpha_beta)),
                }
            )
            adaptive.append(
                {
                    "fraction": fraction,
                    "volume_cm3": adaptive_volume,
                    "dose_gy": adaptive_dose if fraction else 0.0,
                    "bed_gy": adaptive_bed,
                }
            )
        return {"fixed": fixed, "adaptive": adaptive}

    @staticmethod
    def _advance_volume(volume: float, r: float, carrying_capacity: float) -> float:
        if r == 0:
            return volume
        return float(
            carrying_capacity * np.exp(np.log(volume / carrying_capacity) * np.exp(-r))
        )
