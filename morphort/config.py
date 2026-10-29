"""Configuration commune de la segmentation BraTS."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ThresholdMethod = Literal["otsu", "percentile_90"]


@dataclass(frozen=True)
class SegmentationParameters:
    """Paramètres du pipeline morphologique en millimètres."""

    opening_radius_mm: float
    closing_radius_mm: float
    threshold_method: ThresholdMethod


DEFAULT_SEGMENTATION = SegmentationParameters(
    opening_radius_mm=1.0,
    closing_radius_mm=1.0,
    threshold_method="percentile_90",
)

OPENING_RADII_MM = (0.5, 1.0)
CLOSING_RADII_MM = (0.5, 1.0)
THRESHOLD_METHODS: tuple[ThresholdMethod, ...] = ("otsu", "percentile_90")

BRATS_RAW_DIRECTORY = Path.home() / "Downloads" / "brats_raw"
BRATS_EXPECTED_CASE_DIRECTORIES = 369
BRATS_EXPECTED_LABELED_CASES = 368
BRATS_SPLIT_SEED = 2026
BRATS_TUNING_CASES = 100
BRATS_BOOTSTRAP_REPLICATES = 10_000
