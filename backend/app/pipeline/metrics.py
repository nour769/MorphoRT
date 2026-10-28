import numpy as np
from numpy.typing import NDArray


def dice_coefficient(
    predicted: NDArray[np.bool_], reference: NDArray[np.bool_]
) -> float:
    if predicted.shape != reference.shape:
        raise ValueError("Predicted and reference masks must have the same shape.")
    predicted = np.asarray(predicted, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    denominator = int(predicted.sum()) + int(reference.sum())
    if denominator == 0:
        return 1.0
    intersection = int(np.logical_and(predicted, reference).sum())
    return 2.0 * intersection / denominator


def linear_quadratic_survival(alpha: float, beta: float, dose_gy: float) -> float:
    if alpha < 0 or beta < 0 or dose_gy < 0:
        raise ValueError("Alpha, beta, and dose must be non-negative.")
    return float(np.exp(-(alpha * dose_gy + beta * dose_gy**2)))
