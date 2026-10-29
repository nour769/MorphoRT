"""Segmentation pédagogique par seuillage et morphologie mathématique."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from morphort.config import DEFAULT_SEGMENTATION, ThresholdMethod
from morphort.morphology import (
    BinaryImage,
    FloatImage,
    closing,
    fill_holes,
    largest_component,
    opening,
    opening_by_reconstruction,
    reconstruct,
    structuring_element,
)


@dataclass(frozen=True)
class SegmentationStep:
    """Image produite à une étape et explication mathématique associée."""

    name: str
    formula: str
    explanation: str
    image: BinaryImage | FloatImage


class EmptySegmentationError(ValueError):
    """Aucun voxel n'a subsisté au nettoyage morphologique."""


def otsu_threshold(values: FloatImage, bins: int = 256) -> float:
    """Calcule le seuil d'Otsu en maximisant la variance entre classes.

    Formule : sigma²_inter(t) = omega0(t) * omega1(t) * (mu0(t)-mu1(t))².
    L'histogramme remplace un tri coûteux et garde un seuil reproductible.
    """
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("Otsu nécessite au moins une intensité finie.")
    if bins < 2:
        raise ValueError("L'histogramme Otsu doit avoir au moins deux classes.")

    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    if minimum == maximum:
        return minimum

    histogram, edges = np.histogram(finite_values, bins=bins, range=(minimum, maximum))
    probabilities = histogram.astype(np.float64) / finite_values.size
    centers = (edges[:-1] + edges[1:]) / 2
    weight_left = np.cumsum(probabilities)
    mean_left = np.cumsum(probabilities * centers)
    total_mean = float(mean_left[-1])
    denominator = weight_left * (1.0 - weight_left)
    between_variance = np.zeros_like(denominator)
    valid = denominator > 0
    between_variance[valid] = (
        total_mean * weight_left[valid] - mean_left[valid]
    ) ** 2 / denominator[valid]
    threshold_index = int(np.argmax(between_variance))
    return float(edges[threshold_index + 1])


def robust_normalize(volume: FloatImage) -> tuple[FloatImage, BinaryImage]:
    """Normalise l'IRM par le z-score robuste dans le masque anatomique.

    Formule : z(x) = (I(x) - médiane(I_cerveau)) / MAD(I_cerveau).
    La médiane et l'écart absolu médian réduisent l'effet des valeurs extrêmes.
    """
    image = np.asarray(volume, dtype=np.float64)
    if image.ndim != 3 or image.size == 0:
        raise ValueError("La normalisation nécessite un volume IRM 3D non vide.")
    finite = np.isfinite(image)
    positive = image[finite & (image > 0)]
    if positive.size < 8:
        raise ValueError("Le volume ne contient pas assez d'intensités cérébrales.")

    lower_bound = float(np.percentile(positive, 10))
    brain = finite & (image > lower_bound)
    brain_values = image[brain]
    median = float(np.median(brain_values))
    mad = float(np.median(np.abs(brain_values - median)))
    scale = 1.4826 * mad
    if scale <= np.finfo(np.float64).eps:
        scale = float(np.std(brain_values))
    if scale <= np.finfo(np.float64).eps:
        raise ValueError(
            "L'intensité cérébrale est constante, impossible à normaliser."
        )

    normalized = np.zeros(image.shape, dtype=np.float64)
    normalized[brain] = (brain_values - median) / scale
    return normalized, brain


def segment_tumor(
    flair: FloatImage,
    spacing_mm: tuple[float, float, float],
    *,
    opening_radius_mm: float = DEFAULT_SEGMENTATION.opening_radius_mm,
    closing_radius_mm: float | None = DEFAULT_SEGMENTATION.closing_radius_mm,
    threshold_method: ThresholdMethod = DEFAULT_SEGMENTATION.threshold_method,
    threshold: float | None = None,
    opening_method: Literal["standard", "reconstruction"] = "standard",
    include_diagnostic_steps: bool = True,
) -> tuple[BinaryImage, list[SegmentationStep], float]:
    """Exécute le pipeline de segmentation morphologique en 3D.

    Les étapes sont conservées pour expliquer chaque transformation à l'écran.
    Les seuils automatiques s'appliquent aux intensités normalisées dans le
    cerveau. Les rayons d'ouverture et de fermeture sont indépendants et en mm.
    Sans étapes diagnostiques, les reconstructions après la plus grande
    composante sont omises car elles restituent exactement cette composante.
    """
    if not np.isfinite(opening_radius_mm) or opening_radius_mm <= 0:
        raise ValueError("Le rayon morphologique doit être positif en mm.")
    if closing_radius_mm is None:
        closing_radius_mm = 1.5 * opening_radius_mm
    if not np.isfinite(closing_radius_mm) or closing_radius_mm <= 0:
        raise ValueError("Le rayon de fermeture doit être positif en mm.")
    if threshold_method not in ("otsu", "percentile_90"):
        raise ValueError("La méthode de seuil doit être 'otsu' ou 'percentile_90'.")
    if opening_method not in ("standard", "reconstruction"):
        raise ValueError(
            "La méthode d'ouverture doit être 'standard' ou 'reconstruction'."
        )
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("Le seuil manuel doit être une valeur finie.")
    normalized, brain = robust_normalize(flair)
    otsu_value = (
        otsu_threshold(normalized[brain]) if threshold_method == "otsu" else None
    )
    if threshold is not None:
        selected_threshold = threshold
    elif threshold_method == "otsu":
        assert otsu_value is not None
        selected_threshold = otsu_value
    else:
        selected_threshold = float(np.percentile(normalized[brain], 90))
    threshold_mask = brain & (normalized > selected_threshold)
    if threshold_method == "otsu":
        threshold_formula = (
            r"t^*=\arg\max_t\omega_0(t)\omega_1(t)"
            r"\left[\mu_0(t)-\mu_1(t)\right]^2"
        )
    else:
        threshold_formula = r"t=P_{90}(z(x):x\in\Omega_{\mathrm{cerveau}})"

    opening_element = structuring_element(2.0 * opening_radius_mm, spacing_mm)
    opened = (
        opening(threshold_mask, opening_element)
        if opening_method == "standard"
        else opening_by_reconstruction(threshold_mask, opening_element)
    )
    closing_element = structuring_element(2.0 * closing_radius_mm, spacing_mm)
    closed = closing(opened, closing_element)
    filled = fill_holes(closed)
    connected = largest_component(filled)
    if not np.any(connected):
        raise EmptySegmentationError(
            "Aucune région n'a survécu au nettoyage. Vérifiez la séquence FLAIR."
        )
    if not include_diagnostic_steps:
        return connected, [], selected_threshold

    eroded = opening(connected, opening_element)
    marker = eroded if np.any(eroded) else connected.copy()
    reconstructed = reconstruct(marker, connected, np.ones((3, 3, 3), dtype=bool))
    # Toutes les graines reçoivent l'étiquette 1 : le watershed ne peut donc
    # découper cette composante en régions différentes.
    segmented = reconstructed.copy()

    steps = [
        SegmentationStep(
            "Normalisation robuste",
            r"z(x)=\frac{I(x)-\operatorname{med}(I_{\text{cerveau}})}{1.4826\,\operatorname{MAD}(I_{\text{cerveau}})}",
            "Recentre les intensités du cerveau et limite l'effet du bruit.",
            normalized,
        ),
        SegmentationStep(
            (
                "Seuil d'Otsu"
                if threshold_method == "otsu"
                else "Seuil percentile 90"
            ),
            threshold_formula,
            (
                f"Seuil Otsu : {otsu_value:.2f}. "
                f"Seuil appliqué : {selected_threshold:.2f}."
                if threshold_method == "otsu"
                else "Le seuil est le 90e percentile du cerveau normalisé. "
                f"Seuil appliqué : {selected_threshold:.2f}."
            ),
            threshold_mask,
        ),
        SegmentationStep(
            (
                "Ouverture"
                if opening_method == "standard"
                else "Ouverture par reconstruction"
            ),
            (
                r"A\circ B=(A\ominus B)\oplus B"
                if opening_method == "standard"
                else r"R_A(A\ominus B)"
            ),
            (
                "Élimine les petits points brillants avant d'étudier les régions."
                if opening_method == "standard"
                else "Élimine les petites composantes sans déformer les objets "
                "conservés."
            ),
            opened,
        ),
        SegmentationStep(
            "Fermeture et trous",
            r"A\bullet B=(A\oplus B)\ominus B;"
            r"\quad\operatorname{trous}(A)=A^c\setminus R_{A^c}(\partial A^c)",
            (
                f"Rayon de fermeture : {closing_radius_mm:.2f} mm. "
                "Raccorde les petites interruptions et remplit les trous internes."
            ),
            filled,
        ),
        SegmentationStep(
            "Reconstruction géodésique et composante",
            r"R_M(X)=\bigcup_{k\geq0}\left((X\oplus kB)\cap M\right)",
            "Garde la plus grande région, puis la reconstruit dans son masque.",
            reconstructed,
        ),
        SegmentationStep(
            "Watershed · une seule composante",
            r"\operatorname{cost}(x)=\min_{\pi\in\Pi(x,\mathcal{S})}\max_{y\in\pi}f(y)",
            (
                "Toutes les graines portent la même étiquette ; "
                "la région binaire ne change pas."
            ),
            segmented,
        ),
    ]
    return segmented, steps, selected_threshold
