"""Segmentation pédagogique par seuillage et morphologie mathématique."""

from dataclasses import dataclass

import numpy as np

from morphort.morphology import (
    BinaryImage,
    FloatImage,
    closing,
    dilate,
    erode,
    fill_holes,
    largest_component,
    opening,
    reconstruct,
    structuring_element,
    watershed,
)


@dataclass(frozen=True)
class SegmentationStep:
    """Image produite à une étape et explication mathématique associée."""

    name: str
    formula: str
    explanation: str
    image: BinaryImage | FloatImage


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
    opening_radius_mm: float = 1.0,
    threshold: float | None = None,
) -> tuple[BinaryImage, list[SegmentationStep], float]:
    """Exécute le pipeline de segmentation morphologique en 3D.

    Les étapes sont conservées pour expliquer chaque transformation à l'écran.
    Le seuil est appliqué aux intensités normalisées et le rayon est en mm.
    """
    if not np.isfinite(opening_radius_mm) or opening_radius_mm <= 0:
        raise ValueError("Le rayon morphologique doit être positif en mm.")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("Le seuil manuel doit être une valeur finie.")
    normalized, brain = robust_normalize(flair)
    otsu_value = otsu_threshold(normalized[brain])
    selected_threshold = otsu_value if threshold is None else threshold
    threshold_mask = brain & (normalized > selected_threshold)

    opening_element = structuring_element(2.0 * opening_radius_mm, spacing_mm)
    opened = opening(threshold_mask, opening_element)
    closing_element = structuring_element(3.0 * opening_radius_mm, spacing_mm)
    closed = closing(opened, closing_element)
    filled = fill_holes(closed)
    connected = largest_component(filled)
    if not np.any(connected):
        raise ValueError(
            "Aucune région n'a survécu au nettoyage. Vérifiez la séquence FLAIR."
        )

    eroded = opening(connected, opening_element)
    marker = eroded
    if not np.any(marker):
        marker = connected.copy()
    reconstructed = reconstruct(marker, connected, np.ones((3, 3, 3), dtype=bool))

    distance = _distance_from_boundary(reconstructed)
    seed_threshold = float(np.percentile(distance[reconstructed], 80))
    seeds = reconstructed & (distance >= seed_threshold)
    if not np.any(seeds):
        seeds = reconstructed.copy()
    labels = watershed(-distance, seeds.astype(np.int32), reconstructed)
    segmented = labels > 0

    steps = [
        SegmentationStep(
            "Normalisation robuste",
            r"z(x)=\frac{I(x)-\operatorname{med}(I_{\text{cerveau}})}{1.4826\,\operatorname{MAD}(I_{\text{cerveau}})}",
            "Recentre les intensités du cerveau et limite l'effet du bruit.",
            normalized,
        ),
        SegmentationStep(
            "Seuil d'Otsu",
            r"t^*=\arg\max_t\omega_0(t)\omega_1(t)\left[\mu_0(t)-\mu_1(t)\right]^2",
            (
                f"Seuil Otsu : {otsu_value:.2f}. "
                f"Seuil appliqué : {selected_threshold:.2f}."
            ),
            threshold_mask,
        ),
        SegmentationStep(
            "Ouverture",
            r"A\circ B=(A\ominus B)\oplus B",
            "Élimine les petits points brillants avant d'étudier les régions.",
            opened,
        ),
        SegmentationStep(
            "Fermeture et trous",
            r"A\bullet B=(A\oplus B)\ominus B;"
            r"\quad\operatorname{trous}(A)=A^c\setminus R_{A^c}(\partial A^c)",
            "Raccorde les petites interruptions et remplit les trous internes.",
            filled,
        ),
        SegmentationStep(
            "Reconstruction géodésique et composante",
            r"R_M(X)=\bigcup_{k\geq0}\left((X\oplus kB)\cap M\right)",
            "Garde la plus grande région, puis la reconstruit dans son masque.",
            reconstructed,
        ),
        SegmentationStep(
            "Watershed",
            r"\operatorname{cost}(x)=\min_{\pi\in\Pi(x,\mathcal{S})}\max_{y\in\pi}f(y)",
            "Délimite la région à partir des graines du relief de distance.",
            segmented,
        ),
    ]
    return segmented, steps, selected_threshold


def _distance_from_boundary(mask: BinaryImage) -> FloatImage:
    """Calcule la distance intérieure au bord sans opérateur morphologique."""
    distance = np.zeros(mask.shape, dtype=np.float64)
    frontier = mask & ~erode(mask, np.ones((3,) * mask.ndim, dtype=bool))
    distance[frontier] = 1.0

    current = frontier
    depth = 1.0
    footprint = np.ones((3,) * mask.ndim, dtype=bool)
    while np.any(current):
        expanded = dilate(current, footprint) & mask
        next_layer = expanded & (distance == 0)
        if not np.any(next_layer):
            break
        depth += 1.0
        distance[next_layer] = depth
        current = next_layer
    return distance
