"""Mesures physiques et comparaison de masques binaires."""

import numpy as np
from scipy.ndimage import distance_transform_edt

from morphort.morphology import BinaryImage, erode


def measure_mask(
    mask: BinaryImage, spacing_mm: tuple[float, float, float]
) -> dict[str, float]:
    """Calcule volume, surface, sphéricité et compacité d'un masque 3D.

    Formules : V = N_voxels * dx * dy * dz ; C = S³ / (36πV²).
    La sphéricité est le rapport de la surface de la sphère équivalente à S.
    """
    if mask.ndim != 3 or mask.dtype != np.bool_:
        raise ValueError("La mesure nécessite un masque booléen 3D.")
    if len(spacing_mm) != 3 or any(
        not np.isfinite(value) or value <= 0 for value in spacing_mm
    ):
        raise ValueError("L'espacement doit contenir trois valeurs positives en mm.")

    voxel_volume_mm3 = float(np.prod(spacing_mm))
    volume_mm3 = float(np.count_nonzero(mask)) * voxel_volume_mm3
    if volume_mm3 == 0:
        return {
            "volume_ml": 0.0,
            "surface_cm2": 0.0,
            "sphericity": 0.0,
            "compactness": 0.0,
        }

    padded = np.pad(mask, 1, constant_values=False)
    face_areas_mm2 = (
        spacing_mm[1] * spacing_mm[2],
        spacing_mm[0] * spacing_mm[2],
        spacing_mm[0] * spacing_mm[1],
    )
    surface_mm2 = 0.0
    for axis, face_area in enumerate(face_areas_mm2):
        differences = np.diff(padded.astype(np.int8), axis=axis)
        surface_mm2 += float(np.count_nonzero(differences)) * face_area

    surface_cm2 = surface_mm2 / 100.0
    volume_cm3 = volume_mm3 / 1000.0
    sphericity = (
        float(np.pi ** (1.0 / 3.0) * (6.0 * volume_cm3) ** (2.0 / 3.0) / surface_cm2)
        if surface_cm2 > 0
        else 0.0
    )
    compactness = (
        float(surface_cm2**3 / (36.0 * np.pi * volume_cm3**2))
        if volume_cm3 > 0
        else 0.0
    )
    return {
        "volume_ml": volume_cm3,
        "surface_cm2": surface_cm2,
        "sphericity": sphericity,
        "compactness": compactness,
    }


def dice_coefficient(predicted: BinaryImage, expert: BinaryImage) -> float:
    """Mesure le recouvrement entre segmentation automatique et masque expert.

    Formule : Dice(P,G) = 2|P ∩ G| / (|P| + |G|).
    Deux masques vides ont par convention un recouvrement parfait.
    """
    if predicted.shape != expert.shape:
        raise ValueError("Les masques à comparer doivent avoir la même dimension.")
    if predicted.dtype != np.bool_ or expert.dtype != np.bool_:
        raise TypeError("Les masques à comparer doivent être booléens.")
    total = int(np.count_nonzero(predicted)) + int(np.count_nonzero(expert))
    if total == 0:
        return 1.0
    overlap = int(np.count_nonzero(predicted & expert))
    return 2.0 * overlap / total


def jaccard_index(predicted: BinaryImage, expert: BinaryImage) -> float:
    """Calcule l'IoU : intersection sur union des deux masques."""
    _validate_matching_masks(predicted, expert)
    union = int(np.count_nonzero(predicted | expert))
    if union == 0:
        return 1.0
    intersection = int(np.count_nonzero(predicted & expert))
    return intersection / union


def hausdorff_distance_mm(
    predicted: BinaryImage,
    expert: BinaryImage,
    spacing_mm: tuple[float, float, float],
) -> float:
    """Calcule la distance de Hausdorff symétrique entre les surfaces en mm."""
    _validate_matching_masks(predicted, expert)
    if predicted.ndim != 3:
        raise ValueError("Hausdorff nécessite deux masques 3D.")
    if len(spacing_mm) != 3 or any(
        not np.isfinite(value) or value <= 0 for value in spacing_mm
    ):
        raise ValueError("L'espacement doit contenir trois valeurs positives en mm.")

    footprint = np.ones((3, 3, 3), dtype=bool)
    predicted_surface = predicted & ~erode(predicted, footprint)
    expert_surface = expert & ~erode(expert, footprint)
    if not predicted_surface.any() and not expert_surface.any():
        return 0.0
    if not predicted_surface.any() or not expert_surface.any():
        return float("inf")

    distance_to_expert = distance_transform_edt(~expert_surface, sampling=spacing_mm)
    distance_to_predicted = distance_transform_edt(
        ~predicted_surface, sampling=spacing_mm
    )
    return float(
        max(
            np.max(distance_to_expert[predicted_surface]),
            np.max(distance_to_predicted[expert_surface]),
        )
    )


def hausdorff_distance_95_mm(
    predicted: BinaryImage,
    expert: BinaryImage,
    spacing_mm: tuple[float, float, float],
) -> float:
    """Calcule le Hausdorff percentile symétrique des surfaces en millimètres.

    Formule : HD95 = max(P95(d(P,G)), P95(d(G,P))).
    Ce percentile réduit l'influence d'un voxel isolé tout en gardant les
    distances physiques calculées avec l'espacement des voxels.
    """
    _validate_matching_masks(predicted, expert)
    if predicted.ndim != 3:
        raise ValueError("Hausdorff 95 nécessite deux masques 3D.")
    if len(spacing_mm) != 3 or any(
        not np.isfinite(value) or value <= 0 for value in spacing_mm
    ):
        raise ValueError("L'espacement doit contenir trois valeurs positives en mm.")

    footprint = np.ones((3, 3, 3), dtype=bool)
    predicted_surface = predicted & ~erode(predicted, footprint)
    expert_surface = expert & ~erode(expert, footprint)
    if not predicted_surface.any() and not expert_surface.any():
        return 0.0
    if not predicted_surface.any() or not expert_surface.any():
        return float("inf")

    distance_to_expert = distance_transform_edt(~expert_surface, sampling=spacing_mm)
    distance_to_predicted = distance_transform_edt(
        ~predicted_surface, sampling=spacing_mm
    )
    return float(
        max(
            np.percentile(distance_to_expert[predicted_surface], 95),
            np.percentile(distance_to_predicted[expert_surface], 95),
        )
    )


def _validate_matching_masks(predicted: BinaryImage, expert: BinaryImage) -> None:
    """Vérifie que les masques sont booléens et de même forme."""
    if predicted.shape != expert.shape:
        raise ValueError("Les masques à comparer doivent avoir la même dimension.")
    if predicted.dtype != np.bool_ or expert.dtype != np.bool_:
        raise TypeError("Les masques à comparer doivent être booléens.")
