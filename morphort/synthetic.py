"""Génération d'images et de tumeur entièrement synthétiques pour la démo."""

import numpy as np
from numpy.typing import NDArray

FloatVolume = NDArray[np.float32]
BinaryVolume = NDArray[np.bool_]
SyntheticModalities = dict[str, FloatVolume]

MODALITY_NAMES = ("T1", "T1ce", "T2", "FLAIR")


def create_synthetic_mri(
    shape: tuple[int, int, int] = (80, 80, 64),
    seed: int = 17,
) -> tuple[SyntheticModalities, BinaryVolume]:
    """Crée quatre contrastes IRM simulés et un masque de tumeur connu.

    La tumeur est un ellipsoïde ajouté à des volumes artificiels bruités.
    Ces données aident à tester l'application, mais ne représentent aucun patient.
    """
    if len(shape) != 3 or any(size < 16 for size in shape):
        raise ValueError(
            "Le volume de démonstration doit mesurer au moins 16 voxels par axe."
        )

    axes = tuple(np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in shape)
    x, y, z = np.meshgrid(*axes, indexing="ij")
    brain = (x / 0.88) ** 2 + (y / 0.92) ** 2 + (z / 0.86) ** 2 <= 1.0
    tumor = (
        ((x + 0.13) / 0.20) ** 2 + ((y - 0.08) / 0.16) ** 2 + ((z + 0.03) / 0.24) ** 2
        <= 1.0
    ) & brain

    random = np.random.default_rng(seed)
    anatomy = 0.55 + 0.12 * np.cos(7 * x) * np.sin(6 * y) * np.cos(5 * z)
    contrasts = {
        "T1": (0.72, 0.18, 0.85),
        "T1ce": (0.70, 0.22, 1.25),
        "T2": (0.78, 0.13, 1.0),
        "FLAIR": (0.68, 0.16, 1.15),
    }

    modalities: SyntheticModalities = {}
    for name, (baseline, texture, lesion_signal) in contrasts.items():
        noise = random.normal(0.0, 0.035, size=shape).astype(np.float32)
        volume = np.zeros(shape, dtype=np.float32)
        brain_signal = baseline + texture * anatomy[brain] + noise[brain]
        volume[brain] = brain_signal
        volume[tumor] += lesion_signal
        modalities[name] = volume

    return modalities, tumor
