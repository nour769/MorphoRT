"""Segmentation morphologique exploratoire des sous-régions tumorales BraTS."""

from __future__ import annotations

import numpy as np

from morphort.morphology import (
    BinaryImage,
    FloatImage,
    closing,
    fill_holes,
    largest_component,
    opening,
    opening_by_reconstruction,
    structuring_element,
)
from morphort.segmentation import (
    SegmentationStep,
    otsu_threshold,
    robust_normalize,
)
from morphort.volumes import VoxelSpacing

ENHANCING = "Tumeur rehaussée"
NECROTIC = "Noyau nécrotique"
EDEMA = "Œdème"


def segment_tumor_subregions(
    t1ce: FloatImage,
    flair: FloatImage,
    whole_tumor: BinaryImage,
    spacing_mm: VoxelSpacing,
    *,
    opening_radius_mm: float = 1.0,
    closing_radius_mm: float = 1.0,
) -> tuple[dict[str, BinaryImage], dict[str, list[SegmentationStep]]]:
    """Estime trois sous-régions avec intensités IRM et morphologie binaire.

    La tumeur rehaussée est le seuil T1ce élevé dans la tumeur entière ;
    le noyau nécrotique est une cavité fermée et plus sombre ; l'œdème occupe
    les voxels FLAIR élevés restants après ouverture par reconstruction.
    """
    _validate_inputs(t1ce, flair, whole_tumor, spacing_mm)
    if not np.isfinite(opening_radius_mm) or opening_radius_mm <= 0:
        raise ValueError("Le rayon d'ouverture doit être positif en mm.")
    if not np.isfinite(closing_radius_mm) or closing_radius_mm <= 0:
        raise ValueError("Le rayon de fermeture doit être positif en mm.")

    normalized_t1ce, brain = robust_normalize(t1ce)
    normalized_flair, _flair_brain = robust_normalize(flair)
    t1ce_threshold = otsu_threshold(normalized_t1ce[brain])
    footprint_opening = structuring_element(
        2.0 * opening_radius_mm, spacing_mm
    )
    footprint_closing = structuring_element(
        2.0 * closing_radius_mm, spacing_mm
    )

    enhancing_threshold = whole_tumor & (normalized_t1ce > t1ce_threshold)
    enhancing_opened = opening(enhancing_threshold, footprint_opening)
    enhancing = largest_component(enhancing_opened)

    enhancing_closed = closing(enhancing, footprint_closing)
    enhancing_filled = fill_holes(enhancing_closed)
    necrotic_dark = (
        enhancing_filled
        & ~enhancing
        & whole_tumor
        & (normalized_t1ce <= t1ce_threshold)
    )

    flair_threshold = float(np.percentile(normalized_flair[whole_tumor], 50))
    edema_threshold = (
        whole_tumor
        & (normalized_flair >= flair_threshold)
        & ~enhancing
        & ~necrotic_dark
    )
    edema_opened = opening_by_reconstruction(edema_threshold, footprint_opening)
    edema = edema_opened & whole_tumor & ~enhancing & ~necrotic_dark

    masks = {
        ENHANCING: enhancing,
        NECROTIC: necrotic_dark,
        EDEMA: edema,
    }
    steps = {
        ENHANCING: [
            SegmentationStep(
                "Normalisation T1ce",
                r"z(x)=\frac{I_{T1ce}(x)-\operatorname{med}(I_{\text{cerveau}})}{1.4826\,\operatorname{MAD}(I_{\text{cerveau}})}",
                "Exprime le contraste T1ce par rapport à l'intensité cérébrale.",
                normalized_t1ce,
            ),
            SegmentationStep(
                "Seuil T1ce dans la tumeur entière",
                r"R_{4}^{(0)}=T\cap\{x:z_{T1ce}(x)>t_{\mathrm{Otsu}}\}",
                f"Seuil Otsu relatif au cerveau : {t1ce_threshold:.2f}.",
                enhancing_threshold,
            ),
            SegmentationStep(
                "Ouverture de la région rehaussée",
                r"R_{4}^{(1)}=R_{4}^{(0)}\circ B_{2r}",
                (
                    "Retire les petits détails fins avec un rayon de "
                    f"{opening_radius_mm:g} mm."
                ),
                enhancing_opened,
            ),
            SegmentationStep(
                "Plus grande composante rehaussée",
                r"R_{4}=\operatorname{argmax}_{C\subset R_{4}^{(1)}}|C|",
                "Conserve la région rehaussée connexe la plus volumineuse.",
                enhancing,
            ),
        ],
        NECROTIC: [
            SegmentationStep(
                "Fermeture de la région rehaussée",
                r"R_{4}^{c}=R_{4}\bullet B_{2r}",
                (
                    "Ferme les petites ouvertures avec un rayon de "
                    f"{closing_radius_mm:g} mm."
                ),
                enhancing_closed,
            ),
            SegmentationStep(
                "Remplissage géodésique des cavités",
                r"H=\operatorname{trous}(R_{4}^{c})",
                (
                    "La reconstruction du fond depuis le bord distingue "
                    "trous internes et fond extérieur."
                ),
                enhancing_filled,
            ),
            SegmentationStep(
                "Validation par T1ce sombre",
                r"R_{1}=(H\setminus R_{4})\cap T\cap\{z_{T1ce}\leq t_{\mathrm{Otsu}}\}",
                (
                    "Ne garde que les cavités dans la tumeur entière dont le "
                    "T1ce est sous le seuil."
                ),
                necrotic_dark,
            ),
        ],
        EDEMA: [
            SegmentationStep(
                "Seuil FLAIR dans la tumeur entière",
                (
                    r"R_{2}^{(0)}=T\cap"
                    r"\{z_{\mathrm{FLAIR}}\geq P_{50}(z_{\mathrm{FLAIR}}|T)\}"
                ),
                f"Seuil médian FLAIR dans la tumeur entière : {flair_threshold:.2f}.",
                whole_tumor & (normalized_flair >= flair_threshold),
            ),
            SegmentationStep(
                "Retrait des autres régions",
                r"R_{2}^{(1)}=R_{2}^{(0)}\setminus(R_{4}\cup R_{1})",
                "L'œdème ne recouvre ni la région rehaussée ni le noyau nécrotique.",
                edema_threshold,
            ),
            SegmentationStep(
                "Ouverture par reconstruction",
                r"R_{2}=R_T(R_{2}^{(1)}\ominus B_{2r})",
                (
                    "Supprime les petites composantes sans déformer "
                    "les contours conservés."
                ),
                edema,
            ),
        ],
    }
    return masks, steps


def _validate_inputs(
    t1ce: FloatImage,
    flair: FloatImage,
    whole_tumor: BinaryImage,
    spacing_mm: VoxelSpacing,
) -> None:
    """Vérifie les volumes, le masque global et les espacements physiques."""
    if (
        t1ce.ndim != 3
        or flair.ndim != 3
        or whole_tumor.ndim != 3
        or t1ce.shape != flair.shape
        or t1ce.shape != whole_tumor.shape
    ):
        raise ValueError(
            "T1ce, FLAIR et tumeur entière doivent avoir la même forme 3D."
        )
    if whole_tumor.dtype != np.bool_:
        raise TypeError("Le masque de tumeur entière doit être booléen.")
    if not whole_tumor.any():
        raise ValueError(
            "La tumeur entière est vide ; les sous-régions ne sont pas calculables."
        )
    if len(spacing_mm) != 3 or any(
        not np.isfinite(value) or value <= 0 for value in spacing_mm
    ):
        raise ValueError("L'espacement doit contenir trois valeurs positives en mm.")
