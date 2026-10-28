"""Vérifie les opérations NumPy par rapport à scikit-image."""

import numpy as np
import pytest
from skimage import morphology as sk_morphology

from morphort.morphology import (
    closing,
    dilate,
    erode,
    fill_holes,
    gradient,
    granulometry,
    largest_component,
    opening,
    opening_by_reconstruction,
    reconstruct,
    skeleton,
    structuring_element,
    top_hat,
    watershed,
)


@pytest.mark.parametrize("shape", [(17, 15), (9, 10, 8)])
def test_basic_operations_match_scikit_image(shape: tuple[int, ...]) -> None:
    """Les opérations binaires donnent le même résultat en 2D et 3D."""
    random = np.random.default_rng(42)
    image = random.random(shape) > 0.65
    footprint = np.ones((3,) * len(shape), dtype=bool)

    np.testing.assert_array_equal(
        dilate(image, footprint),
        sk_morphology.binary_dilation(image, footprint=footprint),
    )
    interior = tuple(slice(1, -1) for _ in shape)
    np.testing.assert_array_equal(
        erode(image, footprint)[interior],
        sk_morphology.binary_erosion(image, footprint=footprint)[interior],
    )
    np.testing.assert_array_equal(
        opening(image, footprint)[tuple(slice(2, -2) for _ in shape)],
        sk_morphology.binary_opening(image, footprint=footprint)[
            tuple(slice(2, -2) for _ in shape)
        ],
    )
    np.testing.assert_array_equal(
        closing(image, footprint),
        sk_morphology.binary_closing(image, footprint=footprint),
    )


@pytest.mark.parametrize("shape", [(15, 13), (8, 9, 10)])
def test_gradient_and_top_hat_match_definitions(shape: tuple[int, ...]) -> None:
    """Le gradient et les deux top-hat respectent leurs définitions."""
    random = np.random.default_rng(7)
    image = random.random(shape) > 0.55
    footprint = np.ones((3,) * len(shape), dtype=bool)

    expected_gradient = sk_morphology.binary_dilation(
        image, footprint=footprint
    ) & ~sk_morphology.binary_erosion(image, footprint=footprint)
    expected_white = image & ~sk_morphology.binary_opening(
        image, footprint=footprint
    )
    expected_black = sk_morphology.binary_closing(
        image, footprint=footprint
    ) & ~image

    np.testing.assert_array_equal(gradient(image, footprint), expected_gradient)
    np.testing.assert_array_equal(top_hat(image, footprint), expected_white)
    np.testing.assert_array_equal(
        top_hat(image, footprint, kind="black"), expected_black
    )


def test_structuring_element_uses_physical_voxel_spacing() -> None:
    """Un voxel plus large en mm contribue moins à l'élément structurant."""
    footprint = structuring_element(3.0, (1.0, 2.0))

    np.testing.assert_array_equal(
        footprint,
        np.array(
            [[True], [True], [True]]
        ),
    )


@pytest.mark.parametrize("shape", [(13, 14), (7, 8, 9)])
def test_hole_filling_matches_scikit_image(shape: tuple[int, ...]) -> None:
    """Le remplissage conserve l'objet extérieur et bouche un trou fermé."""
    image = np.zeros(shape, dtype=bool)
    center = tuple(length // 2 for length in shape)
    slices = tuple(slice(index - 2, index + 3) for index in center)
    image[slices] = True
    hole = tuple(slice(index, index + 1) for index in center)
    image[hole] = False

    expected = sk_morphology.remove_small_holes(image, max_size=2)
    np.testing.assert_array_equal(fill_holes(image), expected)


@pytest.mark.parametrize("shape", [(15, 13), (8, 9, 10)])
def test_geodesic_reconstruction_matches_scikit_image(
    shape: tuple[int, ...],
) -> None:
    """La propagation géodésique s'arrête aux limites du masque."""
    mask = np.zeros(shape, dtype=bool)
    center = tuple(length // 2 for length in shape)
    slices = tuple(slice(index - 2, index + 3) for index in center)
    mask[slices] = True
    marker = np.zeros_like(mask)
    marker[center] = True
    footprint = np.ones((3,) * len(shape), dtype=bool)

    expected = sk_morphology.reconstruction(
        marker.astype(np.uint8),
        mask.astype(np.uint8),
        method="dilation",
        footprint=footprint,
    ).astype(bool)
    np.testing.assert_array_equal(reconstruct(marker, mask, footprint), expected)


def test_opening_by_reconstruction_and_largest_component() -> None:
    """La reconstruction garde les objets marqués et retire le petit objet."""
    image = np.zeros((25, 30), dtype=bool)
    image[3:17, 4:18] = True
    image[21, 26] = True
    footprint = np.ones((3, 3), dtype=bool)
    eroded = sk_morphology.binary_erosion(image, footprint=footprint)
    expected_opening = sk_morphology.reconstruction(
        eroded.astype(np.uint8),
        image.astype(np.uint8),
        method="dilation",
        footprint=footprint,
    ).astype(bool)

    np.testing.assert_array_equal(
        opening_by_reconstruction(image, footprint), expected_opening
    )
    np.testing.assert_array_equal(
        largest_component(image),
        sk_morphology.remove_small_objects(image, min_size=2, connectivity=2),
    )


def test_watershed_preserves_markers_and_stays_inside_mask() -> None:
    """Le watershed conserve les graines et n'envahit pas le fond interdit."""
    elevation = np.tile(np.array([0, 1, 2, 5, 2, 1, 0]), (7, 1)).astype(float)
    markers = np.zeros_like(elevation, dtype=np.int32)
    markers[3, 1] = 1
    markers[3, 5] = 2
    mask = np.ones_like(elevation, dtype=bool)
    mask[:, 0] = False
    mask[3, 1] = True
    mask[3, 5] = True

    result = watershed(elevation, markers, mask)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result[markers > 0], markers[markers > 0])
    assert np.all(result[~mask] == 0)
    assert np.all(result[:, 1:3] == 1)
    assert np.all(result[:, 3] == 0)
    assert np.all(result[:, 4:] == 2)


def test_skeleton_and_granulometry_are_well_behaved() -> None:
    """Le squelette reste dans l'objet et les granulométries décroissent."""
    image = np.zeros((31, 31), dtype=bool)
    image[5:26, 11:20] = True
    footprint = np.ones((3, 3), dtype=bool)

    thin = skeleton(image, footprint)
    assert np.any(thin)
    assert np.all(thin <= image)
    assert np.array_equal(skeleton(image, np.ones((1, 1), dtype=bool)), image)

    measurements = granulometry(image, (1.0, 3.0, 5.0), (1.0, 1.0))
    fractions = [item["fraction_remaining"] for item in measurements]
    assert fractions[0] >= fractions[1] >= fractions[2]


def test_invalid_inputs_are_reported() -> None:
    """Les dimensions et paramètres invalides sont refusés explicitement."""
    with pytest.raises(ValueError, match="2D ou 3D"):
        dilate(np.zeros((2, 2, 2, 2), dtype=bool), np.ones((3,) * 4, dtype=bool))
    with pytest.raises(ValueError, match="positive"):
        structuring_element(3.0, (1.0, 0.0))
    with pytest.raises(ValueError, match="strictement croissantes"):
        granulometry(np.ones((5, 5), dtype=bool), (3.0, 2.0), (1.0, 1.0))
