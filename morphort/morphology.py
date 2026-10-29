"""Opérations de morphologie mathématique écrites avec NumPy."""

from __future__ import annotations

from heapq import heappop, heappush
from itertools import product
from math import ceil, isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

BinaryImage = NDArray[np.bool_]
IntegerImage = NDArray[np.int32]
FloatImage = NDArray[np.float64]
VoxelSpacing = tuple[float, ...]


def _validate_binary_image(image: BinaryImage) -> None:
    """Vérifie qu'un masque est un tableau booléen 2D ou 3D non vide."""
    if image.ndim not in (2, 3) or any(length == 0 for length in image.shape):
        raise ValueError("Le masque doit être une image 2D ou 3D non vide.")
    if image.dtype != np.bool_:
        raise TypeError("Le masque doit contenir des valeurs booléennes.")


def _validate_spacing(spacing_mm: VoxelSpacing, dimensions: int) -> None:
    """Vérifie que chaque axe possède un espacement physique positif."""
    if (
        dimensions not in (2, 3)
        or len(spacing_mm) != dimensions
        or any(not isfinite(value) or value <= 0 for value in spacing_mm)
    ):
        raise ValueError(
            "L'espacement doit contenir une valeur positive en mm par axe."
        )


def _neighbor_offsets(dimensions: int) -> tuple[tuple[int, ...], ...]:
    """Retourne les voisins directs diagonaux inclus, sans le centre."""
    return tuple(
        offset
        for offset in product((-1, 0, 1), repeat=dimensions)
        if any(value != 0 for value in offset)
    )


def structuring_element(size_mm: float, spacing_mm: VoxelSpacing) -> BinaryImage:
    """Construit une boule discrète de diamètre `size_mm`.

    Formule : B = {x : somme((x_i * s_i)^2) <= (d / 2)^2}.
    L'espacement `s_i` convertit les pas de voxel en distance physique.
    """
    _validate_spacing(spacing_mm, len(spacing_mm))
    if not isfinite(size_mm) or size_mm <= 0:
        raise ValueError("La taille de l'élément doit être positive en mm.")

    radius_voxels = tuple(ceil(size_mm / (2 * spacing)) for spacing in spacing_mm)
    axes = tuple(
        np.arange(-radius, radius + 1, dtype=np.float64) * spacing
        for radius, spacing in zip(radius_voxels, spacing_mm, strict=True)
    )
    coordinates = np.meshgrid(*axes, indexing="ij")
    distance_squared = sum(coordinate**2 for coordinate in coordinates)
    footprint = distance_squared <= (size_mm / 2) ** 2
    points = np.argwhere(footprint)
    bounds = tuple(
        slice(int(points[:, axis].min()), int(points[:, axis].max()) + 1)
        for axis in range(footprint.ndim)
    )
    return footprint[bounds]


def _shift(
    image: BinaryImage, offset: tuple[int, ...], outside_value: bool
) -> BinaryImage:
    """Déplace un masque sans faire revenir les pixels par le bord opposé."""
    padding = tuple((max(0, -step), max(0, step)) for step in offset)
    padded = np.pad(image, padding, constant_values=outside_value)
    slices = tuple(
        slice(before + step, before + step + length)
        for (before, _), step, length in zip(padding, offset, image.shape, strict=True)
    )
    return padded[slices]


def dilate(image: BinaryImage, footprint: BinaryImage) -> BinaryImage:
    """Agrandit l'objet par union des translations de B.

    Formule : A ⊕ B = union des translations de A par les points de B.
    Les pixels hors image valent zéro afin de ne pas créer d'objet aux bords.
    """
    _validate_binary_image(image)
    _validate_binary_image(footprint)
    if image.ndim != footprint.ndim:
        raise ValueError(
            "L'image et l'élément structurant ont des dimensions différentes."
        )
    center = tuple(length // 2 for length in footprint.shape)
    result = np.zeros_like(image)
    for location in np.argwhere(footprint):
        offset = tuple(
            int(position - middle)
            for position, middle in zip(location, center, strict=True)
        )
        result |= _shift(image, offset, outside_value=False)
    return result


def erode(
    image: BinaryImage,
    footprint: BinaryImage,
    *,
    outside_value: bool = False,
) -> BinaryImage:
    """Conserve les points dont toute la translation de B tient dans A.

    Formule : A ⊖ B = {x : x + B est inclus dans A}.
    Par défaut, les pixels hors image valent zéro, comme pour une image entourée
    de fond. La fermeture utilise un bord neutre à un pour préserver les bords.
    """
    _validate_binary_image(image)
    _validate_binary_image(footprint)
    if image.ndim != footprint.ndim:
        raise ValueError(
            "L'image et l'élément structurant ont des dimensions différentes."
        )
    center = tuple(length // 2 for length in footprint.shape)
    result = np.ones_like(image)
    for location in np.argwhere(footprint):
        offset = tuple(
            int(position - middle)
            for position, middle in zip(location, center, strict=True)
        )
        result &= _shift(image, offset, outside_value=outside_value)
    return result


def opening(image: BinaryImage, footprint: BinaryImage) -> BinaryImage:
    """Supprime les petits objets par érosion puis dilatation.

    Formule : A ○ B = (A ⊖ B) ⊕ B.
    Les objets plus petits que B sont éliminés avant de regagner leur forme.
    """
    return dilate(erode(image, footprint), footprint)


def closing(image: BinaryImage, footprint: BinaryImage) -> BinaryImage:
    """Comble les petits trous par dilatation puis érosion.

    Formule : A ● B = (A ⊕ B) ⊖ B.
    Les petits creux sont comblés sans agrandir durablement l'objet.
    """
    return erode(dilate(image, footprint), footprint, outside_value=True)


def gradient(image: BinaryImage, footprint: BinaryImage) -> BinaryImage:
    """Extrait la frontière morphologique de l'objet.

    Formule : gradient(A) = (A ⊕ B) moins (A ⊖ B).
    La frontière apparaît là où l'érosion et la dilatation diffèrent.
    """
    return dilate(image, footprint) & ~erode(image, footprint)


def top_hat(
    image: BinaryImage,
    footprint: BinaryImage,
    kind: Literal["white", "black"] = "white",
) -> BinaryImage:
    """Extrait les détails retirés par l'ouverture ou comblés par la fermeture.

    Formule blanche : A moins (A ○ B) ; formule noire : (A ● B) moins A.
    Le choix `kind` indique si l'on cherche les petites bosses ou les petits trous.
    """
    _validate_binary_image(image)
    if kind == "white":
        return image & ~opening(image, footprint)
    if kind == "black":
        return closing(image, footprint) & ~image
    raise ValueError("Le top-hat doit être de type 'white' ou 'black'.")


def reconstruct(
    marker: BinaryImage, mask: BinaryImage, footprint: BinaryImage
) -> BinaryImage:
    """Propage un marqueur dans un masque par dilatations géodésiques.

    Formule : R_M(marker) = union_k ((marker ⊕ kB) inter M).
    La propagation s'arrête aux limites du masque, à convergence.
    """
    _validate_binary_image(marker)
    _validate_binary_image(mask)
    _validate_binary_image(footprint)
    if marker.shape != mask.shape or marker.ndim != footprint.ndim:
        raise ValueError(
            "Le marqueur, le masque et l'élément doivent être compatibles."
        )
    if np.any(marker & ~mask):
        raise ValueError("Le marqueur doit être inclus dans le masque.")

    if footprint.shape == (3,) * mask.ndim and np.all(footprint):
        return _select_connected_regions(mask, marker, largest_only=False)

    result = marker.copy()
    while True:
        expanded = dilate(result, footprint) & mask
        if np.array_equal(expanded, result):
            return result
        result = expanded


def fill_holes(image: BinaryImage) -> BinaryImage:
    """Remplit les régions de fond qui ne touchent pas le bord de l'image.

    Formule : trous = fond moins reconstruction du fond de bord.
    Le fond extérieur est reconstruit avec une connexité diagonale en 2D/3D.
    """
    _validate_binary_image(image)
    background = ~image
    border = np.zeros_like(image)
    for axis in range(image.ndim):
        first = [slice(None)] * image.ndim
        last = [slice(None)] * image.ndim
        first[axis] = 0
        last[axis] = -1
        border[tuple(first)] |= background[tuple(first)]
        border[tuple(last)] |= background[tuple(last)]

    footprint = np.ones((3,) * image.ndim, dtype=np.bool_)
    outside = reconstruct(border, background, footprint)
    return image | (background & ~outside)


def opening_by_reconstruction(
    image: BinaryImage, footprint: BinaryImage
) -> BinaryImage:
    """Supprime les petites composantes sans déformer les autres objets.

    Formule : ouverture par reconstruction = R_A(A ⊖ B).
    L'érosion choisit les marqueurs et la reconstruction restaure leur contour.
    """
    marker = erode(image, footprint)
    return reconstruct(marker, image, footprint)


def largest_component(image: BinaryImage) -> BinaryImage:
    """Garde la plus grande composante connexe d'un masque.

    Formule : composante maximale = argmax_C |C| pour la connexité choisie.
    La connexité inclut les voisins diagonaux et fonctionne en 2D et en 3D.
    """
    _validate_binary_image(image)
    return _select_connected_regions(image, image, largest_only=True)


def _select_connected_regions(
    image: BinaryImage, seeds: BinaryImage, *, largest_only: bool
) -> BinaryImage:
    """Sélectionne les composantes 8/26-connexes par fusion de segments NumPy."""
    row_length = image.shape[-1]
    transition_width = row_length + 1
    row_count = int(np.prod(image.shape[:-1]))
    rows = image.reshape(row_count, row_length)
    seed_rows = seeds.reshape(row_count, row_length)

    padded = np.pad(rows, ((0, 0), (1, 1)), constant_values=False)
    transitions = np.diff(padded.astype(np.int8), axis=1)
    starts_flat = np.flatnonzero(transitions == 1)
    ends_flat = np.flatnonzero(transitions == -1)
    if starts_flat.size == 0:
        return np.zeros_like(image)

    run_rows = starts_flat // transition_width
    run_starts = starts_flat % transition_width
    run_ends = ends_flat % transition_width - 1
    run_lengths = run_ends - run_starts + 1
    run_offsets = np.zeros(row_count + 1, dtype=np.int64)
    np.cumsum(np.bincount(run_rows, minlength=row_count), out=run_offsets[1:])

    parents = np.arange(starts_flat.size, dtype=np.int32)

    def find_root(run_id: int) -> int:
        root = run_id
        while parents[root] != root:
            root = int(parents[root])
        while parents[run_id] != run_id:
            next_id = int(parents[run_id])
            parents[run_id] = root
            run_id = next_id
        return root

    def join_rows(previous_row: int, current_row: int) -> None:
        previous_start = int(run_offsets[previous_row])
        previous_stop = int(run_offsets[previous_row + 1])
        current_start = int(run_offsets[current_row])
        current_stop = int(run_offsets[current_row + 1])
        previous_run = previous_start
        for current_run in range(current_start, current_stop):
            first = int(run_starts[current_run])
            last = int(run_ends[current_run])
            while (
                previous_run < previous_stop
                and run_ends[previous_run] < first - 1
            ):
                previous_run += 1
            candidate = previous_run
            while (
                candidate < previous_stop
                and run_starts[candidate] <= last + 1
            ):
                first_root = find_root(current_run)
                second_root = find_root(candidate)
                if first_root != second_root:
                    parents[second_root] = first_root
                candidate += 1

    if image.ndim == 2:
        for row in range(1, row_count):
            join_rows(row - 1, row)
    else:
        columns_per_plane = image.shape[-2]
        for row in range(row_count):
            row_in_plane = row % columns_per_plane
            if row_in_plane > 0:
                join_rows(row - 1, row)
            if row >= columns_per_plane:
                if row_in_plane > 0:
                    join_rows(row - columns_per_plane - 1, row)
                join_rows(row - columns_per_plane, row)
                if row_in_plane + 1 < columns_per_plane:
                    join_rows(row - columns_per_plane + 1, row)

    roots = np.fromiter(
        (find_root(run_id) for run_id in range(starts_flat.size)),
        dtype=np.int32,
        count=starts_flat.size,
    )
    run_seed_prefix = np.pad(
        np.cumsum(seed_rows, axis=1, dtype=np.int32),
        ((0, 0), (1, 0)),
    )
    seed_counts = (
        run_seed_prefix[run_rows, run_ends + 1]
        - run_seed_prefix[run_rows, run_starts]
    )
    seeded_roots = np.unique(roots[seed_counts > 0])
    if seeded_roots.size == 0:
        return np.zeros_like(image)

    if largest_only:
        component_sizes = np.bincount(roots, weights=run_lengths)
        selected_roots = np.array(
            [int(np.argmax(component_sizes))], dtype=np.int32
        )
    else:
        selected_roots = seeded_roots

    selected_runs = np.isin(roots, selected_roots)
    result_rows = np.zeros_like(rows)
    for run_id in np.flatnonzero(selected_runs):
        result_rows[
            run_rows[run_id], run_starts[run_id] : run_ends[run_id] + 1
        ] = True
    return result_rows.reshape(image.shape)


def watershed(
    elevation: FloatImage,
    markers: IntegerImage,
    mask: BinaryImage | None = None,
) -> IntegerImage:
    """Sépare un relief en propageant les marqueurs depuis les bassins.

    Formule : coût du chemin = maximum des altitudes rencontrées.
    Un voxel reçoit l'étiquette du marqueur atteignant son coût minimal.
    """
    if elevation.ndim not in (2, 3) or elevation.size == 0:
        raise ValueError("Le relief doit être une image 2D ou 3D non vide.")
    if not np.all(np.isfinite(elevation)):
        raise ValueError("Le relief doit contenir uniquement des valeurs finies.")
    if markers.shape != elevation.shape or not np.issubdtype(markers.dtype, np.integer):
        raise ValueError("Les marqueurs entiers doivent avoir la forme du relief.")
    if np.any(markers < 0):
        raise ValueError("Les marqueurs ne peuvent pas être négatifs.")

    allowed = np.ones(elevation.shape, dtype=np.bool_) if mask is None else mask
    _validate_binary_image(allowed)
    if allowed.shape != elevation.shape or np.any((markers > 0) & ~allowed):
        raise ValueError(
            "Le masque doit avoir la forme du relief et contenir les marqueurs."
        )

    best_level = np.full(elevation.shape, np.inf, dtype=np.float64)
    labels = np.zeros(elevation.shape, dtype=np.int32)
    queue: list[tuple[float, tuple[int, ...], int]] = []
    for location in np.argwhere((markers > 0) & allowed):
        point = tuple(int(value) for value in location)
        label = int(markers[point])
        best_level[point] = elevation[point]
        labels[point] = label
        heappush(queue, (float(elevation[point]), point, label))

    if not queue:
        raise ValueError("Le watershed nécessite au moins un marqueur.")

    while queue:
        level, point, label = heappop(queue)
        if level != best_level[point] or labels[point] != label:
            continue
        for offset in _neighbor_offsets(elevation.ndim):
            neighbor = tuple(
                coordinate + step
                for coordinate, step in zip(point, offset, strict=True)
            )
            if not all(
                0 <= coordinate < length
                for coordinate, length in zip(neighbor, elevation.shape, strict=True)
            ):
                continue
            if not allowed[neighbor] or markers[neighbor] > 0:
                continue
            new_level = max(level, float(elevation[neighbor]))
            if new_level < best_level[neighbor]:
                best_level[neighbor] = new_level
                labels[neighbor] = label
                heappush(queue, (new_level, neighbor, label))
            elif new_level == best_level[neighbor] and labels[neighbor] != label:
                labels[neighbor] = 0

    return labels


def skeleton(image: BinaryImage, footprint: BinaryImage) -> BinaryImage:
    """Calcule le squelette morphologique par couches d'érosion.

    Formule : S(A) = union_k ((A ⊖ kB) moins ((A ⊖ kB) ○ B)).
    Chaque couche conserve les centres qui disparaissent à cette échelle.
    """
    _validate_binary_image(image)
    _validate_binary_image(footprint)
    if image.ndim != footprint.ndim:
        raise ValueError(
            "L'image et l'élément structurant ont des dimensions différentes."
        )
    if np.count_nonzero(footprint) == 1:
        return image.copy()
    current = image.copy()
    result = np.zeros_like(image)
    while np.any(current):
        opened_current = opening(current, footprint)
        result |= current & ~opened_current
        eroded = erode(current, footprint)
        if np.array_equal(eroded, current):
            break
        current = eroded
    return result


def granulometry(
    image: BinaryImage, sizes_mm: tuple[float, ...], spacing_mm: VoxelSpacing
) -> list[dict[str, float]]:
    """Mesure la taille restante après des ouvertures de diamètre croissant.

    Formule : G(d) = |A ○ B_d| / |A|.
    La fraction restante décroît lorsque le diamètre de l'élément augmente.
    """
    _validate_binary_image(image)
    _validate_spacing(spacing_mm, image.ndim)
    if not sizes_mm or any(not isfinite(size) or size <= 0 for size in sizes_mm):
        raise ValueError("La granulométrie nécessite des tailles positives en mm.")
    original_size = int(np.count_nonzero(image))
    measurements: list[dict[str, float]] = []
    previous_size = 0.0
    for size_mm in sizes_mm:
        if size_mm <= previous_size:
            raise ValueError("Les tailles doivent être strictement croissantes.")
        footprint = structuring_element(size_mm, spacing_mm)
        remaining = int(np.count_nonzero(opening(image, footprint)))
        fraction = remaining / original_size if original_size else 0.0
        measurements.append({"size_mm": float(size_mm), "fraction_remaining": fraction})
        previous_size = size_mm
    return measurements
