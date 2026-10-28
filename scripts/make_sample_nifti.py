import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


def make_sample_nifti(output: Path, size: int) -> None:
    coordinates = np.ogrid[tuple(slice(-1.0, 1.0, complex(size)) for _ in range(3))]
    distance = sum(axis**2 for axis in coordinates)
    rng = np.random.default_rng(7)
    volume = rng.normal(0.0, 0.025, (size, size, size)).astype(np.float32)
    synthetic_region = distance <= 0.16
    smooth_region = gaussian_filter(synthetic_region.astype(np.float32), sigma=1.2)
    volume += smooth_region * 0.8
    image = nib.Nifti1Image(volume, affine=np.diag([1.0, 1.0, 1.0, 1.0]))
    image.header.set_zooms((1.0, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic demo NIfTI volume."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sample-synthetic.nii.gz"),
        help="Output path (default: sample-synthetic.nii.gz).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=64,
        help="Voxel count per axis (default: 64).",
    )
    args = parser.parse_args()
    if args.size < 8:
        parser.error("--size must be at least 8.")
    make_sample_nifti(args.output, args.size)
    print(f"Synthetic demo volume written to {args.output}")


if __name__ == "__main__":
    main()
