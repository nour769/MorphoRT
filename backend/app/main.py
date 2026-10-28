import base64
import csv
import logging
import os
import statistics
import tempfile
import zlib
from pathlib import Path
from typing import Annotated, Literal

import nibabel as nib
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from app.pipeline.base import ModelName, PipelineBackend
from app.pipeline.mock import MockBackend
from app.pipeline.real import RealBackend

MAX_UPLOAD_BYTES = 300 * 1024 * 1024
RESULTS_FILE = Path(__file__).resolve().parents[1] / "data" / "results.csv"
logger = logging.getLogger(__name__)
BACKEND_NAME = os.getenv("PIPELINE_BACKEND", "mock").strip().lower()
if BACKEND_NAME == "mock":
    DATA_SOURCE: Literal["mock", "real"] = "mock"
elif BACKEND_NAME == "real":
    DATA_SOURCE = "real"
else:
    raise RuntimeError("PIPELINE_BACKEND must be either 'mock' or 'real'.")


def get_backend() -> tuple[PipelineBackend, Literal["mock", "real"]]:
    if DATA_SOURCE == "mock":
        return MockBackend(), "mock"
    if DATA_SOURCE == "real":
        return RealBackend(), "real"
    raise RuntimeError("The configured pipeline backend is invalid.")


app = FastAPI(title="MorphoRT Platform API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def handle_http_exception(
    _request: Request, error: HTTPException
) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"data_source": DATA_SOURCE, "detail": error.detail},
        headers=error.headers,
    )


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    _request: Request, error: RequestValidationError
) -> JSONResponse:
    details = [
        {
            "field": ".".join(str(part) for part in issue["loc"][1:]),
            "message": issue["msg"],
        }
        for issue in error.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"data_source": DATA_SOURCE, "detail": details},
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(
    _request: Request, error: Exception
) -> JSONResponse:
    logger.error(
        "Unhandled API exception",
        exc_info=(type(error), error, error.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={
            "data_source": DATA_SOURCE,
            "detail": "Une erreur interne est survenue.",
        },
    )


class VolumePoint(BaseModel):
    day: float = Field(ge=0)
    volume_cm3: float = Field(gt=0)


class GrowthRequest(BaseModel):
    volumes_by_day: list[VolumePoint] = Field(min_length=3, max_length=500)
    model: ModelName


class RadiotherapyRequest(BaseModel):
    volume_cm3: float = Field(gt=0)
    r: float = Field(ge=0)
    K: float = Field(gt=0)
    alpha: float = Field(gt=0)
    beta: float = Field(gt=0)
    bed_max: float = Field(gt=0)
    n_fractions: int = Field(ge=1, le=100)


def _check_nifti_filename(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Le nom du fichier est manquant.")
    lower_name = filename.lower()
    if lower_name.endswith(".nii.gz"):
        return ".nii.gz"
    if lower_name.endswith(".nii"):
        return ".nii"
    raise HTTPException(
        status_code=415,
        detail=(
            "Format non pris en charge. Fournissez un fichier NIfTI .nii ou .nii.gz."
        ),
    )


async def _read_upload(upload: UploadFile) -> bytes:
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Le fichier dépasse la taille maximale autorisée de 300 Mo.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Le fichier transmis est vide.")
    return content


def _load_nifti(
    content: bytes, suffix: str
) -> tuple[np.ndarray, tuple[float, float, float]]:
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(content)
            temporary_path = temporary_file.name
        image = nib.load(temporary_path)
        if len(image.shape) != 3:
            raise HTTPException(
                status_code=422,
                detail="Le volume NIfTI doit être une image 3D.",
            )
        volume = np.asarray(image.get_fdata(dtype=np.float32), dtype=np.float32)
        voxel_size = tuple(float(size) for size in image.header.get_zooms()[:3])
        if len(voxel_size) != 3 or any(size <= 0 for size in voxel_size):
            raise HTTPException(
                status_code=422,
                detail="Les dimensions des voxels doivent être positives.",
            )
        return volume, voxel_size
    except HTTPException:
        raise
    except (
        OSError,
        EOFError,
        ValueError,
        zlib.error,
        nib.filebasedimages.ImageFileError,
    ) as error:
        raise HTTPException(
            status_code=422,
            detail="Le fichier ne peut pas être lu comme un volume NIfTI valide.",
        ) from error
    finally:
        if temporary_path is not None:
            Path(temporary_path).unlink(missing_ok=True)


@app.get("/api/health")
def health() -> dict[str, str]:
    _, source = get_backend()
    return {"status": "ok", "data_source": source}


@app.post("/api/analyze")
async def analyze(
    flair: Annotated[UploadFile, File()],
    t1ce: Annotated[UploadFile | None, File()] = None,
) -> dict[str, object]:
    flair_suffix = _check_nifti_filename(flair.filename)
    flair_data = await _read_upload(flair)
    flair_volume, voxel_size = _load_nifti(flair_data, flair_suffix)
    t1ce_volume: np.ndarray | None = None
    if t1ce is not None:
        t1ce_suffix = _check_nifti_filename(t1ce.filename)
        t1ce_data = await _read_upload(t1ce)
        t1ce_volume, _ = _load_nifti(t1ce_data, t1ce_suffix)
        if t1ce_volume.shape != flair_volume.shape:
            raise HTTPException(
                status_code=422,
                detail="Les volumes FLAIR et T1ce doivent avoir la même dimension.",
            )
    backend, source = get_backend()
    try:
        mask, parameters = backend.segment(flair_volume, t1ce_volume)
        measurements = backend.measure(mask, voxel_size)
    except NotImplementedError as error:
        raise HTTPException(status_code=501, detail=str(error)) from error
    packed_mask = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
    finite_volume = np.nan_to_num(flair_volume, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = (float(value) for value in np.percentile(finite_volume, [1, 99]))
    if high > low:
        image_u8 = np.clip((finite_volume - low) * (255.0 / (high - low)), 0, 255)
    else:
        image_u8 = np.zeros_like(finite_volume)
    return {
        "data_source": source,
        "shape": list(mask.shape),
        "voxel_size_mm": list(voxel_size),
        "image": {
            "encoding": "zlib-u8",
            "data": base64.b64encode(
                zlib.compress(image_u8.astype(np.uint8).tobytes())
            ).decode("ascii"),
        },
        "mask": {
            "encoding": "zlib-packbits-little",
            "data": base64.b64encode(zlib.compress(packed_mask)).decode("ascii"),
        },
        "measurements": measurements,
        "parameters": parameters,
    }


@app.post("/api/growth")
def growth(request: GrowthRequest) -> dict[str, object]:
    backend, source = get_backend()
    try:
        result = backend.fit_growth(
            [point.model_dump() for point in request.volumes_by_day],
            request.model,
        )
    except NotImplementedError as error:
        raise HTTPException(status_code=501, detail=str(error)) from error
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"data_source": source, "model": request.model, **result}


@app.post("/api/radiotherapy")
def radiotherapy(request: RadiotherapyRequest) -> dict[str, object]:
    backend, source = get_backend()
    try:
        trajectories = backend.simulate_rt(
            {
                "volume_cm3": request.volume_cm3,
                "r": request.r,
                "K": request.K,
            },
            request.alpha,
            request.beta,
            request.bed_max,
            request.n_fractions,
        )
    except NotImplementedError as error:
        raise HTTPException(status_code=501, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"data_source": source, "simulation": True, **trajectories}


@app.get("/api/validation")
def validation() -> dict[str, object]:
    _, source = get_backend()
    if not RESULTS_FILE.is_file():
        return {
            "data_source": source,
            "available": False,
            "message": "Fichier de validation indisponible.",
            "count": 0,
            "dice": [],
            "volumes": [],
            "mean": None,
            "median": None,
            "std": None,
        }
    try:
        with RESULTS_FILE.open(encoding="utf-8-sig", newline="") as results_file:
            reader = csv.DictReader(results_file)
            columns = set(reader.fieldnames or [])
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise HTTPException(
            status_code=500,
            detail="Le fichier de validation ne peut pas être lu.",
        ) from error
    required_columns = {"cas", "dice", "vol", "vol_gt"}
    missing_columns = required_columns - columns
    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail=(
                "Colonnes manquantes dans le fichier de validation : "
                f"{', '.join(sorted(missing_columns))}."
            ),
        )
    volumes: list[dict[str, str | float]] = []
    dice_values: list[float] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            case_id = row["cas"]
            dice_value = float(row["dice"])
            volume = float(row["vol"])
            expert_volume = float(row["vol_gt"])
        except (KeyError, TypeError, ValueError) as error:
            raise HTTPException(
                status_code=500,
                detail=f"Ligne de validation invalide : {line_number}.",
            ) from error
        if not case_id or not case_id.strip():
            raise HTTPException(
                status_code=500,
                detail=f"Ligne de validation invalide : {line_number}.",
            )
        if (
            not all(np.isfinite([dice_value, volume, expert_volume]))
            or not 0.0 <= dice_value <= 1.0
            or volume < 0.0
            or expert_volume < 0.0
        ):
            raise HTTPException(
                status_code=500,
                detail=f"Ligne de validation invalide : {line_number}.",
            )
        dice_values.append(dice_value)
        volumes.append(
            {
                "cas": case_id,
                "volume_cm3": volume,
                "volume_expert_cm3": expert_volume,
            }
        )
    return {
        "data_source": source,
        "available": True,
        "count": len(dice_values),
        "dice": dice_values,
        "volumes": volumes,
        "mean": statistics.fmean(dice_values) if dice_values else None,
        "median": statistics.median(dice_values) if dice_values else None,
        "std": statistics.stdev(dice_values) if len(dice_values) > 1 else None,
    }
