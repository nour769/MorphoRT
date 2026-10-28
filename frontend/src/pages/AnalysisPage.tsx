import { useEffect, useRef, useState } from "react";

import { analyzeVolume } from "../api";
import { Banner, Card, DataTable, Dropzone, Slider, Tabs } from "../components";
import { decompressBase64, errorMessage, formatMetric } from "../format";
import type { AnalysisResult, DataSource, GrowthParameters } from "../types";

type Orientation = "axial" | "coronal" | "sagittal";

const orientations: Array<{ id: Orientation; label: string }> = [
  { id: "axial", label: "Axial" },
  { id: "coronal", label: "Coronal" },
  { id: "sagittal", label: "Sagittal" },
];

interface AnalysisPageProps {
  onDataSource: (source: DataSource) => void;
  onGrowthParameters: (parameters: GrowthParameters | null) => void;
}

function voxelIndex(
  x: number,
  y: number,
  z: number,
  shape: [number, number, number],
) {
  return (x * shape[1] + y) * shape[2] + z;
}

export function AnalysisPage({
  onDataSource,
  onGrowthParameters,
}: AnalysisPageProps) {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [image, setImage] = useState<Uint8Array | null>(null);
  const [mask, setMask] = useState<Uint8Array | null>(null);
  const [orientation, setOrientation] = useState<Orientation>("axial");
  const [slice, setSlice] = useState(0);
  const [opacity, setOpacity] = useState(0.45);
  const [flairFile, setFlairFile] = useState<File | null>(null);
  const [t1ceFile, setT1ceFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const validateFile = (file: File): string | null => {
    const name = file.name.toLowerCase();
    if (!name.endsWith(".nii") && !name.endsWith(".nii.gz")) {
      return "Format non pris en charge. Choisissez un fichier .nii ou .nii.gz.";
    }
    if (file.size > 300 * 1024 * 1024) {
      return "Le fichier dépasse la taille maximale de 300 Mo.";
    }
    return null;
  };

  const chooseFlair = (file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }
    setFlairFile(file);
    setError(null);
  };

  const chooseT1ce = (file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }
    setT1ceFile(file);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!flairFile) {
      setError("Sélectionnez un volume FLAIR avant de lancer l’analyse.");
      return;
    }
    if (t1ceFile && t1ceFile.size > 300 * 1024 * 1024) {
      setError("Le volume T1ce dépasse la taille maximale de 300 Mo.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const analyzed = await analyzeVolume(flairFile, t1ceFile ?? undefined);
      const [imageBytes, packedMask] = await Promise.all([
        decompressBase64(analyzed.image.data),
        decompressBase64(analyzed.mask.data),
      ]);
      const voxelCount = analyzed.shape.reduce(
        (total, size) => total * size,
        1,
      );
      if (
        imageBytes.length !== voxelCount ||
        packedMask.length * 8 < voxelCount
      ) {
        throw new Error("Le volume compressé reçu de l’API est incomplet.");
      }
      const maskBytes = new Uint8Array(voxelCount);
      for (let index = 0; index < voxelCount; index += 1) {
        maskBytes[index] = (packedMask[index >> 3] >> (index & 7)) & 1;
      }
      setResult(analyzed);
      setImage(imageBytes);
      setMask(maskBytes);
      setOrientation("axial");
      setSlice(Math.floor(analyzed.shape[2] / 2));
      onDataSource(analyzed.data_source);
      onGrowthParameters(null);
    } catch (reason: unknown) {
      setError(errorMessage(reason));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !result || !image || !mask) return;
    const [xSize, ySize, zSize] = result.shape;
    const plane =
      orientation === "axial"
        ? { width: xSize, height: ySize }
        : orientation === "coronal"
          ? { width: xSize, height: zSize }
          : { width: ySize, height: zSize };
    canvas.width = plane.width;
    canvas.height = plane.height;
    const context = canvas.getContext("2d");
    if (!context) {
      setError("Le navigateur ne peut pas afficher le visualiseur de coupes.");
      return;
    }
    const imageData = context.createImageData(plane.width, plane.height);
    const accentHex = getComputedStyle(document.documentElement)
      .getPropertyValue("--color-accent")
      .trim();
    const rgb = /^#([0-9a-f]{6})$/i.exec(accentHex)?.[1];
    if (!rgb) {
      setError("La couleur de superposition du visualiseur est invalide.");
      return;
    }
    const overlay = [
      Number.parseInt(rgb.slice(0, 2), 16),
      Number.parseInt(rgb.slice(2, 4), 16),
      Number.parseInt(rgb.slice(4, 6), 16),
    ];
    for (let row = 0; row < plane.height; row += 1) {
      for (let column = 0; column < plane.width; column += 1) {
        let x = column;
        let y = row;
        let z = slice;
        if (orientation === "coronal") {
          y = slice;
          z = row;
        } else if (orientation === "sagittal") {
          x = slice;
          y = column;
          z = row;
        }
        const sourceIndex = voxelIndex(x, y, z, result.shape);
        const canvasRow = plane.height - row - 1;
        const targetIndex = (canvasRow * plane.width + column) * 4;
        const gray = image[sourceIndex];
        const isTumor = mask[sourceIndex] === 1;
        imageData.data[targetIndex] = isTumor
          ? Math.round(gray * (1 - opacity) + overlay[0] * opacity)
          : gray;
        imageData.data[targetIndex + 1] = isTumor
          ? Math.round(gray * (1 - opacity) + overlay[1] * opacity)
          : gray;
        imageData.data[targetIndex + 2] = isTumor
          ? Math.round(gray * (1 - opacity) + overlay[2] * opacity)
          : gray;
        imageData.data[targetIndex + 3] = 255;
      }
    }
    context.putImageData(imageData, 0, 0);
  }, [image, mask, opacity, orientation, result, slice]);

  const sliceCount = result
    ? orientation === "axial"
      ? result.shape[2]
      : orientation === "coronal"
        ? result.shape[1]
        : result.shape[0]
    : 0;
  const measurementRows = result
    ? [
        {
          label: "Volume",
          value: formatMetric(result.measurements.volume_cm3, 3),
          unit: "cm³",
        },
        {
          label: "Surface",
          value: formatMetric(result.measurements.surface_cm2, 3),
          unit: "cm²",
        },
        {
          label: "Sphéricité",
          value: formatMetric(result.measurements.sphericity, 3),
          unit: "—",
        },
      ]
    : [];

  return (
    <section className="stack">
      <div className="page-header">
        <div>
          <h1>Analyse</h1>
          <p>
            Importez un volume 3D NIfTI. Les fichiers sont traités
            temporairement et ne sont pas conservés.
          </p>
        </div>
      </div>
      <div className="grid-2">
        <Dropzone
          onFile={chooseFlair}
          disabled={loading}
          title="Volume FLAIR requis"
          inputLabel="Sélectionner un volume FLAIR"
          hint={
            flairFile?.name ?? "Fichier NIfTI .nii ou .nii.gz · max. 300 Mo"
          }
        />
        <Dropzone
          onFile={chooseT1ce}
          disabled={loading}
          title="Volume T1ce facultatif"
          inputLabel="Sélectionner un volume T1ce facultatif"
          hint={t1ceFile?.name ?? "Optionnel · mêmes dimensions que FLAIR"}
        />
      </div>
      {error && <Banner tone="error">{error}</Banner>}
      <div className="analysis-actions">
        <button
          className="button button-primary"
          type="button"
          onClick={handleAnalyze}
          disabled={loading || !flairFile}
        >
          {loading ? "Analyse en cours…" : "Lancer l’analyse"}
        </button>
        {result && (
          <span className="source-inline">
            Source : <strong>{result.data_source}</strong>
          </span>
        )}
      </div>
      {result && (
        <div className="analysis-layout">
          <Card className="slice-card">
            <div className="section-heading">
              <div>
                <h2>Visualiseur de coupes</h2>
                <p>
                  Dimensions : {result.shape.join(" × ")} voxels · voxel :{" "}
                  {result.voxel_size_mm
                    .map((value) => formatMetric(value, 2))
                    .join(" × ")}{" "}
                  mm
                </p>
              </div>
              <Tabs
                items={orientations}
                selected={orientation}
                onChange={(selected) => {
                  setOrientation(selected);
                  setSlice(
                    selected === "axial"
                      ? Math.floor(result.shape[2] / 2)
                      : selected === "coronal"
                        ? Math.floor(result.shape[1] / 2)
                        : Math.floor(result.shape[0] / 2),
                  );
                }}
              />
            </div>
            <div className="slice-canvas-wrap">
              <canvas
                ref={canvasRef}
                className="slice-canvas"
                role="img"
                aria-label={`${orientations.find((item) => item.id === orientation)?.label} slice ${slice + 1} with segmentation overlay`}
              />
            </div>
            <div className="slice-controls">
              <label className="slice-position" htmlFor="slice-position">
                Coupe{" "}
                <span className="mono">
                  {slice + 1} / {sliceCount}
                </span>
              </label>
              <input
                id="slice-position"
                aria-label="Position de la coupe"
                type="range"
                min={0}
                max={Math.max(sliceCount - 1, 0)}
                value={slice}
                onChange={(event) => setSlice(Number(event.target.value))}
              />
              <Slider
                label="Opacité de la superposition"
                value={Math.round(opacity * 100)}
                min={0}
                max={100}
                step={1}
                unit="%"
                onChange={(value) => setOpacity(value / 100)}
              />
            </div>
          </Card>
          <Card>
            <h2>Mesures morphologiques</h2>
            <p>Valeurs calculées par le backend pour ce volume.</p>
            <DataTable
              columns={[
                { key: "label", label: "Mesure", render: (row) => row.label },
                {
                  key: "value",
                  label: "Valeur",
                  render: (row) => <span className="mono">{row.value}</span>,
                },
                { key: "unit", label: "Unité", render: (row) => row.unit },
              ]}
              rows={measurementRows}
              emptyMessage="Aucune mesure disponible."
            />
          </Card>
        </div>
      )}
    </section>
  );
}
