import { useMemo, useState } from "react";

import {
  applyMorphology,
  SAMPLE_IMAGE,
  type MorphologyOperation,
} from "../morphology";

const operations: Array<{
  id: MorphologyOperation;
  label: string;
  summary: string;
}> = [
  {
    id: "dilation",
    label: "Dilatation",
    summary: "Agrandit les zones blanches.",
  },
  {
    id: "erosion",
    label: "Érosion",
    summary: "Rétrécit les zones blanches.",
  },
  {
    id: "opening",
    label: "Ouverture",
    summary: "Enlève les petits points isolés.",
  },
  {
    id: "closing",
    label: "Fermeture",
    summary: "Comble les petits trous.",
  },
];

const operationLabels: Record<MorphologyOperation, string> = {
  dilation: "Dilatation",
  erosion: "Érosion",
  opening: "Ouverture",
  closing: "Fermeture",
};

function countPixels(image: number[][]): number {
  return image.reduce(
    (total, row) => total + row.reduce((sum, pixel) => sum + pixel, 0),
    0,
  );
}

function ImageGrid({ image, label }: { image: number[][]; label: string }) {
  return (
    <div
      className="image-grid"
      role="img"
      aria-label={`${label}, grille de ${image.length} par ${image[0].length} pixels`}
      style={{ gridTemplateColumns: `repeat(${image[0].length}, 1fr)` }}
    >
      {image.flatMap((row, y) =>
        row.map((pixel, x) => (
          <span
            aria-hidden="true"
            className={pixel ? "image-pixel image-pixel-on" : "image-pixel"}
            key={`${y}-${x}`}
          />
        )),
      )}
    </div>
  );
}

export function MorphologyPage() {
  const [operation, setOperation] =
    useState<MorphologyOperation>("opening");
  const [size, setSize] = useState(3);
  const result = useMemo(
    () => applyMorphology(SAMPLE_IMAGE, operation, size),
    [operation, size],
  );
  const operationLabel = operationLabels[operation];

  return (
    <main className="app-shell">
      <header className="page-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            M
          </span>
          <span className="eyebrow">Atelier interactif</span>
        </div>
        <h1>La morphologie, simplement</h1>
        <p>
          Choisissez une opération et observez comment elle transforme une
          image en noir et blanc.
        </p>
      </header>

      <section className="workspace" aria-labelledby="operation-title">
        <div className="controls">
          <div>
            <h2 id="operation-title">1. Choisir une opération</h2>
            <div className="operation-list">
              {operations.map((item) => (
                <button
                  aria-pressed={operation === item.id}
                  className={`operation-button ${operation === item.id ? "operation-button-selected" : ""}`}
                  key={item.id}
                  onClick={() => setOperation(item.id)}
                  type="button"
                >
                  <strong>{item.label}</strong>
                  <span>{item.summary}</span>
                </button>
              ))}
            </div>
          </div>
          <label className="size-control" htmlFor="neighborhood-size">
            <span>
              <strong>2. Taille du voisinage</strong>
              <span>Petit carré qui balaie l’image</span>
            </span>
            <select
              id="neighborhood-size"
              onChange={(event) => setSize(Number(event.target.value))}
              value={size}
            >
              <option value={3}>3 × 3 — effet léger</option>
              <option value={5}>5 × 5 — effet plus fort</option>
            </select>
          </label>
          <div className="neighborhood-preview" aria-label="Voisinage carré">
            <span>Exemple de voisinage</span>
            <div
              className="neighborhood-grid"
              style={{ gridTemplateColumns: `repeat(${size}, 1fr)` }}
            >
              {Array.from({ length: size * size }, (_, index) => (
                <span
                  aria-hidden="true"
                  className="neighborhood-pixel"
                  key={index}
                />
              ))}
            </div>
          </div>
        </div>

        <div className="results">
          <h2>3. Observer le résultat</h2>
          <div className="image-pair">
            <figure className="image-card">
              <figcaption>
                <strong>Image de départ</strong>
                <span>{countPixels(SAMPLE_IMAGE)} pixels blancs</span>
              </figcaption>
              <ImageGrid image={SAMPLE_IMAGE} label="Image de départ" />
            </figure>
            <figure className="image-card">
              <figcaption>
                <strong>Après {operationLabel.toLowerCase()}</strong>
                <span>{countPixels(result)} pixels blancs</span>
              </figcaption>
              <ImageGrid image={result} label="Résultat de l’opération" />
            </figure>
          </div>
        </div>
      </section>

      <section className="explanation" aria-live="polite">
        <span className="eyebrow">À retenir</span>
        <h2>{operationLabel}, en clair</h2>
        <p>
          {operation === "dilation" &&
            "Un pixel blanc devient blanc si un voisin est blanc : les formes s’élargissent."}
          {operation === "erosion" &&
            "Un pixel reste blanc seulement si tous ses voisins sont blancs : les formes rétrécissent."}
          {operation === "opening" &&
            "On rétrécit d’abord les formes, puis on les agrandit : les petits points isolés disparaissent."}
          {operation === "closing" &&
            "On agrandit d’abord les formes, puis on les rétrécit : les petits trous se remplissent."}
        </p>
        <p className="boundary-note">
          Ici, le voisinage est un carré. En dehors de l’image, les pixels sont
          considérés comme noirs.
        </p>
      </section>
      <footer>Outil pédagogique — les carrés représentent des pixels.</footer>
    </main>
  );
}
