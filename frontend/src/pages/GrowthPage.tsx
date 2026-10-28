import { useMemo, useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { fitGrowth } from "../api";
import {
  Banner,
  Button,
  Card,
  ChartFrame,
  DataTable,
  Stat,
} from "../components";
import { errorMessage, formatMetric } from "../format";
import type {
  DataSource,
  GrowthParameters,
  GrowthPointInput,
  GrowthResult,
} from "../types";

interface GrowthPageProps {
  onDataSource: (source: DataSource) => void;
  onGrowthParameters: (parameters: GrowthParameters) => void;
}

interface GrowthRow {
  id: number;
  day: string;
  volume: string;
}

const emptyRow = (id: number): GrowthRow => ({ id, day: "", volume: "" });

export function GrowthPage({
  onDataSource,
  onGrowthParameters,
}: GrowthPageProps) {
  const [rows, setRows] = useState<GrowthRow[]>([emptyRow(1)]);
  const [nextId, setNextId] = useState(2);
  const [model, setModel] = useState<"gompertz" | "logistic">("gompertz");
  const [result, setResult] = useState<GrowthResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const points = useMemo(() => {
    const parsed: GrowthPointInput[] = [];
    for (const row of rows) {
      if (!row.day.trim() && !row.volume.trim()) continue;
      if (!row.day.trim() || !row.volume.trim()) return null;
      const day = Number(row.day);
      const volume_cm3 = Number(row.volume);
      if (
        !Number.isFinite(day) ||
        day < 0 ||
        !Number.isFinite(volume_cm3) ||
        volume_cm3 <= 0
      ) {
        return null;
      }
      parsed.push({ day, volume_cm3 });
    }
    return parsed;
  }, [rows]);

  const updateRow = (id: number, field: "day" | "volume", value: string) => {
    setRows((current) =>
      current.map((row) => (row.id === id ? { ...row, [field]: value } : row)),
    );
  };

  const addRow = () => {
    setRows((current) => [...current, emptyRow(nextId)]);
    setNextId((value) => value + 1);
  };

  const importCsv = async (file: File | undefined) => {
    if (!file) return;
    try {
      const contents = await file.text();
      const delimiter = contents.split(/\r?\n/, 1)[0].includes(";") ? ";" : ",";
      const lines = contents
        .split(/\r?\n/)
        .filter((line) => line.trim().length > 0);
      if (lines.length < 2)
        throw new Error("Le CSV ne contient aucune ligne de mesure.");
      const header = lines[0]
        .split(delimiter)
        .map((value) => value.trim().replace(/^"|"$/g, "").toLowerCase());
      const dayIndex = header.findIndex((value) =>
        ["day", "jour"].includes(value),
      );
      const volumeIndex = header.findIndex((value) =>
        ["volume_cm3", "volume", "vol"].includes(value),
      );
      if (dayIndex < 0 || volumeIndex < 0) {
        throw new Error("Le CSV doit contenir les colonnes day et volume_cm3.");
      }
      const imported = lines.slice(1).map((line, index) => {
        const values = line
          .split(delimiter)
          .map((value) => value.trim().replace(/^"|"$/g, "").replace(",", "."));
        const day = Number(values[dayIndex]);
        const volume = Number(values[volumeIndex]);
        if (
          !Number.isFinite(day) ||
          day < 0 ||
          !Number.isFinite(volume) ||
          volume <= 0
        ) {
          throw new Error(`Valeur invalide à la ligne ${index + 2}.`);
        }
        return { id: nextId + index, day: String(day), volume: String(volume) };
      });
      setRows(imported);
      setNextId(nextId + imported.length);
      setResult(null);
      setError(null);
    } catch (reason: unknown) {
      setError(errorMessage(reason));
    }
  };

  const handleFit = async () => {
    if (points === null) {
      setError("Chaque ligne doit contenir un jour et un volume positif.");
      return;
    }
    if (points.length < 3) {
      setError(
        "Ajoutez au moins trois points de mesure pour ajuster le modèle.",
      );
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const fit = await fitGrowth(points, model);
      const firstPoint = [...points].sort(
        (left, right) => left.day - right.day,
      )[0];
      setResult(fit);
      onDataSource(fit.data_source);
      onGrowthParameters({
        volume_cm3: firstPoint.volume_cm3,
        r: fit.r,
        K: fit.K,
      });
    } catch (reason: unknown) {
      setError(errorMessage(reason));
    } finally {
      setLoading(false);
    }
  };

  const chartColor = "var(--color-chart-accent)";
  const axisColor = "var(--color-secondary)";
  const curve = result?.curve.map((point) => ({
    ...point,
    band_cm3: Math.max(0, point.upper_cm3 - point.lower_cm3),
  }));

  return (
    <section className="stack">
      <div className="page-header">
        <div>
          <h1>Croissance</h1>
          <p>
            Ajustement exploratoire d’un modèle à partir des volumes et jours
            fournis.
          </p>
        </div>
      </div>
      <Card className="stack">
        <div className="section-heading">
          <div>
            <h2>Mesures longitudinales</h2>
            <p>Jour relatif et volume mesuré en cm³.</p>
          </div>
          <label className="file-import">
            Importer un CSV
            <input
              type="file"
              accept=".csv,text/csv"
              aria-label="Importer des mesures au format CSV"
              onChange={(event) => {
                void importCsv(event.target.files?.[0]);
                event.currentTarget.value = "";
              }}
            />
          </label>
        </div>
        <DataTable
          columns={[
            {
              key: "day",
              label: "Jour depuis référence",
              render: (row) => (
                <input
                  className="form-control table-input"
                  aria-label={`Jour de la mesure ${row.id}`}
                  type="number"
                  min="0"
                  step="any"
                  value={row.day}
                  onChange={(event) =>
                    updateRow(row.id, "day", event.target.value)
                  }
                />
              ),
            },
            {
              key: "volume",
              label: "Volume (cm³)",
              render: (row) => (
                <input
                  className="form-control table-input"
                  aria-label={`Volume de la mesure ${row.id} en cm³`}
                  type="number"
                  min="0"
                  step="any"
                  value={row.volume}
                  onChange={(event) =>
                    updateRow(row.id, "volume", event.target.value)
                  }
                />
              ),
            },
            {
              key: "remove",
              label: "Action",
              render: (row) => (
                <button
                  className="text-button"
                  type="button"
                  aria-label={`Supprimer la mesure ${row.id}`}
                  onClick={() =>
                    setRows((current) =>
                      current.filter((item) => item.id !== row.id),
                    )
                  }
                >
                  Supprimer
                </button>
              ),
            },
          ]}
          rows={rows}
          emptyMessage="Ajoutez ou importez des mesures."
        />
        <div className="inline-actions">
          <Button variant="secondary" onClick={addRow}>
            Ajouter une mesure
          </Button>
          <label className="model-select">
            Modèle
            <select
              className="form-control"
              value={model}
              onChange={(event) =>
                setModel(
                  event.target.value === "logistic" ? "logistic" : "gompertz",
                )
              }
            >
              <option value="gompertz">Gompertz</option>
              <option value="logistic">Logistique</option>
            </select>
          </label>
          <Button onClick={handleFit} disabled={loading}>
            {loading ? "Ajustement…" : "Ajuster le modèle"}
          </Button>
        </div>
      </Card>
      {points !== null && points.length < 4 && (
        <Banner tone="warning">
          Moins de 4 points temporels : estimation peu fiable.
        </Banner>
      )}
      {error && <Banner tone="error">{error}</Banner>}
      {result && curve && (
        <>
          <div className="grid-2">
            <Stat
              label="Paramètre r"
              value={formatMetric(result.r, 5)}
              source={`API · ${result.data_source}`}
            />
            <Stat
              label="Intervalle de r"
              value={`${formatMetric(result.ci_r[0], 5)} – ${formatMetric(result.ci_r[1], 5)}`}
              source={`API · ${result.data_source}`}
            />
            <Stat
              label="Capacité K"
              value={formatMetric(result.K, 4)}
              unit="cm³"
              source={`API · ${result.data_source}`}
            />
            <Stat
              label="Intervalle de K"
              value={`${formatMetric(result.ci_K[0], 4)} – ${formatMetric(result.ci_K[1], 4)}`}
              unit="cm³"
              source={`API · ${result.data_source}`}
            />
          </div>
          <ChartFrame
            title={`Ajustement ${result.model === "gompertz" ? "de Gompertz" : "logistique"}`}
            caption="Bande d’incertitude et points de mesure fournis"
            source={`API · ${result.data_source}`}
          >
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={curve} accessibilityLayer>
                <CartesianGrid strokeDasharray="3 3" stroke={axisColor} />
                <XAxis
                  dataKey="day"
                  type="number"
                  domain={["dataMin", "dataMax"]}
                  tick={{ fill: axisColor, fontSize: 11 }}
                  label={{
                    value: "Jour",
                    position: "insideBottom",
                    offset: -5,
                    fill: axisColor,
                  }}
                />
                <YAxis
                  tick={{ fill: axisColor, fontSize: 11 }}
                  label={{
                    value: "Volume (cm³)",
                    angle: -90,
                    position: "insideLeft",
                    fill: axisColor,
                  }}
                />
                <Tooltip />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="lower_cm3"
                  stackId="confidence"
                  stroke="none"
                  fill="transparent"
                  name="Borne inférieure"
                  legendType="none"
                  isAnimationActive={false}
                />
                <Area
                  type="monotone"
                  dataKey="band_cm3"
                  stackId="confidence"
                  stroke="none"
                  fill={chartColor}
                  fillOpacity={0.18}
                  name="Bande d’incertitude"
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="volume_cm3"
                  stroke={chartColor}
                  strokeWidth={2}
                  dot={false}
                  name="Ajustement"
                  isAnimationActive={false}
                />
                <Scatter
                  data={points ?? []}
                  dataKey="volume_cm3"
                  fill={axisColor}
                  name="Mesures"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartFrame>
        </>
      )}
    </section>
  );
}
