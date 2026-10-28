import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { getValidation } from "../api";
import { Banner, Card, ChartFrame, Stat } from "../components";
import { errorMessage, formatMetric } from "../format";
import type { DataSource, ValidationResult } from "../types";

interface ValidationPageProps {
  onDataSource: (source: DataSource) => void;
}

function makeHistogram(values: number[]) {
  return Array.from({ length: 10 }, (_, index) => {
    const lower = index / 10;
    const upper = (index + 1) / 10;
    return {
      interval: `${formatMetric(lower, 1)}–${formatMetric(upper, 1)}`,
      count: values.filter((value) =>
        index === 9
          ? value >= lower && value <= upper
          : value >= lower && value < upper,
      ).length,
    };
  });
}

export function ValidationPage({ onDataSource }: ValidationPageProps) {
  const [result, setResult] = useState<ValidationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getValidation()
      .then((data) => {
        setResult(data);
        onDataSource(data.data_source);
      })
      .catch((reason: unknown) => setError(errorMessage(reason)))
      .finally(() => setLoading(false));
  }, [onDataSource]);

  const sourceLabel =
    result?.data_source === "mock" ? "Backend mock" : "Backend réel";
  const histogram = makeHistogram(result?.dice ?? []);
  const chartColor = "var(--color-chart-accent)";
  const axisColor = "var(--color-secondary)";

  return (
    <section className="stack">
      <div className="page-header">
        <div>
          <h1>Validation</h1>
          <p>
            Indicateurs calculés depuis{" "}
            <span className="mono">backend/data/results.csv</span>.
          </p>
        </div>
      </div>
      {error && <Banner tone="error">{error}</Banner>}
      {loading && <Banner>Chargement des résultats de validation…</Banner>}
      {result && !result.available && (
        <Banner tone="warning">{result.message}</Banner>
      )}
      {result?.available && result.count === 0 && (
        <Banner tone="warning">
          Le fichier existe, mais ne contient aucune ligne de validation
          exploitable.
        </Banner>
      )}
      {result?.available && result.count > 0 && (
        <>
          <div className="grid-3">
            <Stat
              label="Cas disponibles"
              value={formatMetric(result.count, 0)}
              source={sourceLabel}
            />
            <Stat
              label="Dice moyen"
              value={result.mean === null ? "—" : formatMetric(result.mean, 4)}
              source={sourceLabel}
            />
            <Stat
              label="Dice médian"
              value={
                result.median === null ? "—" : formatMetric(result.median, 4)
              }
              source={sourceLabel}
            />
            <Stat
              label="Écart-type du Dice"
              value={result.std === null ? "—" : formatMetric(result.std, 4)}
              source={sourceLabel}
            />
          </div>
          <div className="grid-2">
            <ChartFrame
              title="Distribution du Dice"
              caption="Répartition des valeurs du fichier de validation"
              source={sourceLabel}
            >
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={histogram} accessibilityLayer>
                  <CartesianGrid strokeDasharray="3 3" stroke={axisColor} />
                  <XAxis
                    dataKey="interval"
                    tick={{ fill: axisColor, fontSize: 11 }}
                    label={{
                      value: "Dice",
                      position: "insideBottom",
                      offset: -5,
                      fill: axisColor,
                    }}
                  />
                  <YAxis
                    allowDecimals={false}
                    tick={{ fill: axisColor, fontSize: 11 }}
                    label={{
                      value: "Nombre de cas",
                      angle: -90,
                      position: "insideLeft",
                      fill: axisColor,
                    }}
                  />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" name="Cas">
                    {histogram.map((entry) => (
                      <Cell key={entry.interval} fill={chartColor} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartFrame>
            <ChartFrame
              title="Volumes segmentés et experts"
              caption="Comparaison par cas, en cm³"
              source={sourceLabel}
            >
              <ResponsiveContainer width="100%" height={280}>
                <ScatterChart accessibilityLayer>
                  <CartesianGrid strokeDasharray="3 3" stroke={axisColor} />
                  <XAxis
                    type="number"
                    dataKey="volume_expert_cm3"
                    name="Volume expert"
                    unit=" cm³"
                    tick={{ fill: axisColor, fontSize: 11 }}
                    label={{
                      value: "Volume expert (cm³)",
                      position: "insideBottom",
                      offset: -5,
                      fill: axisColor,
                    }}
                  />
                  <YAxis
                    type="number"
                    dataKey="volume_cm3"
                    name="Volume segmenté"
                    unit=" cm³"
                    tick={{ fill: axisColor, fontSize: 11 }}
                    label={{
                      value: "Volume segmenté (cm³)",
                      angle: -90,
                      position: "insideLeft",
                      fill: axisColor,
                    }}
                  />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                  <Legend />
                  <Scatter name="Cas" data={result.volumes} fill={chartColor} />
                </ScatterChart>
              </ResponsiveContainer>
            </ChartFrame>
          </div>
          <Card>
            <h2>Source des données</h2>
            <p>
              {result.count} lignes valides lues depuis le fichier CSV fourni.
              Aucun résultat de validation n’est généré si le fichier est
              absent.
            </p>
          </Card>
        </>
      )}
      {!loading && !error && !result && (
        <Card>
          <h2>Résultats indisponibles</h2>
          <p>La réponse de validation n’a pas pu être chargée.</p>
        </Card>
      )}
    </section>
  );
}
