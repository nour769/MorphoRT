import { useState } from "react";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { simulateRadiotherapy } from "../api";
import { Banner, Button, Card, ChartFrame, Slider, Stat } from "../components";
import { errorMessage, formatMetric } from "../format";
import type {
  DataSource,
  GrowthParameters,
  RadiotherapyResult,
} from "../types";

interface RadiotherapyPageProps {
  growthParameters: GrowthParameters | null;
  onDataSource: (source: DataSource) => void;
}

export function RadiotherapyPage({
  growthParameters,
  onDataSource,
}: RadiotherapyPageProps) {
  const [alpha, setAlpha] = useState(0.05);
  const [beta, setBeta] = useState(0.01);
  const [bedMax, setBedMax] = useState(30);
  const [fractions, setFractions] = useState(10);
  const [result, setResult] = useState<RadiotherapyResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSimulation = async () => {
    if (!growthParameters) {
      setError(
        "Ajustez d’abord un modèle de croissance pour définir les paramètres initiaux.",
      );
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const simulation = await simulateRadiotherapy({
        ...growthParameters,
        alpha,
        beta,
        bed_max: bedMax,
        n_fractions: fractions,
      });
      setResult(simulation);
      onDataSource(simulation.data_source);
    } catch (reason: unknown) {
      setError(errorMessage(reason));
    } finally {
      setLoading(false);
    }
  };

  const chartColor = "var(--color-chart-accent)";
  const axisColor = "var(--color-secondary)";
  const chartData =
    result?.fixed.map((fixedPoint, index) => ({
      fraction: fixedPoint.fraction,
      fixed: fixedPoint.volume_cm3,
      adaptive: result.adaptive[index]?.volume_cm3,
    })) ?? [];
  const fixedFinal = result ? result.fixed[result.fixed.length - 1] : undefined;
  const adaptiveFinal = result
    ? result.adaptive[result.adaptive.length - 1]
    : undefined;

  return (
    <section className="stack">
      <div className="page-header">
        <div>
          <h1>Radiothérapie</h1>
          <p>
            Comparaison de trajectoires simulées à partir du dernier ajustement
            de croissance.
          </p>
        </div>
        <span className="status-pill">Simulation uniquement</span>
      </div>
      <Banner tone="warning">
        Cette comparaison est une simulation exploratoire. Les paramètres ne
        sont pas des recommandations ni des prescriptions cliniques.
      </Banner>
      {!growthParameters && (
        <Banner>
          Aucun ajustement de croissance disponible. Utilisez la page Croissance
          avant de lancer une simulation.
        </Banner>
      )}
      <Card className="stack">
        <div className="grid-2">
          <Slider
            label="Alpha (simulation)"
            value={alpha}
            min={0.01}
            max={0.5}
            step={0.01}
            onChange={setAlpha}
          />
          <Slider
            label="Beta (simulation)"
            value={beta}
            min={0.001}
            max={0.1}
            step={0.001}
            onChange={setBeta}
          />
          <Slider
            label="Limite BED (simulation)"
            value={bedMax}
            min={1}
            max={120}
            step={1}
            unit="Gy"
            onChange={setBedMax}
          />
          <Slider
            label="Nombre de fractions"
            value={fractions}
            min={1}
            max={30}
            step={1}
            onChange={setFractions}
          />
        </div>
        <div className="simulation-context">
          {growthParameters ? (
            <span>
              Paramètres du modèle : V₀{" "}
              {formatMetric(growthParameters.volume_cm3, 3)} cm³ · r{" "}
              {formatMetric(growthParameters.r, 5)} · K{" "}
              {formatMetric(growthParameters.K, 3)} cm³
            </span>
          ) : (
            <span>Les paramètres V₀, r et K seront issus de l’ajustement.</span>
          )}
        </div>
        <div>
          <Button
            onClick={handleSimulation}
            disabled={loading || !growthParameters}
          >
            {loading ? "Simulation en cours…" : "Lancer la simulation"}
          </Button>
        </div>
      </Card>
      {error && <Banner tone="error">{error}</Banner>}
      {result && fixedFinal && adaptiveFinal && (
        <>
          <div className="grid-2">
            <Stat
              label="Volume final · stratégie fixe"
              value={formatMetric(fixedFinal.volume_cm3, 4)}
              unit="cm³"
              source={`Simulation API · ${result.data_source}`}
            />
            <Stat
              label="Volume final · stratégie adaptative"
              value={formatMetric(adaptiveFinal.volume_cm3, 4)}
              unit="cm³"
              source={`Simulation API · ${result.data_source}`}
            />
          </div>
          <ChartFrame
            title="Trajectoires simulées"
            caption="Volume simulé par fraction (cm³)"
            source={`Simulation API · ${result.data_source}`}
          >
            <ResponsiveContainer width="100%" height={320}>
              <ComposedChart data={chartData} accessibilityLayer>
                <CartesianGrid strokeDasharray="3 3" stroke={axisColor} />
                <XAxis
                  dataKey="fraction"
                  tick={{ fill: axisColor, fontSize: 11 }}
                  label={{
                    value: "Fraction",
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
                <Line
                  type="monotone"
                  dataKey="fixed"
                  stroke={axisColor}
                  strokeWidth={2}
                  name="Fixe · simulation"
                  isAnimationActive={false}
                />
                <Line
                  type="monotone"
                  dataKey="adaptive"
                  stroke={chartColor}
                  strokeWidth={2}
                  name="Adaptative · simulation"
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartFrame>
        </>
      )}
    </section>
  );
}
