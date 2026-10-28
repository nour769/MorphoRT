import { Card } from "../components";
import type { PageKey } from "../types";

const stages: Array<{ number: string; title: string; page: PageKey }> = [
  { number: "01", title: "IRM", page: "analysis" },
  { number: "02", title: "Segmentation", page: "analysis" },
  { number: "03", title: "Mesures", page: "analysis" },
  { number: "04", title: "Croissance", page: "growth" },
  { number: "05", title: "Radiothérapie", page: "radiotherapy" },
];

interface HomePageProps {
  onNavigate: (page: PageKey) => void;
}

export function HomePage({ onNavigate }: HomePageProps) {
  return (
    <section className="stack">
      <div className="page-header">
        <div>
          <h1>MorphoRT Platform</h1>
          <p>
            Prototype de recherche pour l’analyse morphologique et la simulation
            de trajectoires tumorales à partir d’une IRM cérébrale 3D.
          </p>
        </div>
      </div>
      <Card>
        <div className="section-heading">
          <div>
            <span className="eyebrow">Pipeline de recherche</span>
            <h2>De l’image à la simulation</h2>
          </div>
          <span className="status-pill">Une image à la fois</span>
        </div>
        <div className="pipeline">
          {stages.map((stage, index) => (
            <button
              type="button"
              className="pipeline-step"
              key={stage.number}
              onClick={() => onNavigate(stage.page)}
            >
              <span className="pipeline-number">{stage.number}</span>
              <strong>{stage.title}</strong>
              {index < stages.length - 1 && (
                <span className="pipeline-connector" aria-hidden="true" />
              )}
            </button>
          ))}
        </div>
      </Card>
      <div className="grid-2">
        <Card>
          <h2>À propos du prototype</h2>
          <p>
            Explorez une segmentation synthétique, des mesures morphologiques,
            un ajustement de modèle de croissance et une comparaison de
            radiothérapie simulée.
          </p>
        </Card>
        <Card>
          <h2>Cadre d’utilisation</h2>
          <p>
            Les sorties sont expérimentales. Aucune mesure, simulation ou valeur
            affichée ne doit être interprétée comme une recommandation médicale.
          </p>
        </Card>
      </div>
    </section>
  );
}
