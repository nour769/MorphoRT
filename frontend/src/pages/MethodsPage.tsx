import katex from "katex";

import { Card } from "../components";

const formulas = [
  {
    title: "Érosion",
    expression: String.raw`\varepsilon_B(A)=\{x\mid B_x\subseteq A\}`,
    explanation: "Réduit une région selon un élément structurant B.",
  },
  {
    title: "Dilatation",
    expression: String.raw`\delta_B(A)=\{x\mid \widehat{B}_x\cap A\neq\varnothing\}`,
    explanation: "Étend une région selon un élément structurant B.",
  },
  {
    title: "Ouverture et fermeture",
    expression: String.raw`A\circ B=(A\ominus B)\oplus B,\qquad A\bullet B=(A\oplus B)\ominus B`,
    explanation:
      "Opérations morphologiques composées d’érosion et de dilatation.",
  },
  {
    title: "Coefficient de Dice",
    expression: String.raw`\operatorname{Dice}(P,G)=\frac{2|P\cap G|}{|P|+|G|}`,
    explanation: "Mesure de recouvrement entre les régions P et G.",
  },
  {
    title: "Modèle de Gompertz",
    expression: String.raw`\frac{dV}{dt}=rV\ln\left(\frac{K}{V}\right)`,
    explanation: "Modèle de croissance avec capacité limite K.",
  },
  {
    title: "Modèle logistique",
    expression: String.raw`\frac{dV}{dt}=rV\left(1-\frac{V}{K}\right)`,
    explanation: "Modèle de croissance bornée par la capacité K.",
  },
  {
    title: "Survie linéaire-quadratique",
    expression: String.raw`S=\exp\left[-\left(\alpha d+\beta d^2\right)\right]`,
    explanation: "Expression utilisée dans la simulation de survie cellulaire.",
  },
  {
    title: "BED",
    expression: String.raw`\operatorname{BED}=nd\left(1+\frac{d}{\alpha/\beta}\right)`,
    explanation:
      "Dose biologiquement efficace selon le modèle linéaire-quadratique.",
  },
];

function Formula({ expression }: { expression: string }) {
  return (
    <div
      className="formula"
      aria-label={expression}
      dangerouslySetInnerHTML={{
        __html: katex.renderToString(expression, {
          displayMode: true,
          throwOnError: false,
        }),
      }}
    />
  );
}

export function MethodsPage() {
  return (
    <section className="stack">
      <div className="page-header">
        <div>
          <h1>Méthodes</h1>
          <p>Rappels mathématiques associés aux modules de recherche.</p>
        </div>
      </div>
      <div className="grid-2">
        {formulas.map((formula) => (
          <Card key={formula.title} className="formula-card">
            <h2>{formula.title}</h2>
            <Formula expression={formula.expression} />
            <p>{formula.explanation}</p>
          </Card>
        ))}
      </div>
      <Card>
        <h2>Limites</h2>
        <ul className="limits-list">
          <li>Prototype de recherche, non destiné à un usage clinique.</li>
          <li>Traitement d’un seul patient et d’un seul volume à la fois.</li>
          <li>
            Les comparaisons de radiothérapie sont des simulations uniquement.
          </li>
          <li>
            Le backend réel est un squelette de recherche non implémenté ; le
            backend de démonstration produit des données synthétiques.
          </li>
        </ul>
      </Card>
    </section>
  );
}
