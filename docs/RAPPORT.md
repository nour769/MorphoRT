# Rapport de réalisation MorphoRT

## Objectif

MorphoRT est un prototype Streamlit local qui illustre un pipeline allant d'un
volume IRM à des opérations morphologiques, des mesures descriptives, des
ajustements de croissance et une optimisation radiobiologique simplifiée.

## Étapes disponibles

1. Import en mémoire de volumes NIfTI `.nii` / `.nii.gz` et affichage axial,
   coronal et sagittal ; mode démo avec quatre modalités et masque synthétiques.
2. Normalisation robuste, seuil d'Otsu fait main, ouverture, fermeture,
   remplissage des trous, reconstruction, composante connexe et watershed.
3. Volume, surface voxelisée, sphéricité, compacité et Dice du masque démo.
4. Ajustement des modèles de Gompertz et logistique, à partir de trois examens
   au minimum.
5. Comparaison illustrative d'un plan fixe et d'un scénario adaptatif avec
   modèle linéaire-quadratique et contrainte BED, optimisés par SLSQP.

## Architecture

Les calculs sont isolés dans `morphort/`, les vues sont dans `pages/`, les tests
dans `tests/`. scikit-image est réservé aux vérifications de tests ; les
opérations morphologiques du produit sont implémentées avec NumPy.

## Limites et sécurité

Le prototype n'est ni un dispositif médical validé, ni un système de diagnostic
ou de planification de traitement. Aucune sortie ne doit servir à choisir une
dose, une cible ou une conduite clinique. Le modèle de segmentation est fondé
sur un seuil global et n'est pas robuste à la diversité des examens. Les
mesures de surface dépendent de la discrétisation des voxels. Les ajustements
de croissance sont sensibles au nombre, à la qualité et à l'espacement des
examens. Le modèle LQ et la contrainte BED ne représentent pas toute la
radiobiologie, l'anatomie, les contraintes d'organes ou les pratiques de soins.

Toutes les images de démonstration sont artificielles. Les scans chargés sont
traités localement en mémoire et ne sont pas enregistrés par l'application ;
les règles de sécurité et de confidentialité de l'environnement restent
applicables.
