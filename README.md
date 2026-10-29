# MorphoRT

Application locale en français pour parcourir, à des fins pédagogiques, les
étapes d'analyse d'un volume IRM cérébral : affichage, segmentation
morphologique, mesures, ajustement de croissance et simulation LQ.

> **Simulation uniquement, pas d'usage clinique.** Ce prototype n'est ni validé,
> ni destiné au diagnostic, à la prescription ou à la planification d'un
> traitement réel. Les données de démonstration sont artificielles.

## Installation et lancement en trois commandes

Depuis la racine du dépôt, sous Windows PowerShell avec Python 3.11 ou plus
récent :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Ouvrez ensuite l'adresse locale affichée par Streamlit, généralement
`http://localhost:8501`.

## Démonstration

Sur la page **Charger l'IRM**, sélectionnez **Démo synthétique**, puis cliquez
sur **Charger l'IRM**. La démonstration crée quatre volumes NIfTI simulés, un
masque de tumeur de référence et trois mesures de croissance artificielles.
Elle permet de parcourir les cinq pages sans données de patient.

Pour vos propres volumes, choisissez **Fichiers NIfTI** et ajoutez au moins une
séquence parmi T1, T1ce, T2 ou FLAIR. Les séquences fournies ensemble doivent
avoir la même taille, le même espacement voxel et le même repère. Les fichiers
sont lus en mémoire et ne sont pas conservés par l'application.

## Préparer un échantillon BraTS2020

BraTS2020 contient des IRM réelles et des masques experts, mais les données ne
sont pas incluses dans le dépôt. La préparation conserve les fichiers bruts hors
du projet et ne copie dans `data/brats/` qu'un échantillon représentatif de 20
patients d'entraînement disposant des cinq séquences.

1. Installez le client Kaggle si nécessaire :

   ```powershell
   python -m pip install kaggle
   ```

2. Configurez votre jeton Kaggle dans la variable d'environnement utilisateur
   `KAGGLE_API_TOKEN`, sans l'inscrire dans le projet :

   ```powershell
   $token = Read-Host "Jeton Kaggle"
   [Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", $token, "User")
   Remove-Variable token
   ```

   Ouvrez un nouveau terminal après cette configuration.

3. Téléchargez et décompressez l'archive dans votre dossier utilisateur hors du
   dépôt, puis copiez seulement 20 patients complets répartis dans la liste :

   ```powershell
   kaggle datasets download awsaf49/brats20-dataset-training-validation --path "$HOME\Downloads\brats_raw" --unzip
   python scripts\prepare_brats.py --n 20
   ```

   L'archive compressée est supprimée après extraction par `--unzip`. Les NIfTI
   bruts restent dans `~/Downloads/brats_raw/`; le sous-échantillon préparé va
   dans `data/brats/<patient>/`. Git ignore `data/`, `.kaggle/`, `*.nii`,
   `*.nii.gz` et `.env`. Vérifiez les conditions de licence et d'accès Kaggle
   avant d'utiliser ces données.

Sur la première page, choisissez **BraTS2020** puis un patient. Les labels
experts 1, 2 et 4 correspondent respectivement au noyau nécrotique, à l'œdème
et à la tumeur rehaussée ; la tumeur entière est l'ensemble des labels > 0.
BraTS2020 ne fournit qu'une acquisition par patient : toute courbe longitudinale
est explicitement simulée à partir du volume initial réel. Les résultats restent
pédagogiques et ne doivent pas guider des décisions cliniques.

### Évaluation globale de la tumeur entière

Après avoir préparé les patients dans `data/brats/`, relancez l'évaluation avec :

```powershell
python -m scripts.evaluate_brats
```

Le script segmente chaque patient complet avec les paramètres morphologiques
actuels et produit `results/brats_eval.csv` (une ligne par patient : Dice,
Jaccard et Hausdorff 95 en mm). Il affiche aussi, pour chaque métrique, la
moyenne, l'écart-type échantillonnal, le minimum et le maximum. Dans Streamlit,
la page **Résultats globaux** montre la distribution et les cas triés du Dice le
plus faible au plus élevé ; aucun score faible n'est supprimé ou corrigé. Les
résultats dépendent des patients réellement préparés et ne sont donc pas
présentés ici comme des valeurs attendues fixes. Le CSV ne contient que les
identifiants BraTS et les métriques calculées.

Les paramètres sont centralisés dans `morphort/config.py`. Le réglage et
l'évaluation historique n'utilisaient pas le même pipeline : l'évaluation de
base utilisait Otsu (ouverture 1 mm, fermeture 1,5 mm), tandis que les moyennes
0,7241 et 0,7748 provenaient du percentile 90 (ouverture et fermeture 1 mm),
sur deux groupes différents de 10 patients. La page 2 avait encore les anciens
réglages Otsu par défaut. L'évaluation, la page 2 et la page globale utilisent
maintenant le triplet retenu par le réglage ; le CSV enregistre ses paramètres
et la page globale signale un CSV calculé avec une autre configuration.

### Analyse des cinq échecs les plus marqués

Après l'évaluation et le réglage, lancez :

```powershell
python -m scripts.analyze_failures
```

Le script compare l'ouverture standard à l'ouverture par reconstruction
uniquement sur les patients du groupe de réglage, puis garde la variante ayant
le meilleur Dice moyen. Il écrit le tableau des cinq Dice les plus faibles et
leurs superpositions dans `results/brats_failure_analysis.csv` et
`results/failure_overlays/`. La page **Résultats globaux** montre ces métriques,
les causes probables et les images. Les causes sont des hypothèses descriptives,
pas des diagnostics.

### Évaluation BraTS2020 complète avec groupe test réservé

Le chemin du téléchargement brut est configuré dans `morphort/config.py`
(`~/Downloads/brats_raw`). Après avoir extrait BraTS2020 à cet emplacement,
lancez depuis la racine du dépôt :

```powershell
python -m scripts.evaluate_full_brats
```

Le script crée `results/split.json` une seule fois avec la graine configurée.
Il règle les huit configurations sur 100 patients, puis calcule les Dice de la
tumeur entière, du rehaussé, du noyau nécrotique et de l'œdème sur les patients
test uniquement. Le CSV `results/brats_test.csv` contient les scores
individuels ; `results/brats_test_summary.csv` contient moyenne, écart-type
échantillonnal, médiane, minimum, maximum et IC bootstrap à 95 % de la moyenne
(10 000 réplications). Les mêmes statistiques apparaissent dans la page
**Résultats globaux**.

Le téléchargement local contient 369 dossiers d'entraînement, mais
`BraTS20_Training_355` ne possède pas de fichier `seg`. Il est donc exclu du
réglage et du test supervisés, et son absence est consignée dans le split ; les
368 patients ayant les cinq NIfTI donnent 100 patients de réglage et 268 de
test. Le script s'arrête explicitement si la structure ou le nombre de cas
change. Un CSV de progression permet de reprendre les cas test déjà terminés
sans les recalculer.

### Réglage des paramètres et test indépendant

Pour comparer les rayons d'ouverture et de fermeture ainsi que deux méthodes
de seuil, lancez :

```powershell
python -m scripts.tune_params
```

Le script exige les 20 patients préparés. Il trie les identifiants et affecte
les indices pairs (10 patients répartis dans la liste) au réglage, les indices
impairs (10 autres) au test indépendant. Une grille fixe de huit combinaisons
(rayons 0,5 ou 1 mm, seuil Otsu ou percentile 90 du cerveau normalisé) est
classée sur le Dice moyen du seul groupe de réglage. Les paramètres retenus
sont ensuite évalués une seule fois sur le groupe indépendant, sans modifier
la sélection. Les égalités sont départagées par rayons puis nom de méthode,
dans un ordre déterministe.

Le script écrit `results/brats_tuning_grid.csv` et
`results/brats_tuning_patients.csv`. La page **Résultats globaux** montre la
grille, les statistiques et les scores individuels des deux groupes dans des
sections séparées. Les échecs de segmentation et les scores faibles ne sont
pas cachés. Cette séparation rend la comparaison plus honnête, mais 10 patients
par groupe restent un petit échantillon exploratoire, sans valeur clinique.

Avec la configuration commune, l'évaluation relancée sur les 20 cas obtient un
Dice moyen de 0,7494 (écart-type 0,2011, min. 0,0000, max. 0,9431), un Jaccard
moyen de 0,6289 et un Hausdorff 95 moyen de 21,0895 mm. La moyenne sur les 20
cas n'est pas directement comparable aux moyennes de réglage et de test : ce
sont des patients différents, et le groupe de réglage sert à sélectionner les
paramètres. Le score nul est conservé. Ces valeurs sont exploratoires et
dépendent de cet échantillon.

### Sous-régions morphologiques

Sur la page 2, lorsque T1ce et FLAIR sont disponibles, MorphoRT estime la tumeur
rehaussée par seuil T1ce normalisé au cerveau, ouverture et plus grande
composante ; le noyau nécrotique par fermeture, remplissage géodésique des
cavités et vérification d'un T1ce sombre ; l'œdème par seuil FLAIR dans la
tumeur entière, soustraction des autres régions et ouverture par reconstruction.
L'onglet **Superposition et scores** compare les trois régions colorées au masque
expert BraTS et affiche leurs Dice sans filtrer les valeurs faibles. **Voir les
étapes** donne les formules et images intermédiaires de chaque opération.
Les seuils d'intensité sont des heuristiques explicables, non un modèle validé ;
un mauvais Dice est affiché et expliqué tel quel.

## Tests

Installez les outils de test et exécutez pytest :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check morphort pages tests streamlit_app.py
```

scikit-image sert uniquement de référence indépendante dans les tests des
opérations morphologiques. Les fonctions de production sont écrites à la main
avec NumPy.

## Organisation

- `morphort/` : morphologie, segmentation, mesures, modèles et données.
- `pages/` : les cinq étapes Streamlit et la page d'évaluation globale.
- `tests/` : tests unitaires et comparaisons aux références.
- `docs/MATHS.md` : formules expliquées.
- `docs/RAPPORT.md` : périmètre et limites du prototype.

La segmentation est un exemple algorithmique basé sur l'intensité. Les mesures
et courbes sont uniquement descriptives et exploratoires. L'optimisation SLSQP
produit un scénario mathématique simplifié et ne doit jamais guider des doses
ou décisions de soins.
