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

Sur l'échantillon actuel de 20 patients, l'évaluation de base donne un Dice
moyen de 0,6176 (écart-type 0,3307, de 0,0662 à 0,9652), un Jaccard moyen de
0,5216 et un Hausdorff 95 moyen de 36,1005 mm. Ce relevé décrit cette exécution,
pas un autre échantillon ni une référence clinique.

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

Sur l'échantillon actuel, la sélection retient une ouverture de 1 mm, une
fermeture de 1 mm et le seuil percentile 90. Le Dice moyen est de 0,7241 sur le
groupe de réglage et de 0,7748 sur les 10 patients indépendants ; le minimum
sur le groupe de réglage est 0,0000 et reste visible dans les résultats. Ces
chiffres décrivent uniquement cette exécution.

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
