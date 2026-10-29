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
- `pages/` : cinq étapes Streamlit.
- `tests/` : tests unitaires et comparaisons aux références.
- `docs/MATHS.md` : formules expliquées.
- `docs/RAPPORT.md` : périmètre et limites du prototype.

La segmentation est un exemple algorithmique basé sur l'intensité. Les mesures
et courbes sont uniquement descriptives et exploratoires. L'optimisation SLSQP
produit un scénario mathématique simplifié et ne doit jamais guider des doses
ou décisions de soins.
