# Atelier de morphologie mathématique

Une petite plateforme pédagogique pour comprendre les opérations de base de la
morphologie mathématique sur une image binaire. L’atelier compare l’image de
départ au résultat de la dilatation, de l’érosion, de l’ouverture ou de la
fermeture. Le calcul est expliqué dans `frontend/src/morphology.ts`.

L’interface fonctionne seule avec une image d’exemple : le serveur Python n’est
pas nécessaire pour utiliser l’atelier. L’API de recherche existante reste
disponible séparément.

## Lancer l’atelier

Node.js 20 ou plus récent est nécessaire.

```powershell
cd frontend
npm install
npm run dev
```

Ouvrez l’adresse indiquée par Vite, généralement `http://localhost:5173`.

## Vérifier l’interface

```powershell
npm run build
npm run lint
```

## API de recherche facultative

Le backend Python historique peut être lancé indépendamment depuis la racine du
projet avec les commandes suivantes :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
$env:PIPELINE_BACKEND = "mock"
python -m uvicorn app.main:app --app-dir backend --reload
```

Cette API fournit des routes distinctes pour l’analyse d’IRM et les simulations
de recherche. Elle n’est pas utilisée par le nouvel atelier pédagogique.
