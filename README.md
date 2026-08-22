# Bank Churn Prediction - Pipeline MLOps Complet


La présentation + La vidéo du pipeline MLOps au complet sont consultables via ce lien https://drive.google.com/drive/folders/1bGnrVLeOrvdb0vtf8ifiAq_58H_oSlTe?usp=sharing



[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-orange.svg)](https://cloud.google.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black.svg)](https://github.com/features/actions)

> **Pipeline MLOps de bout en bout** pour la prédiction du churn bancaire : de l'entraînement du modèle au déploiement en production avec monitoring automatisé.

![Banner](docs/images/banner.png)

---

## Table des Matières

- [ Aperçu du Projet](#-aperçu-du-projet)
- [ Fonctionnalités](#-fonctionnalités)
- [ Architecture](#️-architecture)
- [ Démarrage Rapide](#-démarrage-rapide)
- [ Dataset](#-dataset)
- [ Modèle ML](#-modèle-ml)
- [ API Documentation](#-api-documentation)
- [ Docker](#-docker)
- [ Déploiement GCP](#️-déploiement-gcp)
- [ Monitoring](#-monitoring)
- [ CI/CD Pipeline](#-cicd-pipeline)
- [ Compétences Acquises](#-compétences-acquises)
- [ Structure du Projet](#-structure-du-projet)
- [ Technologies Utilisées](#️-technologies-utilisées)


---

## Aperçu du Projet

### Contexte Business

ABC Multistate Bank fait face à un **taux de churn de 20%**, générant des pertes significatives. Ce projet développe une solution d'IA prédictive permettant d'identifier les clients à risque de départ **avant** qu'ils ne partent.

### Objectif

Manquer un client sur le point de partir coûte plus cher qu'alerter à tort
sur un client fidèle : une fausse alerte déclenche une offre de rétention,
un départ non détecté fait perdre le client. Le modèle est donc optimisé
sur le **recall**, quitte à accepter une précision basse.

Tous les chiffres de ce README proviennent de `models/model_metadata.json`
et de `models/model_comparison.json`, régénérés par `make train-complet`.

### Solution Déployée

```
📊 Données → 🤖 Modèle ML → 🐳 Docker → ☁️ GCP Cloud Run → 📈 Monitoring
     ↓            ↓            ↓              ↓               ↓
  Kaggle      Logistic      Container      Production       Evidently
            Regression
```

---

## Fonctionnalités

### 🤖 Machine Learning
- ✅ Exploration des données (EDA)
- ✅ Feature engineering : 14 variables créées, 27 features vues par le modèle
- ✅ Gestion du déséquilibre par SMOTE **et** `class_weight="balanced"`
- ✅ Comparaison de 4 modèles (LR, RF, GB, XGBoost)
- ✅ Interprétabilité par les coefficients du modèle

### 🔌 API & Déploiement
- ✅ API REST avec FastAPI
- ✅ Documentation Swagger/OpenAPI automatique
- ✅ Conteneurisation Docker
- ✅ Déploiement serverless sur GCP Cloud Run
- ✅ Auto-scaling (0 à 10 instances)

### 🔄 MLOps
- ✅ Pipeline CI/CD avec GitHub Actions
- ✅ Tests automatisés (pytest)
- ✅ Scan de secrets (gitleaks, en pre-commit et en CI)
- ✅ Monitoring avec Evidently (détection de dérive)
- ✅ Pipeline de réentraînement déclenché sur seuil
- ✅ Versioning des modèles par horodatage

### 🎨 Interface Utilisateur
- ✅ Dashboard Streamlit interactif
- ✅ Prédictions en temps réel
- ✅ Visualisations des KPIs
- ✅ Recommandations d'actions

### Non implémenté

Ces briques appartiennent à un pipeline MLOps complet mais ne sont pas
présentes dans ce dépôt. Elles sont listées ici pour ce qu'elles sont :
des pistes, pas des fonctionnalités.

| Piste | État |
|-------|------|
| Rollback automatique en cas d'échec de déploiement | Non configuré : la CI signale l'échec sans revenir en arrière |
| Déploiement canary et A/B testing | Non implémenté |
| Mesure de latence et de disponibilité en production | Aucune instrumentation en place |
| Optimisation des hyperparamètres | Les modèles utilisent des valeurs fixées à la main |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GITHUB REPOSITORY                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │  Code   │───▶│  Tests  │───▶│  Build  │───▶│ Deploy  │      │
│  │  Push   │    │ Pytest  │    │ Docker  │    │Cloud Run│      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘      │
│                                                     │            │
└─────────────────────────────────────────────────────┼────────────┘
                                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GOOGLE CLOUD PLATFORM                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Cloud Run   │  │    GCS       │  │   Logging    │          │
│  │   (API)      │◀▶│  (Modèles)   │  │ (Monitoring) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                                                        │
└─────────┼────────────────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENTS                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Swagger    │  │  Streamlit   │  │  Applications│          │
│  │     UI       │  │  Dashboard   │  │   Tierces    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Démarrage Rapide

### Prérequis

- Python 3.11+
- Docker (optionnel)
- Compte GCP (pour déploiement cloud)

### Installation Locale

```bash
# 1. Cloner le repository
git clone https://github.com/DenisMT22/bank-churn-mlops.git
cd bank-churn-mlops

# 2. Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Télécharger le dataset
# Placer Bank_Churn_Prediction.csv dans data/raw/

# 5. Entraîner le modèle
cd src/models
python train.py

# 6. Lancer l'API
cd ../api
uvicorn main:app --reload --port 8080

# 7. Lancer le Dashboard (nouveau terminal)
cd ../..
streamlit run streamlit_app.py
```

### Accès aux Interfaces

| Interface | URL | Description |
|-----------|-----|-------------|
| API Swagger | http://localhost:8080/docs | Documentation interactive |
| API ReDoc | http://localhost:8080/redoc | Documentation alternative |
| Health Check | http://localhost:8080/health | État de l'API |
| Dashboard | http://localhost:8501 | Interface Streamlit |

---

## Dataset

### Source
**Bank Customer Churn Dataset** - [Kaggle](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset)

### Caractéristiques

| Propriété | Valeur |
|-----------|--------|
| Observations | 10 000 |
| Colonnes | 12 (identifiant, 10 variables, cible) |
| Target | `churn` (0/1) |
| Déséquilibre | 79.6% / 20.4% (7 963 / 2 037) |

### Variables

| Variable | Type | Description |
|----------|------|-------------|
| customer_id | int | Identifiant client, écarté à l'entraînement |
| credit_score | int | Score de crédit (300-900) |
| country | cat | Pays (France/Germany/Spain) |
| gender | cat | Genre (Male/Female) |
| age | int | Âge du client |
| tenure | int | Ancienneté (années) |
| balance | float | Solde du compte |
| products_number | int | Nombre de produits |
| credit_card | bin | Possède carte crédit |
| active_member | bin | Membre actif |
| estimated_salary | float | Salaire estimé |
| **churn** | bin | **Target - A quitté (1) ou non (0)** |

---

## Modèle ML

### Comparaison des Modèles

Le script `train.py` compare automatiquement 4 algorithmes et sélectionne le meilleur basé sur le **Recall** :

| Modèle | Accuracy | Precision | Recall ⭐ | F1-Score | ROC-AUC |
|--------|----------|-----------|----------|----------|---------|
| **Logistic Regression** 🏆 | 78.00% | 47.48% | **76.41%** | 58.57% | 85.39% |
| XGBoost | 81.65% | 53.68% | 71.74% | 61.41% | 85.43% |
| Random Forest | 83.75% | 59.36% | 63.88% | 61.54% | 85.68% |
| Gradient Boosting | 85.60% | 67.66% | 56.02% | 61.29% | 85.73% |

> **🏆 Gagnant : Logistic Regression** avec un Recall de 76.41%

> Ces vingt valeurs sont écrites par `make train-complet` dans
> `models/model_comparison.json`. Aucune n'est saisie à la main.

### Pourquoi Logistic Regression ?

Bien que d'autres modèles aient une meilleure Accuracy, **Logistic Regression** a été sélectionné car :

1. **Meilleur Recall (76.41%)** : Détecte le plus de churners
2. **Interprétabilité** : Coefficients explicables pour le métier
3. **Rapidité** : Inférence ultra-rapide en production
4. **Robustesse** : Moins de risque d'overfitting

### Matrice de Confusion (Test Set : 2,000 samples)

```
              Prédit 0    Prédit 1
Réel 0         1,249         344      (TN / FP)
Réel 1            96         311      (FN / TP)
```

**Interprétation :**
- **311 churners correctement identifiés** (True Positives)
- **96 churners manqués** (False Negatives) - à minimiser
- **344 fausses alertes** (False Positives) - acceptables

### Métriques du Modèle Retenu

Mesurées sur le jeu de test de 2 000 clients, jamais vu à l'entraînement.

| Métrique | Score |
|----------|-------|
| **Recall** | 76.41% |
| **Precision** | 47.48% |
| **F1-Score** | 58.57% |
| **ROC-AUC** | 85.39% |
| **Accuracy** | 78.00% |
| Recall en validation croisée | 75.71% ± 2.90% |

Une précision de 47% signifie qu'environ une alerte sur deux est une fausse
alerte. C'est le prix assumé du recall élevé, et le principal axe
d'amélioration du modèle.

> Source : `models/model_metadata.json`, régénéré par `make train`.

### Variables les Plus Influentes

Une régression logistique n'expose pas de `feature_importances_` : ce sont
ses coefficients qui portent l'information. Appliqués à des variables
standardisées, ils se comparent entre eux. Un coefficient positif pousse
vers le départ, un coefficient négatif vers la fidélité.

| Variable | Coefficient | Lecture |
|----------|-------------|---------|
| `HasMultipleProducts` | −3.58 | Détenir plusieurs produits retient fortement |
| `products_number` | +2.58 | Mais le nombre brut de produits joue en sens inverse |
| `balance` | +1.33 | Un solde élevé accompagne les départs |
| `BalancePerProduct` | −1.32 | Rapporté au nombre de produits, l'effet s'inverse |
| `age` | +1.29 | Les clients plus âgés partent davantage |
| `active_member` | −0.64 | Un membre actif reste |

Les deux premières lignes se lisent ensemble : le modèle distingue le fait
d'avoir plusieurs produits du nombre exact de produits détenus.

> Extraits de `models/trained/model_latest.pkl` après `make train`.

---

## API Documentation

### Endpoints

#### `GET /health`
Vérifier l'état de l'API.

```bash
curl http://localhost:8080/health
```

**Réponse :**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "2025-11-19T14:30:00",
  "uptime_seconds": 3600.5
}
```

#### `GET /metrics`
Obtenir les métriques du modèle.

```bash
curl http://localhost:8080/metrics
```

#### `POST /predict`
Prédire le churn pour un client.

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "credit_score": 650,
    "country": "France",
    "gender": "Female",
    "age": 35,
    "tenure": 5,
    "balance": 125000.0,
    "products_number": 2,
    "credit_card": 1,
    "active_member": 1,
    "estimated_salary": 50000.0
  }'
```

**Réponse :**
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.234,
  "risk_level": "Low",
  "confidence": 0.766,
  "timestamp": "2025-11-19T16:40:15"
}
```

#### `POST /predict/batch`
Prédictions pour plusieurs clients (max 1000).

---

## Docker

### Build et Run

```bash
# Build l'image
docker build -f deployment/Dockerfile -t churn-api:latest .

# Run le conteneur
docker run -d -p 8080:8080 --name churn-api churn-api:latest

# Vérifier les logs
docker logs -f churn-api

# Arrêter
docker stop churn-api && docker rm churn-api
```

### Docker Compose

```bash
cd deployment
docker-compose up -d
```

---

## Déploiement GCP

### Configuration Initiale

```bash
# 1. Installer gcloud CLI
# https://cloud.google.com/sdk/docs/install

# 2. Authentification
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 3. Activer les APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable storage.googleapis.com

# 4. Exécuter le script de setup
chmod +x scripts/setup_gcp.sh
./scripts/setup_gcp.sh
```

### Déploiement Manuel

```bash
# Build et push l'image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/churn-api

# Déployer sur Cloud Run
gcloud run deploy churn-api \
  --image gcr.io/YOUR_PROJECT/churn-api \
  --region europe-west1 \
  --platform managed \
  --allow-unauthenticated
```

### URL de Production

Après déploiement, l'API est accessible à :
```
https://churn-api-xxxxx-ew.a.run.app
```

---

## Monitoring

### Evidently AI

Le monitoring utilise Evidently pour détecter :
- **Data Drift** : Changements dans la distribution des features
- **Model Drift** : Dégradation des performances
- **Data Quality** : Valeurs manquantes, outliers

### Génération des Rapports

```bash
cd src/monitoring
python evidently_monitor.py
```

Les rapports HTML sont générés dans `monitoring/reports/`.

### Alertes Configurées

| Métrique | Seuil | Action | Défini dans |
|----------|-------|--------|-------------|
| Recall | < 70% | Réentraînement | `evidently_monitor.py` |
| Colonnes en dérive | > 30% | Alerte dans le rapport | `evidently_monitor.py` |

---

## CI/CD Pipeline

### Workflow GitHub Actions

```yaml
# .github/workflows/ci-cd.yml
on:
  push:
    branches: [main]

jobs:
  test → build → deploy
```

### Étapes du Pipeline

Le cœur du pipeline ne dépend d'aucun compte cloud : les quatre premiers
jobs tournent entièrement sur le runner GitHub.

| Job | Contenu | Bloquant |
|-----|---------|----------|
| `secret-scan` | gitleaks sur tout l'historique et sur les fichiers | oui |
| `lint` | ruff pour le style, bandit pour la sécurité statique | oui |
| `test` | entraînement réel puis pytest | oui |
| `docker-build` | build de l'image, démarrage, `/health` et une prédiction | oui |
| `deploy` | Cloud Run — **déclenchement manuel uniquement** | non |

Aucun filtre de chemin n'est appliqué : chaque poussée déclenche l'ensemble
du pipeline, y compris le scan de secrets. Un filtre laisserait passer sans
contrôle un commit ne touchant pas à `src/`, ce qui est exactement le cas
d'une clé déposée à la racine.

Le job `test` entraîne réellement le modèle depuis `data/raw` avant de
lancer la suite, puis compare les métriques régénérées à celles versionnées.
Il n'y a plus ni `continue-on-error`, ni artefacts factices.

### Tests

62 tests, répartis en trois fichiers :

| Fichier | Portée |
|---------|--------|
| `test_preprocessing.py` | Création des features, encodage, cohérence des colonnes |
| `test_model.py` | Chargement des artefacts, forme des sorties, déterminisme, métriques |
| `test_api.py` | Endpoints, validation des entrées, gestion des erreurs |

Couverture mesurée : **25 % des lignes de `src/`**. Le détail est plus
parlant que le total : `schemas.py` 100 %, `preprocessing.py` 70 %,
`api/main.py` 40 %, tandis que `train.py`, `retrain.py` et
`evidently_monitor.py` sont à 0 % — ce sont des scripts exécutés de bout
en bout, non couverts par des tests unitaires. C'est le chiffre réel,
pas un objectif.

```bash
make test                                  # la suite
pytest tests/ --cov=src --cov-report=term  # avec la couverture
```

### Retraining Automatique

```yaml
# .github/workflows/retrain.yml
on:
  schedule:
    - cron: '0 2 * * 1'  # Tous les lundis à 2h
  workflow_dispatch:      # Déclenchement manuel
```

---

## Compétences Acquises

Ce projet a permis de développer et démontrer les compétences suivantes :

### Data Science & Machine Learning

| Compétence | Description |
|------------|-------------|
| **Analyse Exploratoire (EDA)** | Exploration statistique, visualisations, détection d'outliers |
| **Feature Engineering** | Création de 14 variables métier, 27 features en entrée du modèle |
| **Modélisation ML** | Entraînement, comparaison et sélection de modèles |
| **Gestion du Déséquilibre** | SMOTE et pondération des classes, échantillonnage stratifié |
| **Évaluation de Modèles** | Métriques adaptées (Recall prioritaire), cross-validation |
| **Interprétabilité** | Lecture des coefficients du modèle retenu |

### Développement & API

| Compétence | Description |
|------------|-------------|
| **Développement API REST** | Conception et implémentation avec FastAPI |
| **Documentation API** | OpenAPI/Swagger, schémas Pydantic |
| **Tests Unitaires** | pytest, tests d'API sur les endpoints |
| **Validation de Données** | Schémas Pydantic, gestion des erreurs |
| **Logging & Monitoring** | Logs structurés, métriques applicatives |

### DevOps & Infrastructure

| Compétence | Description |
|------------|-------------|
| **Conteneurisation** | Docker, Docker Compose, optimisation images |
| **CI/CD** | GitHub Actions, pipelines automatisés |
| **Cloud Computing** | GCP Cloud Run, Cloud Storage, IAM |
| **Infrastructure as Code** | Scripts de déploiement automatisés |
| **Gestion des Secrets** | Variables d'environnement, scan gitleaks, aucun identifiant versionné |

### MLOps

| Compétence | Description |
|------------|-------------|
| **Pipeline ML Automatisé** | Preprocessing → Training → Deployment |
| **Versioning de Modèles** | Horodatage des modèles, comparaison des exécutions |
| **Monitoring ML** | Détection de drift avec Evidently |
| **Retraining Automatique** | Pipelines déclenchés sur conditions |
| **Reproductibilité** | Pipeline régénérable depuis la donnée source, graine fixée |

### Gestion de Projet

| Compétence | Description |
|------------|-------------|
| **Rédaction de Cahier des Charges** | Expression des besoins, spécifications |
| **Documentation Technique** | README, API docs, architecture |
| **Présentation** | Communication technique et business |
| **Versioning** | Git, branching strategy, pull requests |

---

## Structure du Projet

```
bank-churn-mlops/
├── 📁 .github/
│   └── workflows/
│       ├── ci-cd.yml              # Pipeline CI/CD principal
│       └── retrain.yml            # Pipeline de retraining
├── 📁 data/
│   ├── raw/                       # Données brutes
│   │   └── Churn_Modelling.csv
│   └── processed/                 # Données transformées
├── 📁 models/
│   ├── trained/                   # Modèles entraînés
│   │   └── model_latest.pkl
│   ├── preprocessor.pkl           # Pipeline de preprocessing
│   └── model_metadata.json        # Métadonnées du modèle
├── 📁 src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py               # Application FastAPI
│   │   └── schemas.py            # Schémas Pydantic
│   ├── models/
│   │   ├── __init__.py
│   │   ├── preprocessing.py      # Pipeline de preprocessing
│   │   ├── train.py              # Script d'entraînement
│   │   └── retrain.py            # Script de retraining
│   └── monitoring/
│       ├── __init__.py
│       └── evidently_monitor.py  # Monitoring ML
├── 📁 tests/
│   ├── test_api.py               # Tests API
│   ├── test_preprocessing.py     # Tests preprocessing
│   └── test_model.py             # Tests modèle
├── 📁 notebooks/
│   └── 01_eda.ipynb              # Exploration des données
├── 📁 deployment/
│   ├── Dockerfile                # Image Docker
│   ├── docker-compose.yml        # Orchestration locale
│   └── cloudbuild.yaml           # GCP Cloud Build
├── 📁 monitoring/
│   └── reports/                  # Rapports Evidently
├── 📁 scripts/
│   ├── setup_gcp.sh              # Configuration GCP
│   └── deploy.sh                 # Script de déploiement
├── 📁 docs/
│   ├── api_documentation.md      # Documentation API
│   ├── architecture.md           # Documentation architecture
│   └── images/                   # Images documentation
├── 📁 presentation/
│   └── slides.html               # Présentation
├── 📄 .gitignore
├── 📄 .dockerignore
├── 📄 requirements.txt           # Dépendances Python
├── 📄 requirements-dev.txt       # Dépendances développement
├── 📄 streamlit_app.py           # Dashboard Streamlit
└── 📄 README.md                  # Ce fichier
```

---

## Technologies Utilisées

### Machine Learning
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### API & Backend
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?style=for-the-badge&logo=uvicorn&logoColor=white)

### DevOps & Cloud
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)
![Google Cloud](https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)

### Monitoring & Visualisation
![Evidently](https://img.shields.io/badge/Evidently-FF6F61?style=for-the-badge&logo=evidently&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

