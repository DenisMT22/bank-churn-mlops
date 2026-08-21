"""
Chemins et parametres partages du projet
========================================

Tous les chemins du projet sont derives de la racine du depot, calculee
a partir de l'emplacement de ce fichier. Le code fonctionne donc a
l'identique depuis un clone place n'importe ou, dans un conteneur ou
dans un runner de CI, sans chemin absolu code en dur.

La racine peut etre forcee par la variable d'environnement PROJECT_ROOT,
ce qui sert lorsque le code est copie ailleurs que dans l'arborescence
du depot, par exemple dans l'image Docker.
"""

import os
from pathlib import Path

# src/utils/config.py -> parents[0] = src/utils, [1] = src, [2] = racine
_DEFAULT_ROOT = Path(__file__).resolve().parents[2]

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", _DEFAULT_ROOT)).resolve()

# --- Donnees ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Jeu de donnees source, seule donnee suivie par git.
RAW_DATASET = RAW_DATA_DIR / "Bank_Churn_Prediction.csv"
# Jeu de donnees enrichi, regenere par le pipeline.
PROCESSED_DATASET = PROCESSED_DATA_DIR / "data_with_features.csv"

# --- Artefacts de modelisation, tous regeneres par l'entrainement ---
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained"
MODEL_LATEST = TRAINED_MODELS_DIR / "model_latest.pkl"
MODEL_METADATA = MODELS_DIR / "model_metadata.json"
# Comparaison des modeles evalues, ecrite par la comparaison complete.
MODEL_COMPARISON = MODELS_DIR / "model_comparison.json"
# Horodatage et nom du fichier produit par le dernier entrainement.
# Volatil par nature, donc exclu du suivi git.
LAST_RUN = MODELS_DIR / "last_run.json"
PREPROCESSOR = MODELS_DIR / "preprocessor.pkl"

# --- Sorties ---
DOCS_DIR = PROJECT_ROOT / "docs"
MONITORING_REPORTS_DIR = PROJECT_ROOT / "src" / "monitoring" / "reports"

# --- Parametres d'entrainement ---
# Une seule graine pour tout le projet : sans elle, deux executions
# successives ne donnent pas les memes metriques.
RANDOM_STATE = 42
TARGET_COLUMN = "churn"
TEST_SIZE = 0.2


def ensure_directories() -> None:
    """Cree les repertoires de sortie s'ils n'existent pas encore."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        TRAINED_MODELS_DIR,
        DOCS_DIR,
        MONITORING_REPORTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"Racine du projet : {PROJECT_ROOT}")
    for nom, chemin in [
        ("Donnees source", RAW_DATASET),
        ("Donnees enrichies", PROCESSED_DATASET),
        ("Modele", MODEL_LATEST),
        ("Metadonnees", MODEL_METADATA),
        ("Preprocessor", PREPROCESSOR),
        ("Rapports monitoring", MONITORING_REPORTS_DIR),
    ]:
        etat = "present" if chemin.exists() else "absent"
        print(f"  {nom:22} {chemin}  [{etat}]")
