# Raccourcis pour les taches courantes du projet.
# Tout fonctionne en local, sans dependance a un service cloud.

PYTHON ?= python3.10
VENV   := venv
BIN    := $(VENV)/bin

.DEFAULT_GOAL := aide
.PHONY: aide setup install train train-complet test lint scan api dashboard monitor clean

aide:  ## Afficher cette aide
	@echo "Cibles disponibles :"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: $(BIN)/python  ## Installer les dependances (execution + developpement)
	$(BIN)/pip install -r requirements-dev.txt

setup: install train  ## Preparer un clone neuf : dependances puis artefacts du modele
	@echo ""
	@echo "Projet pret. Le modele, le preprocessor et les metriques sont regeneres."

train: ## Entrainer le modele retenu et regenerer les artefacts (~3 s)
	$(BIN)/python -m src.models.train --rapide --sans-figures

train-complet: ## Comparer les quatre modeles et regenerer les figures (~70 s)
	$(BIN)/python -m src.models.train

test: ## Lancer la suite de tests
	$(BIN)/pytest tests/ -v

lint: ## Verifier le style du code
	$(BIN)/ruff check src/ tests/ streamlit_app.py

scan: ## Rechercher des secrets dans l'historique et les fichiers
	gitleaks git . --config .gitleaks.toml --redact --no-banner --verbose
	gitleaks detect --no-git --source . --config .gitleaks.toml --redact --no-banner --verbose

api: ## Demarrer l'API de prediction sur le port 8080
	$(BIN)/uvicorn src.api.main:app --host 0.0.0.0 --port 8080 --reload

dashboard: ## Demarrer le tableau de bord Streamlit
	$(BIN)/streamlit run streamlit_app.py

monitor: ## Generer les rapports de derive Evidently
	$(BIN)/python -m src.monitoring.evidently_monitor

# models/model_metadata.json n'est pas supprime : il est suivi par git et
# sert de reference aux metriques citees dans la documentation.
clean: ## Supprimer les artefacts regeneres et les caches
	rm -rf .pytest_cache .coverage coverage.xml
	find . -path ./$(VENV) -prune -o -name '__pycache__' -type d -exec rm -rf {} +
	rm -f models/preprocessor.pkl models/last_run.json
	rm -f models/trained/*.pkl
	rm -f data/processed/data_with_features.csv
	rm -rf src/monitoring/reports
