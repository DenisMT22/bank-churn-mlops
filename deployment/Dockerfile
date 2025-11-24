# Utiliser image Python officielle
FROM python:3.10-slim

# Métadonnées
LABEL maintainer="votre.email@example.com"
LABEL description="API de prédiction de churn bancaire"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 
    
# Créer répertoire de travail
WORKDIR /app

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements
COPY requirements.txt .

# Installer dépendances Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copier le code source
COPY src/api /app/api
COPY src/models /app/src/models
COPY models /app/models

# Créer utilisateur non-root pour sécurité
RUN useradd -m -u 1000 apiuser && \
    chown -R apiuser:apiuser /app

USER apiuser

# Exposer le port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Commande de démarrage
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]