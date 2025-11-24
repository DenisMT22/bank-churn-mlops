#!/bin/bash

# Script de Configuration GCP pour le Projet MLOps Churn Prediction
# ==================================================================

set -e  # Arrêter en cas d'erreur

echo "============================================================"
echo "   CONFIGURATION GCP - MLOPS CHURN PREDICTION"
echo "============================================================"

# Vérifier si gcloud est installé
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI n'est pas installé"
    echo "Installez-le depuis: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Variables (à personnaliser)
read -p "Entrez votre PROJECT_ID GCP: " PROJECT_ID
read -p "Entrez la REGION (default: europe-west1): " REGION
REGION=${REGION:-europe-west1}

SERVICE_ACCOUNT_NAME="mlops-service-account"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo ""
echo "Configuration:"
echo "  PROJECT_ID: $PROJECT_ID"
echo "  REGION: $REGION"
echo "  SERVICE_ACCOUNT: $SERVICE_ACCOUNT_EMAIL"
echo ""

read -p "Continuer? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "❌ Configuration annulée"
    exit 0
fi

# 1. Définir le projet
echo ""
echo "📋 1. Configuration du projet..."
gcloud config set project $PROJECT_ID
echo "✅ Projet configuré"

# 2. Activer les APIs nécessaires
echo ""
echo "🔌 2. Activation des APIs GCP..."
APIS=(
    "cloudbuild.googleapis.com"
    "run.googleapis.com"
    "storage.googleapis.com"
    "containerregistry.googleapis.com"
    "artifactregistry.googleapis.com"
    "logging.googleapis.com"
    "monitoring.googleapis.com"
    "cloudscheduler.googleapis.com"
    "cloudfunctions.googleapis.com"
)

for api in "${APIS[@]}"; do
    echo "  Activation de $api..."
    gcloud services enable $api --quiet
done
echo "✅ APIs activées"

# 3. Créer Service Account
echo ""
echo "👤 3. Création du Service Account..."
if gcloud iam service-accounts describe $SERVICE_ACCOUNT_EMAIL &>/dev/null; then
    echo "  ⚠️  Service Account existe déjà"
else
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --display-name="MLOps Service Account for Churn Prediction"
    echo "✅ Service Account créé"
fi

# 4. Attribuer les rôles
echo ""
echo "🔐 4. Attribution des permissions..."
ROLES=(
    "roles/storage.admin"
    "roles/run.admin"
    "roles/logging.logWriter"
    "roles/monitoring.metricWriter"
    "roles/cloudbuild.builds.editor"
)

for role in "${ROLES[@]}"; do
    echo "  Attribution de $role..."
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
        --role="$role" \
        --quiet
done
echo "✅ Permissions attribuées"

# 5. Créer la clé du Service Account
echo ""
echo "🔑 5. Création de la clé du Service Account..."
KEY_FILE="gcp-key.json"

if [ -f "$KEY_FILE" ]; then
    read -p "  ⚠️  $KEY_FILE existe. Le remplacer? (y/n): " REPLACE
    if [ "$REPLACE" != "y" ]; then
        echo "  ⏭️  Clé non remplacée"
    else
        gcloud iam service-accounts keys create $KEY_FILE \
            --iam-account=$SERVICE_ACCOUNT_EMAIL
        echo "✅ Nouvelle clé créée: $KEY_FILE"
    fi
else
    gcloud iam service-accounts keys create $KEY_FILE \
        --iam-account=$SERVICE_ACCOUNT_EMAIL
    echo "✅ Clé créée: $KEY_FILE"
fi

# 6. Créer les buckets Cloud Storage
echo ""
echo "🪣 6. Création des buckets Cloud Storage..."

# Bucket pour les modèles
BUCKET_MODELS="${PROJECT_ID}-models"
if gsutil ls -b gs://$BUCKET_MODELS &>/dev/null; then
    echo "  ⚠️  Bucket $BUCKET_MODELS existe déjà"
else
    gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_MODELS
    echo "✅ Bucket créé: gs://$BUCKET_MODELS"
fi

# Bucket pour les données
BUCKET_DATA="${PROJECT_ID}-data"
if gsutil ls -b gs://$BUCKET_DATA &>/dev/null; then
    echo "  ⚠️  Bucket $BUCKET_DATA existe déjà"
else
    gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_DATA
    echo "✅ Bucket créé: gs://$BUCKET_DATA"
fi

# 7. Uploader les modèles initiaux
echo ""
echo "📤 7. Upload des modèles vers GCS..."

if [ -f "../models/trained/model_latest.pkl" ]; then
    gsutil cp ../models/trained/model_latest.pkl gs://$BUCKET_MODELS/
    echo "✅ model_latest.pkl uploadé"
else
    echo "  ⚠️  Modèle non trouvé (entraîner d'abord)"
fi

if [ -f "../models/preprocessor.pkl" ]; then
    gsutil cp ../models/preprocessor.pkl gs://$BUCKET_MODELS/
    echo "✅ preprocessor.pkl uploadé"
else
    echo "  ⚠️  Preprocessor non trouvé"
fi

if [ -f "../models/model_metadata.json" ]; then
    gsutil cp ../models/model_metadata.json gs://$BUCKET_MODELS/
    echo "✅ model_metadata.json uploadé"
else
    echo "  ⚠️  Métadonnées non trouvées"
fi

# 8. Configurer les secrets GitHub
echo ""
echo "🔒 8. Configuration des secrets GitHub..."
echo ""
echo "⚠️  IMPORTANT: Ajoutez ces secrets dans GitHub:"
echo "  1. Allez sur: https://github.com/VOTRE-USERNAME/VOTRE-REPO/settings/secrets/actions"
echo "  2. Ajoutez ces secrets:"
echo ""
echo "     GCP_PROJECT_ID = $PROJECT_ID"
echo "     GCP_SA_KEY = (contenu de $KEY_FILE)"
echo ""
echo "  Pour GCP_SA_KEY, copiez le contenu avec:"
echo "     cat $KEY_FILE | base64"
echo ""

# 9. Créer fichier .env local
echo ""
echo "📝 9. Création du fichier .env local..."
cat > ../.env << EOF
# Configuration GCP
PROJECT_ID=$PROJECT_ID
REGION=$REGION
BUCKET_MODELS=$BUCKET_MODELS
BUCKET_DATA=$BUCKET_DATA
SERVICE_ACCOUNT_EMAIL=$SERVICE_ACCOUNT_EMAIL

# API Configuration
ENVIRONMENT=development
LOG_LEVEL=info
EOF

echo "✅ Fichier .env créé"

# 10. Résumé final
echo ""
echo "============================================================"
echo "   ✅ CONFIGURATION GCP TERMINÉE"
echo "============================================================"
echo ""
echo "📋 Résumé:"
echo "  ✅ Projet configuré: $PROJECT_ID"
echo "  ✅ APIs activées"
echo "  ✅ Service Account créé: $SERVICE_ACCOUNT_EMAIL"
echo "  ✅ Permissions attribuées"
echo "  ✅ Clé générée: $KEY_FILE"
echo "  ✅ Buckets créés:"
echo "     - gs://$BUCKET_MODELS"
echo "     - gs://$BUCKET_DATA"
echo "  ✅ Modèles uploadés (si disponibles)"
echo ""
echo "🚀 Prochaines étapes:"
echo "  1. Configurez les secrets GitHub (voir instructions ci-dessus)"
echo "  2. Committez et pushez votre code"
echo "  3. Le pipeline CI/CD se déclenchera automatiquement"
echo ""
echo "🔗 Liens utiles:"
echo "  GCP Console: https://console.cloud.google.com/home/dashboard?project=$PROJECT_ID"
echo "  Cloud Storage: https://console.cloud.google.com/storage/browser?project=$PROJECT_ID"
echo "  Cloud Run: https://console.cloud.google.com/run?project=$PROJECT_ID"
echo ""
echo "============================================================"