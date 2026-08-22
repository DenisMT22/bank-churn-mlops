#!/bin/bash

# Script de Configuration GCP pour le Projet MLOps Churn Prediction
# ==================================================================

set -e  # Arrêter en cas d'erreur


# Racine du projet, calculee depuis l'emplacement de ce script : les
# chemins ne dependent donc pas du repertoire courant.
RACINE_PROJET="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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
#
# La clé est écrite HORS du dépôt, dans le répertoire de configuration de
# l'utilisateur. Une version antérieure de ce script l'écrivait à la racine
# du projet : un fichier .gitignore était alors la seule chose qui séparait
# une clé privée d'un dépôt public. Un .gitignore mal édité, un git add -f
# ou un clone copié suffisaient à la faire fuir.
#
# En local, plutôt qu'une clé de compte de service, préférer :
#   gcloud auth application-default login
# En intégration continue, passer par un secret GitHub, jamais par un fichier.
echo ""
echo "🔑 5. Création de la clé du Service Account..."

KEY_DIR="${BANK_CHURN_CONFIG_DIR:-$HOME/.config/bank-churn-mlops}"
KEY_FILE="$KEY_DIR/gcp-key.json"

mkdir -p "$KEY_DIR"
chmod 700 "$KEY_DIR"

creer_cle() {
    gcloud iam service-accounts keys create "$KEY_FILE" \
        --iam-account="$SERVICE_ACCOUNT_EMAIL"
    # Lecture réservée au propriétaire.
    chmod 600 "$KEY_FILE"
    echo "✅ Clé créée : $KEY_FILE"
    echo "   Pour l'utiliser :"
    echo "     export GOOGLE_APPLICATION_CREDENTIALS=\"$KEY_FILE\""
    echo "   Ne jamais copier ce fichier dans le dépôt."
}

if [ -f "$KEY_FILE" ]; then
    read -p "  ⚠️  Une clé existe déjà dans $KEY_DIR. La remplacer ? (y/n) : " REPLACE
    if [ "$REPLACE" != "y" ]; then
        echo "  ⏭️  Clé non remplacée"
    else
        creer_cle
    fi
else
    creer_cle
fi

# Garde-fou : aucune clé ne doit trainer dans l'arbre du projet.
CLES_EGAREES="$(find "$RACINE_PROJET" -name '*gcp-key*.json' -not -path '*/venv/*' 2>/dev/null || true)"
if [ -n "$CLES_EGAREES" ]; then
    echo ""
    echo "  ⛔ Une clé a été trouvée dans le dépôt :"
    echo "$CLES_EGAREES"
    echo "  La supprimer et révoquer la clé correspondante dans la console IAM."
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

if [ -f "$RACINE_PROJET/models/trained/model_latest.pkl" ]; then
    gsutil cp "$RACINE_PROJET/models/trained/model_latest.pkl" "gs://$BUCKET_MODELS/"
    echo "✅ model_latest.pkl uploadé"
else
    echo "  ⚠️  Modèle non trouvé (entraîner d'abord)"
fi

if [ -f "$RACINE_PROJET/models/preprocessor.pkl" ]; then
    gsutil cp "$RACINE_PROJET/models/preprocessor.pkl" "gs://$BUCKET_MODELS/"
    echo "✅ preprocessor.pkl uploadé"
else
    echo "  ⚠️  Preprocessor non trouvé"
fi

if [ -f "$RACINE_PROJET/models/model_metadata.json" ]; then
    gsutil cp "$RACINE_PROJET/models/model_metadata.json" "gs://$BUCKET_MODELS/"
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
echo "     GCP_SA_KEY = (contenu JSON brut de $KEY_FILE)"
echo ""
echo "  Pour GCP_SA_KEY, copier le contenu tel quel :"
echo "     cat \"$KEY_FILE\""
echo "  Ne pas encoder en base64 : l'action d'authentification attend le JSON brut."
echo ""

# 9. Créer fichier .env local
echo ""
echo "📝 9. Création du fichier .env local..."
# Ecrit a la racine du projet, quel que soit le repertoire d'ou le script
# est lance. Ce fichier ne contient que de la configuration, aucun secret.
cat > "$RACINE_PROJET/.env" << EOF
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