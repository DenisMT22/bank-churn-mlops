"""
API FastAPI pour la Prédiction de Churn Bancaire
=================================================

Endpoints:
- POST /predict : Prédiction individuelle
- POST /predict/batch : Prédictions multiples
- GET /health : Health check
- GET /metrics : Métriques du modèle

"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import json
import time
from datetime import datetime
from pathlib import Path
import logging
import sys

# Import du module de configuration des chemins. Le double essai permet
# de charger l'API aussi bien depuis le depot que depuis l'image Docker.
try:
    from src.utils import config
except ImportError:  # pragma: no cover - depend du mode d'execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils import config

# Import des schémas
from .schemas import (
    CustomerFeatures, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthResponse, ModelMetrics
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialisation de l'application
app = FastAPI(
    title="Bank Churn Prediction API",
    description="API de prédiction du churn bancaire avec MLOps",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
model = None
preprocessor = None
model_metadata = None
last_run = None
start_time = time.time()


def load_model_and_preprocessor():
    global model, preprocessor, model_metadata, last_run

    # Liste des chemins à tester (ordonnés par priorité)
    model_paths = [
        str(config.MODEL_LATEST),
        '/app/models/trained/model_latest.pkl',
    ]
    preprocessor_paths = [
        str(config.PREPROCESSOR),
        '/app/models/preprocessor.pkl',
        # Ancien emplacement, conserve pour les images deja construites.
        '/app/src/models/preprocessor.pkl',
    ]
    metadata_paths = [
        str(config.MODEL_METADATA),
        '/app/models/model_metadata.json',
    ]
    # L'horodatage du dernier entrainement vit dans un fichier separe,
    # volontairement exclu du suivi git car il change a chaque execution.
    last_run_paths = [
        str(config.LAST_RUN),
        '/app/models/last_run.json',
    ]

    # Trouver le premier chemin existant pour chacun
    model_path = next((p for p in model_paths if Path(p).exists()), None)
    preprocessor_path = next((p for p in preprocessor_paths if Path(p).exists()), None)
    metadata_path = next((p for p in metadata_paths if Path(p).exists()), None)

    try:
        if model_path is None:
            raise FileNotFoundError(f"Aucun modèle trouvé dans {model_paths}")
        if preprocessor_path is None:
            raise FileNotFoundError(f"Aucun preprocessor trouvé dans {preprocessor_paths}")
        
        logger.info(f"Chargement du modèle depuis {model_path}")
        model = joblib.load(model_path)
        logger.info("✅ Modèle chargé avec succès")

        logger.info(f"Chargement du preprocessor depuis {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        logger.info("✅ Preprocessor chargé avec succès")

        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            logger.info("✅ Métadonnées chargées")

            # Date du dernier entrainement, si la trace est presente.
            last_run_path = next((p for p in last_run_paths if Path(p).exists()), None)
            if last_run_path:
                with open(last_run_path, 'r') as f:
                    last_run = json.load(f)
                model_metadata.setdefault('timestamp', last_run.get('timestamp'))
                logger.info("✅ Trace du dernier entraînement chargée")
        else:
            logger.warning("⚠️ Fichier de métadonnées introuvable")
            model_metadata = {
                "model_name": "Unknown",
                "timestamp": datetime.now().isoformat(),
                "metrics": {}
            }
        return True

    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement : {str(e)}")
        return False



def calculate_risk_level(probability: float) -> str:
    """
    Calculer le niveau de risque basé sur la probabilité
    
    Parameters:
    -----------
    probability : float
        Probabilité de churn (0-1)
        
    Returns:
    --------
    str : 'Low', 'Medium', ou 'High'
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Medium"
    else:
        return "High"


def prepare_input_data(customer: CustomerFeatures) -> pd.DataFrame:
    """
    Préparer les données d'entrée pour la prédiction
    
    Parameters:
    -----------
    customer : CustomerFeatures
        Données du client
        
    Returns:
    --------
    pd.DataFrame
    """
    # Convertir en DataFrame
    data = {
        'credit_score': customer.credit_score,
        'country': customer.country,
        'gender': customer.gender,
        'age': customer.age,
        'tenure': customer.tenure,
        'balance': customer.balance,
        'products_number': customer.products_number,
        'credit_card': customer.credit_card,
        'active_member': customer.active_member,
        'estimated_salary': customer.estimated_salary,
        'customer_id': customer.customer_id
    }
    
    # Ajouter colonnes optionnelles
    if customer.customer_id is not None:
        data['customer_id'] = customer.customer_id
    
    # Ajouter colonnes nécessaires pour le preprocessing
    data['RowNumber'] = 0
    
    return pd.DataFrame([data])


@app.on_event("startup")
async def startup_event():
    """
    Événement exécuté au démarrage de l'API
    """
    logger.info("=" * 60)
    logger.info("🚀 DÉMARRAGE DE L'API CHURN PREDICTION")
    logger.info("=" * 60)
    
    success = load_model_and_preprocessor()
    
    if not success:
        logger.error("❌ Échec du chargement du modèle")
        logger.error("L'API démarrera mais les prédictions échoueront")
    else:
        logger.info("✅ API prête à recevoir des requêtes")


@app.get("/", tags=["Root"])
async def root():
    """
    Endpoint racine
    """
    return {
        "message": "Bank Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check - Vérifier l'état du service
    """
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_metadata.get('timestamp', 'unknown') if model_metadata else 'unknown',
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.now().isoformat()
    )


@app.get("/metrics", response_model=ModelMetrics, tags=["Model"])
async def get_model_metrics():
    """
    Obtenir les métriques du modèle en production
    """
    if model_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Métadonnées du modèle non disponibles"
        )
    
    metrics = model_metadata.get('metrics', {})
    
    return ModelMetrics(
        model_name=model_metadata.get('model_name', 'Unknown'),
        accuracy=metrics.get('accuracy', 0.0),
        precision=metrics.get('precision', 0.0),
        recall=metrics.get('recall', 0.0),
        f1_score=metrics.get('f1_score', 0.0),
        roc_auc=metrics.get('roc_auc', 0.0),
        training_date=model_metadata.get('timestamp', 'unknown')
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(customer: CustomerFeatures):
    """
    Prédire le churn pour un client individuel
    
    Parameters:
    -----------
    customer : CustomerFeatures
        Données du client
        
    Returns:
    --------
    PredictionResponse
        Prédiction et probabilité de churn
    """
    start_pred = time.time()
    
    try:
        # Vérifier que le modèle est chargé
        if model is None or preprocessor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modèle non chargé"
            )
        
        # Préparer les données
        input_df = prepare_input_data(customer)
        
        # Preprocessing
        input_processed = preprocessor.transform(input_df)
        
        # Prédiction
        prediction = int(model.predict(input_processed)[0])
        probability = float(model.predict_proba(input_processed)[0, 1])
        
        # Calculer le niveau de risque
        risk_level = calculate_risk_level(probability)
        
        # Temps de traitement
        processing_time = (time.time() - start_pred) * 1000  # en ms
        
        logger.info(f"Prédiction effectuée en {processing_time:.2f}ms - Customer: {customer.customer_id} - Churn: {prediction}")
        
        return PredictionResponse(
            customer_id=customer.customer_id,
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=risk_level,
            confidence=round(probability if prediction == 1 else 1 - probability, 4),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Prédictions pour plusieurs clients (batch)
    
    Parameters:
    -----------
    request : BatchPredictionRequest
        Liste de clients
        
    Returns:
    --------
    BatchPredictionResponse
        Prédictions pour tous les clients
    """
    start_batch = time.time()
    
    try:
        # Vérifier que le modèle est chargé
        if model is None or preprocessor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Modèle non chargé"
            )
        
        # Prédictions pour chaque client
        predictions = []
        high_risk_count = 0
        
        for customer in request.customers:
            # Préparer données
            input_df = prepare_input_data(customer)
            input_processed = preprocessor.transform(input_df)
            
            # Prédiction
            prediction = int(model.predict(input_processed)[0])
            probability = float(model.predict_proba(input_processed)[0, 1])
            risk_level = calculate_risk_level(probability)
            
            if risk_level == "High":
                high_risk_count += 1
            
            predictions.append(PredictionResponse(
                customer_id=customer.customer_id,
                churn_prediction=prediction,
                churn_probability=round(probability, 4),
                risk_level=risk_level,
                confidence=round(probability if prediction == 1 else 1 - probability, 4),
                timestamp=datetime.now().isoformat()
            ))
        
        processing_time = (time.time() - start_batch) * 1000  # ms
        
        logger.info(f"Batch de {len(predictions)} prédictions effectué en {processing_time:.2f}ms")
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=high_risk_count,
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du batch : {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du batch : {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Gestionnaire global des exceptions
    """
    logger.error(f"Exception non gérée : {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Lancer le serveur
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )