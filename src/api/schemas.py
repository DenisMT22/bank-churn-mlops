"""
Schémas Pydantic pour l'API de prédiction de Churn
==================================================

"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class CustomerFeatures(BaseModel):
    credit_score: int = Field(..., ge=300, le=900, description="Score de crédit (300-900)")
    country: Literal['France', 'Germany', 'Spain'] = Field(..., description="Pays du client")
    gender: Literal['Male', 'Female'] = Field(..., description="Genre du client")
    age: int = Field(..., ge=18, le=100, description="Âge du client")
    tenure: int = Field(..., ge=0, le=10, description="Ancienneté (années)")
    balance: float = Field(..., ge=0, description="Solde du compte")
    products_number: int = Field(..., ge=1, le=4, description="Nombre de produits")
    credit_card: int = Field(..., ge=0, le=1, description="Possède carte de crédit (0/1)")
    active_member: int = Field(..., ge=0, le=1, description="Membre actif (0/1)")
    estimated_salary: float = Field(..., ge=0, description="Salaire estimé")
    customer_id: Optional[int] = Field(None, description="ID du client")
    
    class Config:
        schema_extra = {
            "example": {
                "credit_score": 650,
                "country": "France",
                "gender": "Female",
                "age": 35,
                "tenure": 5,
                "balance": 125000.0,
                "products_number": 2,
                "credit_card": 1,
                "active_member": 1,
                "estimated_salary": 50000.0,
                "customer_id": 15634602
            }
        }


class PredictionResponse(BaseModel):
    customer_id: Optional[int] = Field(None, description="ID du client")
    churn_prediction: int = Field(..., description="Prédiction (0=Non-Churn, 1=Churn)")
    churn_probability: float = Field(..., ge=0, le=1, description="Probabilité de churn (0-1)")
    risk_level: str = Field(..., description="Niveau de risque (Low/Medium/High)")
    confidence: float = Field(..., ge=0, le=1, description="Niveau de confiance")
    timestamp: str = Field(..., description="Timestamp de la prédiction")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": 15634602,
                "churn_prediction": 1,
                "churn_probability": 0.78,
                "risk_level": "High",
                "confidence": 0.78,
                "timestamp": "2025-11-19T14:30:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., min_items=1, max_items=1000)
    
    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    {
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
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    processing_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str = Field(..., description="État du service")
    model_loaded: bool = Field(..., description="Modèle chargé")
    model_version: str = Field(..., description="Version du modèle")
    uptime_seconds: float = Field(..., description="Temps de fonctionnement")
    timestamp: str


class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    training_date: str
    
    class Config:
        schema_extra = {
            "example": {
                "model_name": "XGBoost",
                "accuracy": 0.8642,
                "precision": 0.7891,
                "recall": 0.8123,
                "f1_score": 0.8005,
                "roc_auc": 0.9234,
                "training_date": "2025-11-19"
            }
        }


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Message d'erreur")
    detail: Optional[str] = Field(None, description="Détails supplémentaires")
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "credit_score must be between 300 and 900",
                "timestamp": "2025-11-19T14:30:00"
            }
        }
