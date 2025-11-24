"""
Tests Unitaires pour l'API de Prédiction de Churn
==================================================

"""

import pytest
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Ajouter le chemin src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from api.main import app

# Client de test
client = TestClient(app)


class TestHealthEndpoint:
    """Tests pour l'endpoint /health"""
    
    def test_health_check_success(self):
        """Test que le health check retourne 200"""
        response = client.get("/health")
        assert response.status_code == 200
        
    def test_health_check_response_structure(self):
        """Test la structure de la réponse health"""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data
    
    def test_health_check_uptime(self):
        """Test que l'uptime est positif"""
        response = client.get("/health")
        data = response.json()
        
        assert data["uptime_seconds"] >= 0


class TestRootEndpoint:
    """Tests pour l'endpoint racine /"""
    
    def test_root_returns_200(self):
        """Test que la racine retourne 200"""
        response = client.get("/")
        assert response.status_code == 200
    
    def test_root_response_content(self):
        """Test le contenu de la réponse racine"""
        response = client.get("/")
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"


class TestMetricsEndpoint:
    """Tests pour l'endpoint /metrics"""
    
    def test_metrics_endpoint_exists(self):
        """Test que l'endpoint metrics existe"""
        response = client.get("/metrics")
        # Peut être 200 ou 503 selon si le modèle est chargé
        assert response.status_code in [200, 503]
    
    def test_metrics_response_structure_when_available(self):
        """Test la structure si les métriques sont disponibles"""
        response = client.get("/metrics")
        
        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "accuracy" in data
            assert "precision" in data
            assert "recall" in data
            assert "f1_score" in data
            assert "roc_auc" in data


class TestPredictionEndpoint:
    """Tests pour l'endpoint /predict"""
    
    @pytest.fixture
    def valid_customer_data(self):
        """Fixture avec des données client valides"""
        return {
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
            "customer_id": 15634602,
            "Surname": "Dupont"
        }
    
    def test_predict_endpoint_exists(self, valid_customer_data):
        """Test que l'endpoint predict existe"""
        response = client.post("/predict", json=valid_customer_data)
        # 200 si modèle chargé, 503 sinon
        assert response.status_code in [200, 503]
    
    def test_predict_with_valid_data(self, valid_customer_data):
        """Test prédiction avec données valides"""
        response = client.post("/predict", json=valid_customer_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "churn_prediction" in data
            assert "churn_probability" in data
            assert "risk_level" in data
            assert data["churn_prediction"] in [0, 1]
            assert 0 <= data["churn_probability"] <= 1
            assert data["risk_level"] in ["Low", "Medium", "High"]
    
    def test_predict_with_invalid_credit_score(self):
        """Test avec CreditScore invalide"""
        invalid_data = {
            "credit_score": 1500,  # Invalide (max 900)
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
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_invalid_geography(self):
        """Test avec Geography invalide"""
        invalid_data = {
            "credit_score": 650,
            "country": "USA",  # Pas dans la liste
            "gender": "Female",
            "age": 35,
            "tenure": 5,
            "balance": 125000.0,
            "products_number": 2,
            "credit_card": 1,
            "active_member": 1,
            "estimated_salary": 50000.0
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_predict_with_missing_fields(self):
        """Test avec champs manquants"""
        incomplete_data = {
            "credit_score": 650,
            "country": "France"
            # Champs manquants
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422


class TestBatchPredictionEndpoint:
    """Tests pour l'endpoint /predict/batch"""
    
    @pytest.fixture
    def valid_batch_data(self):
        """Fixture avec données batch valides"""
        return {
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
                },
                {
                    "credit_score": 700,
                    "country": "Germany",
                    "gender": "Male",
                    "age": 42,
                    "tenure": 8,
                    "balance": 0.0,
                    "products_number": 1,
                    "credit_card": 0,
                    "active_member": 0,
                    "estimated_salary": 75000.0
                }
            ]
        }
    
    def test_batch_predict_endpoint_exists(self, valid_batch_data):
        """Test que l'endpoint batch existe"""
        response = client.post("/predict/batch", json=valid_batch_data)
        assert response.status_code in [200, 503]
    
    def test_batch_predict_with_valid_data(self, valid_batch_data):
        """Test batch avec données valides"""
        response = client.post("/predict/batch", json=valid_batch_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_customers" in data
            assert "high_risk_count" in data
            assert len(data["predictions"]) == 2
            assert data["total_customers"] == 2
    
    def test_batch_predict_empty_list(self):
        """Test batch avec liste vide"""
        empty_batch = {"customers": []}
        
        response = client.post("/predict/batch", json=empty_batch)
        assert response.status_code == 422  # Validation error


class TestAPIDocumentation:
    """Tests pour la documentation de l'API"""
    
    def test_openapi_schema_exists(self):
        """Test que le schéma OpenAPI est disponible"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
    
    def test_swagger_ui_exists(self):
        """Test que Swagger UI est accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc_exists(self):
        """Test que ReDoc est accessible"""
        response = client.get("/redoc")
        assert response.status_code == 200


class TestErrorHandling:
    """Tests pour la gestion des erreurs"""
    
    def test_invalid_endpoint(self):
        """Test endpoint inexistant"""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404
    
    def test_invalid_method(self):
        """Test méthode HTTP invalide"""
        response = client.delete("/predict")  # DELETE non supporté
        assert response.status_code == 405


# Tests de performance (optionnels)
class TestPerformance:
    """Tests de performance basiques"""
    
    @pytest.fixture
    def valid_customer_data(self):
        return {
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
    
    def test_prediction_latency(self, valid_customer_data):
        """Test que la prédiction est rapide"""
        import time
        
        start = time.time()
        response = client.post("/predict", json=valid_customer_data)
        latency = (time.time() - start) * 1000  # ms
        
        if response.status_code == 200:
            # La prédiction devrait prendre moins de 1 seconde
            assert latency < 1000, f"Latency trop élevée: {latency:.2f}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])