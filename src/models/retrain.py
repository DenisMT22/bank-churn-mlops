"""
Pipeline de Réentraînement Automatique
=======================================

Ce script automatise le réentraînement du modèle :
- Télécharge nouvelles données
- Vérifie la qualité des données
- Réentraîne le modèle
- Compare avec l'ancien modèle
- Déploie si performance supérieure

"""

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# Import des modules custom
sys.path.append('..')
from models.preprocessing import DataPreprocessor, prepare_data_for_training
from models.train import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoRetrainer:
    """
    Classe pour automatiser le réentraînement du modèle
    """
    
    def __init__(
        self,
        data_path: str,
        models_dir: str = '../models',
        min_improvement: float = 0.02,  # Amélioration minimale de 2%
        validation_metric: str = 'recall'
    ):
        """
        Initialisation
        
        Parameters:
        -----------
        data_path : str
            Chemin des nouvelles données
        models_dir : str
            Répertoire des modèles
        min_improvement : float
            Amélioration minimale requise pour déployer
        validation_metric : str
            Métrique pour la validation
        """
        self.data_path = data_path
        self.models_dir = Path(models_dir)
        self.min_improvement = min_improvement
        self.validation_metric = validation_metric
        
        self.trained_models_dir = self.models_dir / 'trained'
        self.trained_models_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_model = None
        self.current_preprocessor = None
        self.current_metadata = None
        
        logger.info("✅ AutoRetrainer initialisé")
    
    def load_current_model(self) -> bool:
        """
        Charger le modèle actuellement en production
        
        Returns:
        --------
        bool : Succès du chargement
        """
        try:
            model_path = self.trained_models_dir / 'model_latest.pkl'
            preprocessor_path = self.models_dir / 'preprocessor.pkl'
            metadata_path = self.models_dir / 'model_metadata.json'
            
            logger.info("📥 Chargement du modèle actuel...")
            
            if model_path.exists():
                self.current_model = joblib.load(model_path)
                logger.info("✅ Modèle chargé")
            else:
                logger.warning("⚠️ Pas de modèle existant")
                return False
            
            if preprocessor_path.exists():
                self.current_preprocessor = joblib.load(preprocessor_path)
                logger.info("✅ Preprocessor chargé")
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.current_metadata = json.load(f)
                logger.info("✅ Métadonnées chargées")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle : {e}")
            return False
    
    def validate_data_quality(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Vérifier la qualité des données
        
        Parameters:
        -----------
        df : pd.DataFrame
            Données à valider
            
        Returns:
        --------
        tuple : (is_valid, message)
        """
        logger.info("🔍 Validation de la qualité des données...")
        
        # 1. Vérifier taille minimale
        min_rows = 1000
        if len(df) < min_rows:
            return False, f"Trop peu de données : {len(df)} < {min_rows}"
        
        # 2. Vérifier colonnes requises
        required_cols = [
            'credit_score', 'country', 'gender', 'age', 'tenure',
            'balance', 'products_number', 'credit_card', 'active_member',
            'estimated_salary', 'churn'
        ]
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            return False, f"Colonnes manquantes : {missing_cols}"
        
        # 3. Vérifier valeurs manquantes
        missing_pct = (df.isnull().sum() / len(df)) * 100
        if (missing_pct > 5).any():
            high_missing = missing_pct[missing_pct > 5]
            return False, f"Trop de valeurs manquantes : {high_missing.to_dict()}"
        
        # 4. Vérifier distribution de la cible
        target_dist = df['churn'].value_counts(normalize=True)
        if target_dist.min() < 0.05:  # Au moins 5% de chaque classe
            return False, f"Déséquilibre extrême de la cible : {target_dist.to_dict()}"
        
        # 5. Vérifier plages de valeurs
        if (df['credit_score'] < 300).any() or (df['credit_score'] > 900).any():
            return False, "credit_score hors plage valide (300-900)"
        
        if (df['age'] < 18).any() or (df['age'] > 100).any():
            return False, "age hors plage valide (18-100)"
        
        logger.info("✅ Données validées avec succès")
        return True, "OK"
    
    def train_new_model(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        use_smote: bool = True
    ) -> Tuple[object, Dict]:
        """
        Entraîner un nouveau modèle
        
        Parameters:
        -----------
        X_train, X_test : array-like
            Features
        y_train, y_test : array-like
            Targets
        use_smote : bool
            Utiliser SMOTE
            
        Returns:
        --------
        tuple : (model, metrics)
        """
        logger.info("🎯 Entraînement du nouveau modèle...")
        
        # Utiliser ModelTrainer
        trainer = ModelTrainer(random_state=42)
        trainer.define_models()
        trainer.train_and_evaluate(X_train, X_test, y_train, y_test, use_smote=use_smote)
        trainer.select_best_model(metric=self.validation_metric)
        
        new_model = trainer.best_model
        new_metrics = trainer.results[trainer.best_model_name]['metrics']
        
        logger.info(f"✅ Nouveau modèle entraîné : {trainer.best_model_name}")
        logger.info(f"   {self.validation_metric} = {new_metrics[self.validation_metric]:.4f}")
        
        return new_model, new_metrics, trainer.best_model_name
    
    def compare_models(
        self,
        new_metrics: Dict,
        current_metrics: Dict
    ) -> Tuple[bool, str]:
        """
        Comparer le nouveau modèle avec l'actuel
        
        Parameters:
        -----------
        new_metrics : dict
            Métriques du nouveau modèle
        current_metrics : dict
            Métriques du modèle actuel
            
        Returns:
        --------
        tuple : (should_deploy, reason)
        """
        logger.info("📊 Comparaison des modèles...")
        
        metric = self.validation_metric
        
        new_score = new_metrics.get(metric, 0)
        current_score = current_metrics.get(metric, 0)
        
        improvement = new_score - current_score
        improvement_pct = (improvement / current_score) * 100 if current_score > 0 else 0
        
        logger.info(f"   Modèle actuel : {metric} = {current_score:.4f}")
        logger.info(f"   Nouveau modèle : {metric} = {new_score:.4f}")
        logger.info(f"   Amélioration : {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        # Vérifier seuils
        if new_score < 0.70:  # Seuil minimum absolu
            reason = f"Performance insuffisante : {metric} = {new_score:.4f} < 0.70"
            logger.warning(f"❌ {reason}")
            return False, reason
        
        if improvement < self.min_improvement:
            reason = f"Amélioration insuffisante : {improvement:.4f} < {self.min_improvement}"
            logger.warning(f"❌ {reason}")
            return False, reason
        
        # Vérifier que les autres métriques ne se dégradent pas trop
        for other_metric in ['precision', 'recall', 'f1_score']:
            if other_metric == metric:
                continue
            
            new_val = new_metrics.get(other_metric, 0)
            curr_val = current_metrics.get(other_metric, 0)
            
            if new_val < curr_val * 0.95:  # Tolérance de 5% de dégradation
                reason = f"Dégradation de {other_metric} : {new_val:.4f} vs {curr_val:.4f}"
                logger.warning(f"⚠️ {reason}")
                # Ne pas bloquer mais avertir
        
        reason = f"Amélioration significative : {metric} {improvement:+.4f}"
        logger.info(f"✅ {reason}")
        return True, reason
    
    def deploy_model(
        self,
        new_model,
        new_preprocessor,
        new_metadata: Dict
    ) -> bool:
        """
        Déployer le nouveau modèle
        
        Parameters:
        -----------
        new_model : sklearn model
            Nouveau modèle
        new_preprocessor : DataPreprocessor
            Nouveau preprocessor
        new_metadata : dict
            Métadonnées
            
        Returns:
        --------
        bool : Succès du déploiement
        """
        logger.info("🚀 Déploiement du nouveau modèle...")
        
        try:
            # 1. Sauvegarder l'ancien modèle (backup)
            if (self.trained_models_dir / 'model_latest.pkl').exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.trained_models_dir / f'model_backup_{timestamp}.pkl'
                
                import shutil
                shutil.copy(
                    self.trained_models_dir / 'model_latest.pkl',
                    backup_path
                )
                logger.info(f"✅ Backup créé : {backup_path}")
            
            # 2. Sauvegarder le nouveau modèle
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = self.trained_models_dir / f'model_{timestamp}.pkl'
            joblib.dump(new_model, model_path)
            logger.info(f"✅ Modèle sauvegardé : {model_path}")
            
            # Sauvegarder comme "latest"
            joblib.dump(new_model, self.trained_models_dir / 'model_latest.pkl')
            logger.info("✅ Modèle déployé comme 'latest'")
            
            # 3. Sauvegarder le preprocessor
            preprocessor_path = self.models_dir / 'preprocessor.pkl'
            joblib.dump(new_preprocessor, preprocessor_path)
            logger.info(f"✅ Preprocessor sauvegardé")
            
            # 4. Sauvegarder les métadonnées
            metadata_path = self.models_dir / 'model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(new_metadata, f, indent=4)
            logger.info(f"✅ Métadonnées sauvegardées")
            
            # 5. Uploader vers GCS (si configuré)
            try:
                from google.cloud import storage
                
                project_id = os.getenv('PROJECT_ID')
                if project_id:
                    client = storage.Client()
                    bucket = client.bucket(f"{project_id}-models")
                    
                    # Upload modèle
                    blob = bucket.blob('model_latest.pkl')
                    blob.upload_from_filename(str(self.trained_models_dir / 'model_latest.pkl'))
                    
                    # Upload preprocessor
                    blob = bucket.blob('preprocessor.pkl')
                    blob.upload_from_filename(str(preprocessor_path))
                    
                    # Upload metadata
                    blob = bucket.blob('model_metadata.json')
                    blob.upload_from_filename(str(metadata_path))
                    
                    logger.info("✅ Modèle uploadé vers GCS")
            except Exception as e:
                logger.warning(f"⚠️ Upload GCS échoué : {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du déploiement : {e}")
            return False
    
    def run_retraining_pipeline(self) -> Dict:
        """
        Exécuter le pipeline complet de réentraînement
        
        Returns:
        --------
        dict : Résumé de l'exécution
        """
        start_time = datetime.now()
        
        logger.info("\n" + "=" * 80)
        logger.info(" " * 25 + "🔄 PIPELINE DE RÉENTRAÎNEMENT")
        logger.info("=" * 80)
        
        summary = {
            'start_time': start_time.isoformat(),
            'success': False,
            'model_deployed': False,
            'reason': '',
            'metrics': {}
        }
        
        try:
            # 1. Charger le modèle actuel
            has_current_model = self.load_current_model()
            
            # 2. Charger et valider les données
            logger.info("\n📥 Chargement des données...")
            df = pd.read_csv(self.data_path)
            logger.info(f"✅ Données chargées : {df.shape}")
            
            is_valid, message = self.validate_data_quality(df)
            if not is_valid:
                summary['reason'] = f"Données invalides : {message}"
                logger.error(f"❌ {summary['reason']}")
                return summary
            
            # 3. Préparer les données
            logger.info("\n🔧 Préparation des données...")
            X_train, X_test, y_train, y_test, new_preprocessor = prepare_data_for_training(
                data_path=self.data_path,
                target_col='churn',
                test_size=0.2,
                random_state=42
            )
            
            # 4. Entraîner le nouveau modèle
            logger.info("\n🎯 Entraînement du nouveau modèle...")
            new_model, new_metrics, model_name = self.train_new_model(
                X_train, X_test, y_train, y_test
            )
            
            summary['metrics']['new_model'] = {k: float(v) for k, v in new_metrics.items()}
            
            # 5. Comparer avec l'ancien modèle
            if has_current_model and self.current_metadata:
                logger.info("\n📊 Comparaison des modèles...")
                current_metrics = self.current_metadata.get('metrics', {})
                summary['metrics']['current_model'] = current_metrics
                
                should_deploy, reason = self.compare_models(new_metrics, current_metrics)
                
                if not should_deploy:
                    summary['reason'] = reason
                    summary['success'] = True
                    logger.info(f"\n✅ Pipeline terminé : {reason}")
                    return summary
            else:
                logger.info("ℹ️ Pas de modèle actuel, déploiement automatique")
                should_deploy = True
                reason = "Premier déploiement"
            
            # 6. Déployer le nouveau modèle
            logger.info("\n🚀 Déploiement du nouveau modèle...")
            
            new_metadata = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'metrics': {k: float(v) for k, v in new_metrics.items()},
                'training_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'retraining_reason': reason
            }
            
            deployment_success = self.deploy_model(
                new_model,
                new_preprocessor,
                new_metadata
            )
            
            if deployment_success:
                summary['model_deployed'] = True
                summary['reason'] = reason
                summary['success'] = True
                logger.info("\n✅ Modèle déployé avec succès")
            else:
                summary['reason'] = "Échec du déploiement"
                logger.error("\n❌ Échec du déploiement")
            
        except Exception as e:
            summary['reason'] = f"Erreur : {str(e)}"
            logger.error(f"\n❌ Erreur dans le pipeline : {e}", exc_info=True)
        
        finally:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            summary['end_time'] = end_time.isoformat()
            summary['duration_seconds'] = duration
            
            # Sauvegarder le résumé
            summary_path = self.models_dir / 'retraining_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=4)
            
            logger.info("\n" + "=" * 80)
            logger.info(f"Pipeline terminé en {duration:.2f}s")
            logger.info(f"Résumé sauvegardé : {summary_path}")
            logger.info("=" * 80)
        
        return summary


def main():
    """
    Fonction principale
    """
    import os
    
    # Configuration
    data_path = os.getenv('DATA_PATH', '/Users/denismutombotshituka/bank-churn-mlops/data/raw/Bank_Churn_Prediction.csv')
    models_dir = os.getenv('MODELS_DIR', '../../models')
    
    # Créer et exécuter le retrainer
    retrainer = AutoRetrainer(
        data_path=data_path,
        models_dir=models_dir,
        min_improvement=0.02,
        validation_metric='recall'
    )
    
    summary = retrainer.run_retraining_pipeline()
    
    # Afficher résumé
    print("\n" + "=" * 80)
    print("RÉSUMÉ DU RÉENTRAÎNEMENT")
    print("=" * 80)
    print(f"Succès : {summary['success']}")
    print(f"Modèle déployé : {summary['model_deployed']}")
    print(f"Raison : {summary['reason']}")
    print(f"Durée : {summary['duration_seconds']:.2f}s")
    
    if 'new_model' in summary['metrics']:
        print("\nMétriques nouveau modèle :")
        for k, v in summary['metrics']['new_model'].items():
            print(f"  {k}: {v:.4f}")
    
    return summary


if __name__ == "__main__":
    import os
    os.environ['DATA_PATH'] = '/Users/denismutombotshituka/bank-churn-mlops/data/raw/Bank_Churn_Prediction.csv'
    main()