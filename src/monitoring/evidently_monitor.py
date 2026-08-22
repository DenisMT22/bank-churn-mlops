"""
Monitoring ML avec Evidently AI
================================

Ce module utilise Evidently pour détecter :
- Dérive des données (Data Drift)
- Dérive du modèle (Model Drift)
- Dégradation des performances

"""

import pandas as pd
from datetime import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

# Import du module de configuration des chemins. Le double essai permet
# d'executer ce fichier aussi bien comme script que comme module importe.
try:
    from src.utils import config
except ImportError:  # pragma: no cover - depend du mode d'execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils import config

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    ClassificationPreset
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
    TestAccuracyScore,
    TestPrecisionScore,
    TestRecallScore,
    TestF1Score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLMonitor:
    """
    Classe pour monitorer les performances et la dérive du modèle
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        model,
        preprocessor,
        target_col: str = 'churn',
        prediction_col: str = 'prediction',
        output_dir: str = None
    ):
        """
        Initialisation du moniteur
        
        Parameters:
        -----------
        reference_data : pd.DataFrame
            Données de référence (train set)
        model : sklearn model
            Modèle entraîné
        preprocessor : DataPreprocessor
            Preprocessor entraîné
        target_col : str
            Nom de la colonne cible
        prediction_col : str
            Nom de la colonne de prédiction
        output_dir : str
            Répertoire de sortie des rapports
        """
        self.reference_data = reference_data.copy()
        self.model = model
        self.preprocessor = preprocessor
        self.target_col = target_col
        self.prediction_col = prediction_col
        # Chemin resolu depuis la racine du projet : un chemin relatif
        # dependrait du repertoire courant et creerait un dossier
        # parasite selon l'endroit d'ou le script est lance.
        self.output_dir = Path(output_dir) if output_dir else config.MONITORING_REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration des colonnes pour Evidently
        self.column_mapping = ColumnMapping(
            target=target_col,
            prediction=prediction_col,
            numerical_features=[
                'credit_score', 'age', 'tenure', 'balance',
                'products_number', 'estimated_salary'
            ],
            categorical_features=[
                'country', 'gender', 'credit_card', 'active_member'
            ]
        )
        
        # Ajouter les prédictions aux données de référence
        if self.prediction_col not in self.reference_data.columns:
            self._add_predictions_to_reference()
        
        logger.info("✅ MLMonitor initialisé")
    
    def _add_predictions_to_reference(self):
        """Ajouter les prédictions aux données de référence"""
        X_ref = self.reference_data.drop(columns=[self.target_col])
        X_ref_processed = self.preprocessor.transform(X_ref)
        predictions = self.model.predict(X_ref_processed)
        self.reference_data[self.prediction_col] = predictions
        logger.info("✅ Prédictions ajoutées aux données de référence")
    
    def generate_data_drift_report(
        self,
        current_data: pd.DataFrame,
        save_html: bool = True
    ) -> Dict:
        """
        Générer rapport de dérive des données
        
        Parameters:
        -----------
        current_data : pd.DataFrame
            Données actuelles (production)
        save_html : bool
            Sauvegarder en HTML
            
        Returns:
        --------
        dict : Résumé de la dérive
        """
        logger.info("📊 Génération du rapport Data Drift...")
        
        # Préparer les données actuelles
        current_data = current_data.copy()
        
        # Ajouter prédictions si nécessaire
        if self.prediction_col not in current_data.columns and self.target_col in current_data.columns:
            X_curr = current_data.drop(columns=[self.target_col])
            X_curr_processed = self.preprocessor.transform(X_curr)
            current_data[self.prediction_col] = self.model.predict(X_curr_processed)
        
        # Créer le rapport
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Sauvegarder HTML
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = self.output_dir / f"data_drift_report_{timestamp}.html"
            report.save_html(str(html_path))
            logger.info(f"✅ Rapport HTML sauvegardé : {html_path}")
        
        # Extraire les résultats
        results = report.as_dict()
        logger.info(f"Resultats bruts drift: {results['metrics'][0]['result']}")
        res = results['metrics'][0]['result']
        
        # Résumé
        drift_summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset_drift_detected': res.get('dataset_drift', False),
            'number_of_drifted_columns': res.get('number_of_drifted_columns', 0),
            'share_of_drifted_columns': res.get('share_of_drifted_columns', 0.0),
            'drifted_columns': []
        }
        
        # Détails des colonnes qui ont dérivé
        drift_cols = res.get('drift_by_columns', {})
        if not drift_cols:
          drift_cols = res.get('drift_columns', {})  
        for col_name, col_result in drift_cols.items():
          if col_result.get('drift_detected', False):
              drift_summary['drifted_columns'].append({
                'column': col_name,
                'drift_score': col_result.get('drift_score', 0),
                'stattest_name': col_result.get('stattest_name', 'unknown')
            })
        logger.info(f"📈 Dérive détectée : {drift_summary['dataset_drift_detected']}")
        logger.info(f"📈 Colonnes dérivées : {drift_summary['number_of_drifted_columns']}")
        return drift_summary
    
    def generate_model_performance_report(
        self,
        current_data: pd.DataFrame,
        save_html: bool = True
    ) -> Dict:
        """
        Générer rapport de performance du modèle
        
        Parameters:
        -----------
        current_data : pd.DataFrame
            Données actuelles avec target et prédictions
        save_html : bool
            Sauvegarder en HTML
            
        Returns:
        --------
        dict : Métriques de performance
        """
        logger.info("📊 Génération du rapport Model Performance...")
        
        # Vérifier que target et predictions sont présents
        if self.target_col not in current_data.columns:
            raise ValueError(f"Colonne {self.target_col} manquante")
        
        current_data = current_data.copy()
        
        # Ajouter prédictions si nécessaire
        if self.prediction_col not in current_data.columns:
            X_curr = current_data.drop(columns=[self.target_col])
            X_curr_processed = self.preprocessor.transform(X_curr)
            current_data[self.prediction_col] = self.model.predict(X_curr_processed)
        
        # Créer le rapport
        report = Report(metrics=[
            ClassificationPreset(),
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Sauvegarder HTML
        if save_html:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = self.output_dir / f"model_performance_report_{timestamp}.html"
            report.save_html(str(html_path))
            logger.info(f"✅ Rapport HTML sauvegardé : {html_path}")
        
        # Extraire les métriques
        results = report.as_dict()
        
        performance_summary = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': {},
            'reference_metrics': {}
        }
        
        # Extraire métriques (structure peut varier selon version Evidently)
        try:
            for metric in results['metrics']:
                if 'result' in metric:
                    if 'current' in metric['result']:
                        perf_current = metric['result']['current']
                        if isinstance(perf_current, dict):
                            performance_summary['current_metrics'].update(perf_current)
                    
                    if 'reference' in metric['result']:
                        perf_ref = metric['result']['reference']
                        if isinstance(perf_ref, dict):
                            performance_summary['reference_metrics'].update(perf_ref)
        except Exception as e:
            logger.warning(f"⚠️ Extraction métriques incomplète : {e}")
        
        logger.info(f"📈 Métriques actuelles : {performance_summary['current_metrics']}")
        
        return performance_summary
    
    def run_test_suite(
        self,
        current_data: pd.DataFrame,
        thresholds: Optional[Dict] = None
    ) -> Dict:
        """
        Exécuter une suite de tests automatisés
        
        Parameters:
        -----------
        current_data : pd.DataFrame
            Données actuelles
        thresholds : dict, optional
            Seuils personnalisés
            
        Returns:
        --------
        dict : Résultats des tests
        """
        logger.info("🧪 Exécution de la suite de tests...")
        
        if thresholds is None:
            thresholds = {
                'max_share_drifted_columns': 0.3,  # Max 30% colonnes dérivées
                'min_accuracy': 0.75,
                'min_precision': 0.60,
                'min_recall': 0.70,
                'min_f1': 0.65
            }
        
        # Préparer les données
        current_data = current_data.copy()
        if self.prediction_col not in current_data.columns and self.target_col in current_data.columns:
            X_curr = current_data.drop(columns=[self.target_col])
            X_curr_processed = self.preprocessor.transform(X_curr)
            current_data[self.prediction_col] = self.model.predict(X_curr_processed)
        
        # Créer la suite de tests
        test_suite = TestSuite(tests=[
            TestShareOfDriftedColumns(lt=thresholds['max_share_drifted_columns']),
            TestNumberOfDriftedColumns(lt=5),
        ])
        
        # Ajouter tests de performance si target disponible
        if self.target_col in current_data.columns:
            test_suite._tests.extend([
                TestAccuracyScore(gte=thresholds['min_accuracy']),
                TestPrecisionScore(gte=thresholds['min_precision']),
                TestRecallScore(gte=thresholds['min_recall']),
                TestF1Score(gte=thresholds['min_f1'])
            ])
        
        test_suite.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping
        )
        
        # Sauvegarder résultats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.output_dir / f"test_suite_{timestamp}.html"
        test_suite.save_html(str(html_path))
        logger.info(f"✅ Suite de tests sauvegardée : {html_path}")
        
        # Analyser résultats
        results = test_suite.as_dict()
        
        test_summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results['tests']),
            'passed_tests': sum(1 for t in results['tests'] if t['status'] == 'SUCCESS'),
            'failed_tests': sum(1 for t in results['tests'] if t['status'] == 'FAIL'),
            'all_tests_passed': all(t['status'] == 'SUCCESS' for t in results['tests']),
            'failed_test_details': []
        }
        
        # Détails des tests échoués
        for test in results['tests']:
            if test['status'] == 'FAIL':
                test_summary['failed_test_details'].append({
                    'test_name': test['name'],
                    'description': test.get('description', ''),
                    'status': test['status']
                })
        
        logger.info(f"✅ Tests réussis : {test_summary['passed_tests']}/{test_summary['total_tests']}")
        if not test_summary['all_tests_passed']:
            logger.warning(f"⚠️ Tests échoués : {test_summary['failed_tests']}")
        
        return test_summary
    
    def check_need_for_retraining(
        self,
        current_data: pd.DataFrame,
        auto_threshold: bool = True
    ) -> Tuple[bool, str, Dict]:
        """
        Vérifier si un réentraînement est nécessaire
        
        Parameters:
        -----------
        current_data : pd.DataFrame
            Données de production actuelles
        auto_threshold : bool
            Utiliser seuils automatiques
            
        Returns:
        --------
        tuple : (needs_retraining, reason, details)
        """
        logger.info("🔍 Vérification besoin de réentraînement...")
        
        # 1. Vérifier dérive des données
        drift_summary = self.generate_data_drift_report(current_data, save_html=False)
        
        if drift_summary['dataset_drift_detected']:
            reason = f"Dérive détectée sur {drift_summary['number_of_drifted_columns']} colonnes"
            logger.warning(f"⚠️ {reason}")
            return True, reason, drift_summary
        
        # 2. Vérifier performances si target disponible
        if self.target_col in current_data.columns:
            perf_summary = self.generate_model_performance_report(current_data, save_html=False)
            
            # Vérifier si recall a chuté (métrique clé pour le churn)
            current_recall = perf_summary['current_metrics'].get('recall', 1.0)
            reference_recall = perf_summary['reference_metrics'].get('recall', 1.0)
            
            if current_recall < 0.70:  # Seuil minimum
                reason = f"Recall trop faible : {current_recall:.3f} < 0.70"
                logger.warning(f"⚠️ {reason}")
                return True, reason, perf_summary
            
            # Vérifier dégradation significative (>10%)
            if current_recall < reference_recall * 0.90:
                reason = f"Dégradation du recall : {current_recall:.3f} vs {reference_recall:.3f}"
                logger.warning(f"⚠️ {reason}")
                return True, reason, perf_summary
        
        logger.info("✅ Pas de besoin de réentraînement détecté")
        return False, "Performances stables", {}
    
    def save_monitoring_summary(
        self,
        current_data: pd.DataFrame,
        output_file: str = None
    ):
        """
        Sauvegarder un résumé complet du monitoring
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"monitoring_summary_{timestamp}.json"
        
        logger.info("💾 Sauvegarde du résumé de monitoring...")
        
        # Collecter toutes les informations
        drift_summary = self.generate_data_drift_report(current_data, save_html=False)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': drift_summary,
            'current_data_shape': current_data.shape,
            'reference_data_shape': self.reference_data.shape
        }
        
        # Ajouter performance si target disponible
        if self.target_col in current_data.columns:
            perf_summary = self.generate_model_performance_report(current_data, save_html=False)
            summary['model_performance'] = perf_summary
        
        # Vérifier besoin retraining
        needs_retraining, reason, details = self.check_need_for_retraining(current_data)
        summary['retraining'] = {
            'needs_retraining': needs_retraining,
            'reason': reason,
            'details': details
        }
        
        # Sauvegarder
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"✅ Résumé sauvegardé : {output_file}")
        
        return summary


def main():
    """
    Exemple d'utilisation
    """
    import joblib

    from src.models.preprocessing import prepare_data_for_training
    
    print("\n" + "=" * 60)
    print("TEST DU MONITORING EVIDENTLY")
    print("=" * 60)
    
    # 1. Charger les données
    X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_training(
        data_path=str(config.RAW_DATASET),
        target_col='churn',
        test_size=0.2,
        random_state=42
    )
    
    # 2. Charger le modèle
    model = joblib.load(config.MODEL_LATEST)
    
    # 3. Préparer les dataframes (avec features originales + target)
    df = pd.read_csv(config.RAW_DATASET)
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['churn']
    )
    
    # 4. Initialiser le moniteur
    monitor = MLMonitor(
        reference_data=train_df,
        model=model,
        preprocessor=preprocessor,
        target_col='churn',
        prediction_col='prediction'
    )
    
    # 5. Générer les rapports
    print("\n📊 Génération des rapports...")
    drift_report = monitor.generate_data_drift_report(test_df)
    monitor.generate_model_performance_report(test_df)
    test_results = monitor.run_test_suite(test_df)
    
    # 6. Vérifier besoin de réentraînement
    needs_retraining, reason, details = monitor.check_need_for_retraining(test_df)
    
    print(f"\n{'='*60}")
    print("RÉSULTATS DU MONITORING")
    print(f"{'='*60}")
    print(f"Dérive détectée : {drift_report['dataset_drift_detected']}")
    print(f"Tests réussis : {test_results['passed_tests']}/{test_results['total_tests']}")
    print(f"Réentraînement nécessaire : {needs_retraining}")
    if needs_retraining:
        print(f"Raison : {reason}")
    
    # 7. Sauvegarder résumé
    monitor.save_monitoring_summary(test_df)
    
    print("\n✅ Test du monitoring terminé")
    print(f"📁 Rapports disponibles dans : {monitor.output_dir}")


if __name__ == "__main__":
    main()