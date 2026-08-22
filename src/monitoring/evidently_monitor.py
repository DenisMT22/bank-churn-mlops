"""
Monitoring du Modèle avec Evidently
====================================

Détection de dérive des données et suivi des performances du modèle.

Ce module est écrit pour l'API Evidently 0.7, qui repose sur des objets
`Dataset` et `DataDefinition` là où les versions 0.4 utilisaient un
`ColumnMapping` et des `TestSuite` distincts. Les tests sont désormais
attachés directement aux métriques.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import (
    Accuracy,
    DriftedColumnsCount,
    F1Score,
    Precision,
    Recall,
)
from evidently.presets import DataDriftPreset
from evidently.tests import gte, lte

# Import du module de configuration des chemins. Le double essai permet
# d'executer ce fichier aussi bien comme script que comme module importe.
try:
    from src.utils import config
except ImportError:  # pragma: no cover - depend du mode d'execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

COLONNES_NUMERIQUES = [
    'credit_score', 'age', 'tenure', 'balance',
    'products_number', 'estimated_salary',
]
COLONNES_CATEGORIELLES = ['country', 'gender', 'credit_card', 'active_member']

# Seuils inchanges depuis la version 0.4 du module. Le modele etant
# optimise sur le recall, il ne satisfait pas min_precision ni min_f1 :
# ces deux tests echouent, et c'est le signal attendu. Seul le recall
# declenche un reentrainement, via check_need_for_retraining.
SEUILS_PAR_DEFAUT = {
    'max_share_drifted_columns': 0.3,
    'min_accuracy': 0.75,
    'min_precision': 0.60,
    'min_recall': 0.70,
    'min_f1': 0.65,
}


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
        self.model = model
        self.preprocessor = preprocessor
        self.target_col = target_col
        self.prediction_col = prediction_col
        # Chemin resolu depuis la racine du projet : un chemin relatif
        # dependrait du repertoire courant et creerait un dossier
        # parasite selon l'endroit d'ou le script est lance.
        self.output_dir = Path(output_dir) if output_dir else config.MONITORING_REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.reference_data = self._avec_predictions(reference_data)

        logger.info("✅ MLMonitor initialisé")

    # ------------------------------------------------------------------
    # Utilitaires internes
    # ------------------------------------------------------------------

    def _avec_predictions(self, donnees: pd.DataFrame) -> pd.DataFrame:
        """Ajouter la colonne de prédiction si elle est absente."""
        donnees = donnees.copy()
        if self.prediction_col in donnees.columns:
            return donnees
        entrees = donnees.drop(columns=[self.target_col], errors='ignore')
        donnees[self.prediction_col] = self.model.predict(
            self.preprocessor.transform(entrees)
        )
        return donnees

    def _definition(self, donnees: pd.DataFrame) -> DataDefinition:
        """Décrire les colonnes pour Evidently."""
        cible_presente = self.target_col in donnees.columns
        classification = None
        if cible_presente:
            classification = [
                BinaryClassification(
                    target=self.target_col,
                    prediction_labels=self.prediction_col,
                )
            ]
        return DataDefinition(
            classification=classification,
            numerical_columns=[c for c in COLONNES_NUMERIQUES if c in donnees.columns],
            categorical_columns=[c for c in COLONNES_CATEGORIELLES if c in donnees.columns],
        )

    def _jeux(self, current_data: pd.DataFrame) -> Tuple[Dataset, Dataset]:
        """Emballer référence et données courantes en jeux Evidently."""
        courant = self._avec_predictions(current_data)
        definition = self._definition(courant)
        return (
            Dataset.from_pandas(self.reference_data, data_definition=definition),
            Dataset.from_pandas(courant, data_definition=definition),
        )

    @staticmethod
    def _valeurs(instantane) -> Dict:
        """Indexer les valeurs d'un instantané par nom de métrique."""
        resultat = {}
        for metrique in instantane.dict().get('metrics', []):
            nom = metrique['metric_name'].split('(')[0]
            resultat[nom] = metrique['value']
        return resultat

    def _enregistrer(self, instantane, prefixe: str) -> Path:
        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        chemin = self.output_dir / f"{prefixe}_{horodatage}.html"
        instantane.save_html(str(chemin))
        logger.info(f"✅ Rapport HTML sauvegardé : {chemin}")
        return chemin

    # ------------------------------------------------------------------
    # Rapports
    # ------------------------------------------------------------------

    def generate_data_drift_report(
        self,
        current_data: pd.DataFrame,
        save_html: bool = True
    ) -> Dict:
        """
        Générer un rapport de dérive des données

        Parameters:
        -----------
        current_data : pd.DataFrame
            Données courantes à comparer à la référence
        save_html : bool
            Écrire le rapport HTML sur disque

        Returns:
        --------
        dict : résumé de la dérive
        """
        logger.info("📊 Génération du rapport de dérive...")
        reference, courant = self._jeux(current_data)

        instantane = Report([DataDriftPreset()]).run(
            current_data=courant, reference_data=reference
        )

        if save_html:
            self._enregistrer(instantane, "data_drift_report")

        colonnes_derivees = []
        nombre, part = 0, 0.0
        for metrique in instantane.dict().get('metrics', []):
            nom = metrique['metric_name']
            valeur = metrique['value']
            if nom.startswith('DriftedColumnsCount'):
                nombre = int(valeur.get('count', 0))
                part = float(valeur.get('share', 0.0))
            elif nom.startswith('ValueDrift'):
                # Le nom porte la colonne et le seuil employé.
                colonne = nom.split('column=')[1].split(',')[0]
                seuil = float(nom.split('threshold=')[1].rstrip(')'))
                methode = nom.split('method=')[1].split(',')[0]
                if float(valeur) > seuil:
                    colonnes_derivees.append({
                        'column': colonne,
                        'drift_score': float(valeur),
                        'stattest_name': methode,
                    })

        resume = {
            'timestamp': datetime.now().isoformat(),
            'dataset_drift_detected': part > SEUILS_PAR_DEFAUT['max_share_drifted_columns'],
            'number_of_drifted_columns': nombre,
            'share_of_drifted_columns': part,
            'drifted_columns': colonnes_derivees,
        }

        logger.info(f"📈 Dérive détectée : {resume['dataset_drift_detected']}")
        logger.info(f"📈 Colonnes dérivées : {resume['number_of_drifted_columns']}")
        return resume

    def generate_model_performance_report(
        self,
        current_data: pd.DataFrame,
        save_html: bool = True
    ) -> Dict:
        """
        Générer un rapport de performance du modèle

        Returns:
        --------
        dict : métriques courantes et de référence
        """
        logger.info("📊 Génération du rapport de performance...")

        if self.target_col not in current_data.columns:
            logger.warning("⚠️ Cible absente : performances non calculables")
            return {
                'timestamp': datetime.now().isoformat(),
                'current_metrics': {},
                'reference_metrics': {},
            }

        reference, courant = self._jeux(current_data)

        metriques = [Accuracy(), Precision(), Recall(), F1Score()]
        instantane_courant = Report(metriques).run(current_data=courant)
        instantane_reference = Report(metriques).run(current_data=reference)

        if save_html:
            self._enregistrer(
                Report(metriques).run(current_data=courant, reference_data=reference),
                "model_performance_report",
            )

        def extraire(instantane):
            valeurs = self._valeurs(instantane)
            return {
                'accuracy': valeurs.get('Accuracy'),
                'precision': valeurs.get('Precision'),
                'recall': valeurs.get('Recall'),
                'f1': valeurs.get('F1Score'),
            }

        resume = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': extraire(instantane_courant),
            'reference_metrics': extraire(instantane_reference),
        }

        logger.info(f"📈 Métriques actuelles : {resume['current_metrics']}")
        return resume

    def run_test_suite(
        self,
        current_data: pd.DataFrame,
        thresholds: Optional[Dict] = None
    ) -> Dict:
        """
        Exécuter la suite de tests de monitoring

        Parameters:
        -----------
        thresholds : dict, optional
            Seuils de déclenchement. Voir SEUILS_PAR_DEFAUT.

        Returns:
        --------
        dict : résumé des tests
        """
        logger.info("🧪 Exécution de la suite de tests...")
        seuils = {**SEUILS_PAR_DEFAUT, **(thresholds or {})}

        reference, courant = self._jeux(current_data)

        controles = [
            DriftedColumnsCount(tests=[lte(seuils['max_share_drifted_columns'])]),
        ]
        if self.target_col in current_data.columns:
            controles += [
                Accuracy(tests=[gte(seuils['min_accuracy'])]),
                Precision(tests=[gte(seuils['min_precision'])]),
                Recall(tests=[gte(seuils['min_recall'])]),
                F1Score(tests=[gte(seuils['min_f1'])]),
            ]

        instantane = Report(controles).run(
            current_data=courant, reference_data=reference
        )
        self._enregistrer(instantane, "test_suite")

        tests = instantane.dict().get('tests', [])

        def statut(test):
            valeur = test.get('status')
            return getattr(valeur, 'value', str(valeur)).upper()

        echoues = [t for t in tests if statut(t) != 'SUCCESS']

        resume = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(tests),
            'passed_tests': len(tests) - len(echoues),
            'failed_tests': len(echoues),
            'all_tests_passed': not echoues,
            'failed_test_details': [
                {
                    'test_name': t.get('name', ''),
                    'description': t.get('description', ''),
                    'status': statut(t),
                }
                for t in echoues
            ],
        }

        logger.info(f"✅ Tests réussis : {resume['passed_tests']}/{resume['total_tests']}")
        if echoues:
            logger.warning(f"⚠️ Tests échoués : {resume['failed_tests']}")
        return resume

    # ------------------------------------------------------------------
    # Décision
    # ------------------------------------------------------------------

    def check_need_for_retraining(
        self,
        current_data: pd.DataFrame,
        auto_threshold: bool = True
    ) -> Tuple[bool, str, Dict]:
        """
        Déterminer si un réentraînement est nécessaire

        Returns:
        --------
        tuple : (needs_retraining, reason, details)
        """
        logger.info("🔍 Vérification besoin de réentraînement...")

        derive = self.generate_data_drift_report(current_data, save_html=False)
        if derive['dataset_drift_detected']:
            raison = f"Dérive détectée sur {derive['number_of_drifted_columns']} colonnes"
            logger.warning(f"⚠️ {raison}")
            return True, raison, derive

        if self.target_col in current_data.columns:
            performance = self.generate_model_performance_report(current_data, save_html=False)
            recall_courant = performance['current_metrics'].get('recall') or 1.0
            recall_reference = performance['reference_metrics'].get('recall') or 1.0

            if recall_courant < SEUILS_PAR_DEFAUT['min_recall']:
                raison = (
                    f"Recall trop faible : {recall_courant:.3f} "
                    f"< {SEUILS_PAR_DEFAUT['min_recall']:.2f}"
                )
                logger.warning(f"⚠️ {raison}")
                return True, raison, performance

            # Dégradation de plus de dix points relatifs par rapport à la référence.
            if recall_courant < recall_reference * 0.90:
                raison = (
                    f"Dégradation du recall : {recall_courant:.3f} "
                    f"vs {recall_reference:.3f}"
                )
                logger.warning(f"⚠️ {raison}")
                return True, raison, performance

        logger.info("✅ Pas de besoin de réentraînement détecté")
        return False, "Performances stables", {}

    def save_monitoring_summary(
        self,
        current_data: pd.DataFrame,
        output_file: str = None
    ) -> Dict:
        """
        Sauvegarder un résumé complet du monitoring
        """
        if output_file is None:
            horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"monitoring_summary_{horodatage}.json"

        logger.info("💾 Sauvegarde du résumé de monitoring...")

        resume = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': self.generate_data_drift_report(current_data, save_html=False),
            'current_data_shape': list(current_data.shape),
            'reference_data_shape': list(self.reference_data.shape),
        }

        if self.target_col in current_data.columns:
            resume['model_performance'] = self.generate_model_performance_report(
                current_data, save_html=False
            )

        besoin, raison, details = self.check_need_for_retraining(current_data)
        resume['retraining'] = {
            'needs_retraining': besoin,
            'reason': raison,
            'details': details,
        }

        with open(output_file, 'w') as fichier:
            json.dump(resume, fichier, indent=4, default=str)

        logger.info(f"✅ Résumé sauvegardé : {output_file}")
        return resume


def main():
    """
    Exemple d'utilisation
    """
    import joblib
    from sklearn.model_selection import train_test_split

    from src.models.preprocessing import prepare_data_for_training

    print("\n" + "=" * 60)
    print("MONITORING EVIDENTLY")
    print("=" * 60)

    # 1. Ajuster le preprocessor sur le jeu d'entraînement
    prepare_data_for_training(
        data_path=str(config.RAW_DATASET),
        target_col=config.TARGET_COLUMN,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    # 2. Charger les artefacts
    model = joblib.load(config.MODEL_LATEST)
    preprocessor = joblib.load(config.PREPROCESSOR)

    # 3. Rejouer le découpage sur les colonnes d'origine
    donnees = pd.read_csv(config.RAW_DATASET)
    reference, courant = train_test_split(
        donnees,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=donnees[config.TARGET_COLUMN],
    )

    monitor = MLMonitor(
        reference_data=reference,
        model=model,
        preprocessor=preprocessor,
        target_col=config.TARGET_COLUMN,
        prediction_col='prediction',
    )

    print("\n📊 Génération des rapports...")
    derive = monitor.generate_data_drift_report(courant)
    monitor.generate_model_performance_report(courant)
    tests = monitor.run_test_suite(courant)

    besoin, raison, _ = monitor.check_need_for_retraining(courant)

    print(f"\n{'=' * 60}")
    print("RÉSULTATS DU MONITORING")
    print(f"{'=' * 60}")
    print(f"Dérive détectée : {derive['dataset_drift_detected']}")
    print(f"Colonnes dérivées : {derive['number_of_drifted_columns']}")
    print(f"Tests réussis : {tests['passed_tests']}/{tests['total_tests']}")
    print(f"Réentraînement nécessaire : {besoin}")
    if besoin:
        print(f"Raison : {raison}")

    monitor.save_monitoring_summary(courant)

    print("\n✅ Monitoring terminé")
    print(f"📁 Rapports disponibles dans : {monitor.output_dir}")


if __name__ == "__main__":
    main()
