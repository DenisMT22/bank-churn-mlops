"""
Script d'Entraînement du Modèle de Prédiction de Churn
=======================================================

Ce script entraîne plusieurs modèles, les compare et sauvegarde le meilleur.

"""

import pandas as pd
import numpy as np
import matplotlib

# Backend sans fenetre : le script doit tourner dans un terminal sans
# serveur graphique, notamment en integration continue.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib
import json
import warnings

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
import xgboost as xgb

import sys
from pathlib import Path

# Import du module de configuration des chemins. Le double essai permet
# d'executer ce fichier aussi bien comme script que comme module importe,
# en local comme dans l'image Docker.
try:
    from src.utils import config
except ImportError:  # pragma: no cover - depend du mode d'execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.utils import config

from src.models.preprocessing import prepare_data_for_training, DataPreprocessor

warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Classe pour entraîner et comparer différents modèles
    """
    
    def __init__(self, random_state=42):
        """
        Initialisation
        
        Parameters:
        -----------
        random_state : int
            Seed pour reproductibilité
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.n_train = None
        self.n_test = None
        
    def define_models(self, seulement=None):
        """
        Définir les modèles à tester
        """
        print("\n" + "=" * 60)
        print("DÉFINITION DES MODÈLES")
        print("=" * 60)
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=3,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=self.random_state,
                eval_metric='logloss',
                scale_pos_weight=3  # Pour gérer le déséquilibre
            )
        }
        
        # Restriction eventuelle a un sous-ensemble de modeles : utile pour
        # regenerer rapidement les artefacts sans reentrainer toute la
        # comparaison, qui prend plus d'une minute.
        if seulement:
            inconnus = set(seulement) - set(self.models)
            if inconnus:
                raise ValueError(f"Modeles inconnus : {sorted(inconnus)}")
            self.models = {k: v for k, v in self.models.items() if k in seulement}

        print(f"✅ {len(self.models)} modèles définis :")
        for name in self.models.keys():
            print(f"  • {name}")
        
        return self
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, use_smote=True):
        """
        Entraîner et évaluer tous les modèles
        
        Parameters:
        -----------
        X_train, X_test : array-like
            Features d'entraînement et de test
        y_train, y_test : array-like
            Cibles d'entraînement et de test
        use_smote : bool
            Utiliser SMOTE pour rééquilibrer les classes
        """
        self.n_train, self.n_test = len(X_train), len(X_test)
        print("\n" + "=" * 60)
        print("ENTRAÎNEMENT DES MODÈLES")
        print("=" * 60)
        
        # Application SMOTE si demandé
        if use_smote:
            print("\n🔄 Application de SMOTE pour rééquilibrer les classes...")
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"  Avant SMOTE : {X_train.shape[0]:,} samples")
            print(f"  Après SMOTE : {X_train_resampled.shape[0]:,} samples")
            print(f"  Distribution après SMOTE :")
            print(f"    Classe 0 : {(y_train_resampled==0).sum():,}")
            print(f"    Classe 1 : {(y_train_resampled==1).sum():,}")
        else:
            X_train_resampled = X_train
            y_train_resampled = y_train
        
        # Entraîner chaque modèle
        for name, model in self.models.items():
            print(f"\n{'='*40}")
            print(f"🎯 Entraînement : {name}")
            print(f"{'='*40}")
            
            # Entraînement
            model.fit(X_train_resampled, y_train_resampled)
            print("✅ Entraînement terminé")
            
            # Prédictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Métriques
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Cross-validation sur train set
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall')
            metrics['cv_recall_mean'] = cv_scores.mean()
            metrics['cv_recall_std'] = cv_scores.std()
            
            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            
            # Stocker résultats
            self.results[name] = {
                'model': model,
                'metrics': metrics,
                'confusion_matrix': cm,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Afficher métriques
            print(f"\n📊 Métriques sur Test Set :")
            print(f"  Accuracy  : {metrics['accuracy']:.4f}")
            print(f"  Precision : {metrics['precision']:.4f}")
            print(f"  Recall    : {metrics['recall']:.4f} ⭐")
            print(f"  F1-Score  : {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
            print(f"\n📊 Cross-Validation (Recall) :")
            print(f"  Moyenne : {metrics['cv_recall_mean']:.4f}")
            print(f"  Std Dev : {metrics['cv_recall_std']:.4f}")
            
            print(f"\n🔢 Matrice de Confusion :")
            print(f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
            print(f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")
        
        return self
    
    def select_best_model(self, metric='recall'):
        """
        Sélectionner le meilleur modèle basé sur une métrique
        
        Parameters:
        -----------
        metric : str
            Métrique à optimiser ('recall', 'f1_score', 'roc_auc')
        """
        print("\n" + "=" * 60)
        print(f"SÉLECTION DU MEILLEUR MODÈLE (basé sur {metric})")
        print("=" * 60)
        
        best_score = 0
        best_name = None
        
        print(f"\n📊 Comparaison des modèles :")
        for name, result in self.results.items():
            score = result['metrics'][metric]
            print(f"  {name:25s} : {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.results[best_name]['model']
        
        print(f"\n🏆 Meilleur modèle : {best_name}")
        print(f"   {metric} = {best_score:.4f}")
        
        return self
    
    def plot_results(self, y_test, save_path=None):
        """
        Visualiser les résultats
        """
        print("\n" + "=" * 60)
        print("GÉNÉRATION DES VISUALISATIONS")
        print("=" * 60)

        # Par defaut, les figures vont dans docs/ a la racine du projet.
        if save_path is None:
            save_path = f"{config.DOCS_DIR}/"
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # 1. Comparaison des métriques
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        x = np.arange(len(self.results))
        width = 0.15
        
        for i, metric in enumerate(metrics_to_plot):
            values = [self.results[name]['metrics'][metric] for name in self.results.keys()]
            ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Modèles')
        ax.set_ylabel('Score')
        ax.set_title('Comparaison des Métriques par Modèle')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(self.results.keys(), rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}metrics_comparison.png", dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé : {save_path}metrics_comparison.png")
        plt.close()
        
        # 2. Matrices de confusion
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Non-Churn', 'Churn'],
                       yticklabels=['Non-Churn', 'Churn'])
            axes[idx].set_xlabel('Prédiction')
            axes[idx].set_ylabel('Vérité')
            axes[idx].set_title(f'{name}\n(Recall: {result["metrics"]["recall"]:.3f})')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}confusion_matrices.png", dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé : {save_path}confusion_matrices.png")
        plt.close()
        
        # 3. Courbes ROC
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, result in self.results.items():
            y_pred_proba = result['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = result['metrics']['roc_auc']
            
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Hasard')
        ax.set_xlabel('Taux de Faux Positifs')
        ax.set_ylabel('Taux de Vrais Positifs')
        ax.set_title('Courbes ROC - Comparaison des Modèles')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}roc_curves.png", dpi=300, bbox_inches='tight')
        print(f"✅ Graphique sauvegardé : {save_path}roc_curves.png")
        plt.close()
        
        # 4. Feature Importance (meilleur modèle)
        if hasattr(self.best_model, 'feature_importances_'):
            # Charger les noms de features
            preprocessor = DataPreprocessor.load(str(config.PREPROCESSOR))
            feature_names = preprocessor.get_feature_names()
            
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20
            
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(range(20), importances[indices], color='steelblue')
            ax.set_yticks(range(20))
            ax.set_yticklabels([feature_names[i] for i in indices])
            ax.set_xlabel('Importance')
            ax.set_title(f'Top 20 Features - {self.best_model_name}')
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(f"{save_path}feature_importance.png", dpi=300, bbox_inches='tight')
            print(f"✅ Graphique sauvegardé : {save_path}feature_importance.png")
            plt.close()
        
        print(f"\n✅ Toutes les visualisations ont été sauvegardées dans {save_path}")
        
        return self
    
    def save_best_model(self, model_path=None, metadata_path=None):
        """
        Sauvegarder le meilleur modèle et ses métadonnées
        """
        print("\n" + "=" * 60)
        print("SAUVEGARDE DU MEILLEUR MODÈLE")
        print("=" * 60)

        if model_path is None:
            model_path = f"{config.TRAINED_MODELS_DIR}/"
        if metadata_path is None:
            metadata_path = f"{config.MODELS_DIR}/"
        Path(model_path).mkdir(parents=True, exist_ok=True)
        Path(metadata_path).mkdir(parents=True, exist_ok=True)
        
        # Créer nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"best_model_{timestamp}.pkl"
        
        # Sauvegarder le modèle
        joblib.dump(self.best_model, f"{model_path}{model_filename}")
        print(f"✅ Modèle sauvegardé : {model_path}{model_filename}")
        
        # Sauvegarder aussi comme "latest"
        joblib.dump(self.best_model, f"{model_path}model_latest.pkl")
        print(f"✅ Modèle sauvegardé : {model_path}model_latest.pkl")
        
        # Métadonnées stables : ce fichier est suivi par git et sert de
        # reference aux chiffres cites dans la documentation. Il ne doit
        # donc contenir que ce qui ne bouge pas d'une execution a l'autre.
        metadata = {
            'model_name': self.best_model_name,
            'metrics': {k: float(v) for k, v in self.results[self.best_model_name]['metrics'].items()},
            'training_samples': int(self.n_train) if self.n_train else None,
            'test_samples': int(self.n_test) if self.n_test else None,
            'hyperparameters': self.best_model.get_params()
        }
        
        with open(f"{metadata_path}model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ Métadonnées sauvegardées : {metadata_path}model_metadata.json")

        # Trace de l'execution : horodatage et nom du fichier produit.
        # Ces champs changent a chaque entrainement, ils sont donc isoles
        # ici et exclus du suivi git.
        derniere_execution = {
            'timestamp': timestamp,
            'model_filename': model_filename,
            'model_name': self.best_model_name,
        }
        with open(f"{metadata_path}last_run.json", 'w') as f:
            json.dump(derniere_execution, f, indent=4)
        print(f"✅ Trace d'exécution sauvegardée : {metadata_path}last_run.json")

        # Comparaison de tous les modeles evalues. Elle n'est ecrite que
        # lorsque la comparaison complete a tourne : un entrainement rapide
        # ne porte que sur un modele et ecraserait le tableau.
        if len(self.results) > 1:
            comparaison = {
                nom: {k: float(v) for k, v in donnees['metrics'].items()}
                for nom, donnees in self.results.items()
            }
            with open(f"{metadata_path}model_comparison.json", 'w') as f:
                json.dump(
                    {'best_model': self.best_model_name, 'models': comparaison},
                    f,
                    indent=4,
                )
            print(f"✅ Comparaison sauvegardée : {metadata_path}model_comparison.json")
        
        print(f"\n📋 Résumé du modèle sauvegardé :")
        print(f"  Nom : {self.best_model_name}")
        print(f"  Recall : {metadata['metrics']['recall']:.4f}")
        print(f"  Precision : {metadata['metrics']['precision']:.4f}")
        print(f"  F1-Score : {metadata['metrics']['f1_score']:.4f}")
        print(f"  ROC-AUC : {metadata['metrics']['roc_auc']:.4f}")
        
        return self


def main(rapide=False, figures=True):
    """
    Fonction principale d'entraînement

    Parameters:
    -----------
    rapide : bool
        N'entrainer que la regression logistique, qui est le modele
        retenu. Utile pour regenerer les artefacts en quelques secondes.
    figures : bool
        Regenerer les figures de docs/.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "🚀 ENTRAÎNEMENT DU MODÈLE DE CHURN")
    print("=" * 80)
    
    # 1. Préparer les données
    X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_training(
        data_path=str(config.RAW_DATASET),
        target_col=config.TARGET_COLUMN,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    
    # 2. Entraîner les modèles
    trainer = ModelTrainer(random_state=config.RANDOM_STATE)
    trainer.define_models(seulement=['Logistic Regression'] if rapide else None)
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test, use_smote=True)
    
    # 3. Sélectionner le meilleur
    trainer.select_best_model(metric='recall')
    
    # 4. Visualiser
    if figures:
        trainer.plot_results(y_test)
    
    # 5. Sauvegarder
    trainer.save_best_model()
    
    print("\n" + "=" * 80)
    print(" " * 25 + "✅ ENTRAÎNEMENT TERMINÉ")
    print("=" * 80)
    
    return trainer


if __name__ == "__main__":
    import argparse

    parseur = argparse.ArgumentParser(
        description="Entraine le modele de prediction de churn a partir du "
                    "jeu de donnees source et regenere tous les artefacts."
    )
    parseur.add_argument(
        "--rapide",
        action="store_true",
        help="N'entrainer que la regression logistique, le modele retenu.",
    )
    parseur.add_argument(
        "--sans-figures",
        action="store_true",
        help="Ne pas regenerer les figures de docs/.",
    )
    arguments = parseur.parse_args()

    trainer = main(rapide=arguments.rapide, figures=not arguments.sans_figures)
