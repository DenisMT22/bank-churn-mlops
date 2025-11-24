"""
Pipeline de Preprocessing pour la Prédiction du Churn Bancaire
================================================================

Ce module contient toutes les transformations nécessaires pour préparer
les données avant l'entraînement du modèle.

"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Transformer custom pour créer de nouvelles features
    """
    
    def __init__(self):
        """Initialisation du transformer"""
        pass
    
    def fit(self, X, y=None):
        """Apprentissage (rien à apprendre ici)"""
        return self
    
    def transform(self, X):
        """
        Création de nouvelles features
        
        Parameters:
        -----------
        X : pd.DataFrame
            Données d'entrée
            
        Returns:
        --------
        pd.DataFrame
            Données avec nouvelles features
        """
        X = X.copy()
        
        # 1. Balance per Product (éviter division par zéro)
        X['BalancePerProduct'] = X['balance'] / (X['products_number'] + 0.01)
        
        # 2. Age to Tenure Ratio
        X['AgeToTenureRatio'] = X['age'] / (X['tenure'] + 1)
        
        # 3. Salary per Age
        X['SalaryPerAge'] = X['estimated_salary'] / X['age']
        
        # 4. Is Senior (Age > 60)
        X['IsSenior'] = (X['age'] > 60).astype(int)
        
        # 5. Has Zero Balance
        X['HasZeroBalance'] = (X['balance'] == 0).astype(int)
        
        # 6. Is Young (Age < 30)
        X['IsYoung'] = (X['age'] < 30).astype(int)
        
        # 7. Has Multiple Products
        X['HasMultipleProducts'] = (X['products_number'] > 1).astype(int)
        
        # 8. Credit Score Groups
        X['CreditScore_Low'] = (X['credit_score'] < 600).astype(int)
        X['CreditScore_Medium'] = ((X['credit_score'] >= 600) & (X['credit_score'] < 700)).astype(int)
        X['CreditScore_High'] = (X['credit_score'] >= 700).astype(int)
        
        # 9. Tenure Groups
        X['Tenure_New'] = (X['tenure'] <= 3).astype(int)
        X['Tenure_Medium'] = ((X['tenure'] > 3) & (X['tenure'] <= 6)).astype(int)
        X['Tenure_Long'] = (X['tenure'] > 6).astype(int)
        
        # 10. Interaction: Active Member + Products
        X['Active_Products_Interaction'] = X['active_member'] * X['products_number']
        
        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Gestion des outliers par winsorization (clipping)
    """
    
    def __init__(self, columns=None, lower_quantile=0.01, upper_quantile=0.99):
        """
        Parameters:
        -----------
        columns : list, optional
            Colonnes à traiter (si None, toutes les colonnes numériques)
        lower_quantile : float
            Quantile inférieur pour le clipping
        upper_quantile : float
            Quantile supérieur pour le clipping
        """
        self.columns = columns
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds_ = {}
    
    def fit(self, X, y=None):
        """Calcul des bornes de clipping"""
        X = X.copy()
        
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in self.columns:
            if col in X.columns:
                self.bounds_[col] = {
                    'lower': X[col].quantile(self.lower_quantile),
                    'upper': X[col].quantile(self.upper_quantile)
                }
        
        return self
    
    def transform(self, X):
        """Application du clipping"""
        X = X.copy()
        
        for col, bounds in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(lower=bounds['lower'], upper=bounds['upper'])
        
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodage des variables catégorielles
    """
    
    def __init__(self, encoding_type='onehot'):
        """
        Parameters:
        -----------
        encoding_type : str
            Type d'encodage ('onehot' ou 'label')
        """
        self.encoding_type = encoding_type
        self.encoders_ = {}
        self.encoded_columns_ = []
    
    def fit(self, X, y=None):
        """Apprentissage des encodeurs"""
        X = X.copy()
        
        categorical_cols = ['country', 'gender']
        
        if self.encoding_type == 'onehot':
            for col in categorical_cols:
                if col in X.columns:
                    # Stocker les catégories uniques
                    self.encoders_[col] = X[col].unique().tolist()
        else:  # label encoding
            for col in categorical_cols:
                if col in X.columns:
                    le = LabelEncoder()
                    le.fit(X[col])
                    self.encoders_[col] = le
        
        return self
    
    def transform(self, X):
        """Application de l'encodage"""
        X = X.copy()
        
        if self.encoding_type == 'onehot':
            # One-Hot Encoding
            for col, categories in self.encoders_.items():
                if col in X.columns:
                    # Créer colonnes one-hot
                    for cat in categories:
                        col_name = f"{col}_{cat}"
                        X[col_name] = (X[col] == cat).astype(int)
                        self.encoded_columns_.append(col_name)
                    # Supprimer colonne originale
                    X = X.drop(columns=[col])
        else:
            # Label Encoding
            for col, encoder in self.encoders_.items():
                if col in X.columns:
                    X[col] = encoder.transform(X[col])
        
        return X


class DataPreprocessor:
    """
    Pipeline complet de preprocessing
    """
    
    def __init__(self, handle_outliers=True, feature_engineering=True):
        """
        Parameters:
        -----------
        handle_outliers : bool
            Activer la gestion des outliers
        feature_engineering : bool
            Activer le feature engineering
        """
        self.handle_outliers = handle_outliers
        self.feature_engineering = feature_engineering
        self.scaler = StandardScaler()
        self.feature_engineer = None
        self.outlier_handler = None
        self.categorical_encoder = None
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        """
        Apprentissage du pipeline
        
        Parameters:
        -----------
        X : pd.DataFrame
            Données d'entraînement
        y : pd.Series, optional
            Variable cible
            
        Returns:
        --------
        self
        """
        X = X.copy()
        
        # 1. Feature Engineering
        if self.feature_engineering:
            self.feature_engineer = FeatureEngineering()
            X = self.feature_engineer.fit_transform(X)
        
        # 2. Gestion des outliers
        if self.handle_outliers:
            outlier_cols = ['credit_score', 'age', 'balance', 'estimated_salary']
            self.outlier_handler = OutlierHandler(columns=outlier_cols)
            X = self.outlier_handler.fit_transform(X)
        
        # 3. Encodage catégoriel
        self.categorical_encoder = CategoricalEncoder(encoding_type='onehot')
        X = self.categorical_encoder.fit_transform(X)
        
        # 4. Sélection des features pour scaling
        # Exclure customer_id et autres colonnes non pertinentes
        cols_to_drop = ['RowNumber', 'customer_id', 'Surname']
        X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
        
        # Stocker les noms de features finaux
        self.feature_names_ = X.columns.tolist()
        
        # 5. Standardisation
        self.scaler.fit(X)
        
        return self
    
    def transform(self, X):
        """
        Application du pipeline
        
        Parameters:
        -----------
        X : pd.DataFrame
            Données à transformer
            
        Returns:
        --------
        np.ndarray
            Données transformées
        """
        X = X.copy()
        
        # 1. Feature Engineering
        if self.feature_engineering and self.feature_engineer:
            X = self.feature_engineer.transform(X)
        
        # 2. Gestion des outliers
        if self.handle_outliers and self.outlier_handler:
            X = self.outlier_handler.transform(X)
        
        # 3. Encodage catégoriel
        if self.categorical_encoder:
            X = self.categorical_encoder.transform(X)
        
        # 4. Suppression colonnes non pertinentes
        cols_to_drop = ['RowNumber', 'customer_id', 'Surname']
        X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
        
        # 5. Vérifier que toutes les features attendues sont présentes
        missing_cols = set(self.feature_names_) - set(X.columns)
        if missing_cols:
            # Ajouter colonnes manquantes avec des zéros
            for col in missing_cols:
                X[col] = 0
        
        # Réorganiser les colonnes dans le bon ordre
        X = X[self.feature_names_]
        
        # 6. Standardisation
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def fit_transform(self, X, y=None):
        """Fit puis transform"""
        self.fit(X, y)
        return self.transform(X)
    
    def save(self, filepath):
        """
        Sauvegarder le preprocessor
        
        Parameters:
        -----------
        filepath : str
            Chemin de sauvegarde
        """
        joblib.dump(self, filepath)
        print(f"✅ Preprocessor sauvegardé : {filepath}")
    
    @staticmethod
    def load(filepath):
        """
        Charger un preprocessor sauvegardé
        
        Parameters:
        -----------
        filepath : str
            Chemin du fichier
            
        Returns:
        --------
        DataPreprocessor
        """
        preprocessor = joblib.load(filepath)
        print(f"✅ Preprocessor chargé : {filepath}")
        return preprocessor
    
    def get_feature_names(self):
        """Retourner les noms de features après transformation"""
        return self.feature_names_


def prepare_data_for_training(data_path, target_col='churn', test_size=0.2, random_state=42):
    """
    Fonction complète pour préparer les données pour l'entraînement
    
    Parameters:
    -----------
    data_path : str
        Chemin du fichier CSV
    target_col : str
        Nom de la colonne cible
    test_size : float
        Proportion du test set
    random_state : int
        Seed pour reproductibilité
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, preprocessor)
    """
    from sklearn.model_selection import train_test_split
    
    # 1. Chargement des données
    print("=" * 60)
    print("PRÉPARATION DES DONNÉES")
    print("=" * 60)
    df = pd.read_csv(data_path)
    print(f"✅ Données chargées : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    
    # 2. Séparation X et y
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"\nDistribution cible (avant split):")
    print(f"  Non-Churners (0): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Churners (1): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    # 3. Train-Test Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n✅ Split effectué :")
    print(f"  Train set : {X_train.shape[0]:,} samples")
    print(f"  Test set : {X_test.shape[0]:,} samples")
    
    # 4. Preprocessing
    print("\n🔧 Application du preprocessing...")
    preprocessor = DataPreprocessor(
        handle_outliers=True,
        feature_engineering=True
    )
    
    # Fit sur train uniquement
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"✅ Preprocessing terminé")
    print(f"  Nombre de features finales : {X_train_processed.shape[1]}")
    print(f"  Features créées : {len(preprocessor.get_feature_names())}")
    
    # 5. Sauvegarde du preprocessor
    preprocessor.save('../models/preprocessor.pkl')
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


if __name__ == "__main__":
    """
    Test du pipeline de preprocessing
    """
    print("\n" + "=" * 60)
    print("TEST DU PIPELINE DE PREPROCESSING")
    print("=" * 60)
    
    # Charger et préparer les données
    X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_training(
        data_path='/Users/denismutombotshituka/bank-churn-mlops/data/raw/Bank_Churn_prediction.csv',
        target_col='churn',
        test_size=0.2,
        random_state=42
    )
    
    print("\n✅ Test du preprocessing réussi !")
    print(f"  X_train shape : {X_train.shape}")
    print(f"  X_test shape : {X_test.shape}")
    print(f"  y_train shape : {y_train.shape}")
    print(f"  y_test shape : {y_test.shape}")
    
    print("\n Noms des features :")
    for i, feat in enumerate(preprocessor.get_feature_names()[:20], 1):
        print(f"  {i}. {feat}")
    if len(preprocessor.get_feature_names()) > 20:
        print(f"  ... et {len(preprocessor.get_feature_names()) - 20} autres")
    
    print("\n" + "=" * 60)