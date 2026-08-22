"""
Tests Unitaires du Pipeline de Preprocessing
=============================================

Ces tests portent sur les transformations appliquées aux données avant
l'entraînement : création des features métier, encodage des catégories et
cohérence des colonnes produites.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.preprocessing import (
    CategoricalEncoder,
    DataPreprocessor,
    FeatureEngineering,
    OutlierHandler,
)


@pytest.fixture
def clients():
    """Un petit échantillon couvrant les cas limites des règles métier."""
    return pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "credit_score": [550, 650, 750, 599, 700, 480],
            "country": ["France", "Germany", "Spain", "France", "Germany", "Spain"],
            "gender": ["Male", "Female", "Female", "Male", "Female", "Male"],
            # 25 ans déclenche IsYoung, 61 ans déclenche IsSenior
            "age": [25, 45, 61, 30, 60, 70],
            # 0, 3 et 4 ans encadrent les bornes des groupes d'ancienneté
            "tenure": [0, 3, 4, 6, 7, 10],
            "balance": [0.0, 50_000.0, 120_000.0, 0.0, 90_000.0, 15_000.0],
            "products_number": [1, 2, 3, 1, 4, 2],
            "credit_card": [1, 0, 1, 1, 0, 1],
            "active_member": [1, 1, 0, 0, 1, 0],
            "estimated_salary": [50_000.0, 80_000.0, 120_000.0, 30_000.0, 95_000.0, 60_000.0],
        }
    )


class TestFeatureEngineering:
    """Création des variables métier dérivées."""

    def test_cree_quatorze_variables(self, clients):
        avant = set(clients.columns)
        apres = FeatureEngineering().fit_transform(clients)
        creees = set(apres.columns) - avant
        assert len(creees) == 14

    def test_conserve_les_colonnes_d_origine(self, clients):
        apres = FeatureEngineering().fit_transform(clients)
        assert set(clients.columns).issubset(set(apres.columns))

    def test_ne_modifie_pas_le_dataframe_source(self, clients):
        colonnes_avant = list(clients.columns)
        FeatureEngineering().fit_transform(clients)
        assert list(clients.columns) == colonnes_avant

    def test_solde_nul_ne_provoque_pas_de_division_par_zero(self, clients):
        apres = FeatureEngineering().fit_transform(clients)
        assert np.isfinite(apres["BalancePerProduct"]).all()
        assert np.isfinite(apres["AgeToTenureRatio"]).all()

    def test_indicateur_solde_nul(self, clients):
        apres = FeatureEngineering().fit_transform(clients)
        attendu = (clients["balance"] == 0).astype(int).tolist()
        assert apres["HasZeroBalance"].tolist() == attendu

    def test_bornes_des_tranches_d_age(self, clients):
        apres = FeatureEngineering().fit_transform(clients)
        # IsYoung strictement sous 30 ans, IsSenior strictement au-dessus de 60
        assert apres.loc[clients["age"] == 25, "IsYoung"].iat[0] == 1
        assert apres.loc[clients["age"] == 30, "IsYoung"].iat[0] == 0
        assert apres.loc[clients["age"] == 60, "IsSenior"].iat[0] == 0
        assert apres.loc[clients["age"] == 61, "IsSenior"].iat[0] == 1

    def test_les_tranches_de_score_sont_exclusives(self, clients):
        apres = FeatureEngineering().fit_transform(clients)
        somme = (
            apres["CreditScore_Low"]
            + apres["CreditScore_Medium"]
            + apres["CreditScore_High"]
        )
        assert (somme == 1).all()

    def test_les_tranches_d_anciennete_sont_exclusives(self, clients):
        apres = FeatureEngineering().fit_transform(clients)
        somme = apres["Tenure_New"] + apres["Tenure_Medium"] + apres["Tenure_Long"]
        assert (somme == 1).all()

    def test_interaction_activite_produits(self, clients):
        apres = FeatureEngineering().fit_transform(clients)
        attendu = clients["active_member"] * clients["products_number"]
        assert apres["Active_Products_Interaction"].tolist() == attendu.tolist()


class TestCategoricalEncoder:
    """Encodage des variables catégorielles."""

    def test_ne_laisse_aucune_colonne_texte(self, clients):
        encode = CategoricalEncoder(encoding_type="onehot").fit_transform(clients)
        assert not encode.select_dtypes(include=["object"]).columns.tolist()

    def test_produit_une_colonne_par_modalite_conservee(self, clients):
        encode = CategoricalEncoder(encoding_type="onehot").fit_transform(clients)
        colonnes_pays = [c for c in encode.columns if c.startswith("country_")]
        assert colonnes_pays, "aucune colonne issue de country"

    def test_les_modalites_inconnues_ne_font_pas_echouer(self, clients):
        encodeur = CategoricalEncoder(encoding_type="onehot").fit(clients)
        nouveaux = clients.copy()
        nouveaux.loc[0, "country"] = "Belgium"
        encode = encodeur.transform(nouveaux)
        assert len(encode) == len(nouveaux)


class TestOutlierHandler:
    """Écrêtage des valeurs extrêmes."""

    def test_borne_les_valeurs_extremes(self):
        donnees = pd.DataFrame({"balance": [0.0, 10.0, 20.0, 30.0, 1_000_000.0]})
        handler = OutlierHandler(columns=["balance"], lower_quantile=0.1, upper_quantile=0.9)
        borne = handler.fit_transform(donnees)
        assert borne["balance"].max() < 1_000_000.0

    def test_conserve_le_nombre_de_lignes(self):
        donnees = pd.DataFrame({"balance": [0.0, 10.0, 20.0, 30.0, 1_000_000.0]})
        handler = OutlierHandler(columns=["balance"])
        assert len(handler.fit_transform(donnees)) == len(donnees)


class TestDataPreprocessor:
    """Pipeline complet de préparation."""

    def test_sortie_numerique_et_sans_valeur_manquante(self, clients):
        sortie = DataPreprocessor().fit_transform(clients)
        assert isinstance(sortie, np.ndarray)
        assert np.isfinite(sortie).all()

    def test_l_identifiant_client_est_ecarte(self, clients):
        pre = DataPreprocessor().fit(clients)
        assert "customer_id" not in pre.get_feature_names()

    def test_le_nombre_de_colonnes_est_stable_entre_fit_et_transform(self, clients):
        pre = DataPreprocessor()
        apprentissage = pre.fit_transform(clients)
        application = pre.transform(clients)
        assert apprentissage.shape[1] == application.shape[1]
        assert apprentissage.shape[1] == len(pre.get_feature_names())

    def test_transform_est_reproductible(self, clients):
        pre = DataPreprocessor().fit(clients)
        np.testing.assert_array_equal(pre.transform(clients), pre.transform(clients))

    def test_une_seule_ligne_donne_le_meme_nombre_de_colonnes(self, clients):
        """Cas de l'API : une prédiction unitaire doit passer le pipeline."""
        pre = DataPreprocessor().fit(clients)
        complet = pre.transform(clients)
        unitaire = pre.transform(clients.head(1))
        assert unitaire.shape == (1, complet.shape[1])

    def test_l_ordre_des_colonnes_d_entree_est_sans_effet(self, clients):
        pre = DataPreprocessor().fit(clients)
        melange = clients[list(reversed(clients.columns))]
        np.testing.assert_allclose(pre.transform(clients), pre.transform(melange))

    def test_sauvegarde_et_rechargement(self, clients, tmp_path):
        pre = DataPreprocessor().fit(clients)
        chemin = tmp_path / "preprocessor.pkl"
        pre.save(str(chemin))
        recharge = DataPreprocessor.load(str(chemin))
        np.testing.assert_array_equal(pre.transform(clients), recharge.transform(clients))
