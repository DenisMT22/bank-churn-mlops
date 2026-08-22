"""
Tests Unitaires du Modèle Entraîné
===================================

Ces tests portent sur les artefacts produits par `make train` : le modèle,
le preprocessor et le fichier de métriques. Ils vérifient que le modèle
prédit bien, qu'il est déterministe, et surtout que les métriques publiées
correspondent à ce que le modèle produit réellement sur le jeu de test.

Si les artefacts sont absents, les tests échouent avec un message explicite
plutôt que d'être ignorés en silence : un artefact manquant en intégration
continue est un problème, pas un cas à passer sous silence.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.utils import config

# Tolérance de comparaison des métriques : les artefacts sont produits à
# graine fixée, l'écart attendu est nul, on garde une marge pour les
# différences d'arrondi entre versions de bibliothèques.
TOLERANCE = 1e-6


def _exiger(chemin):
    if not chemin.exists():
        pytest.fail(
            f"Artefact manquant : {chemin}\n"
            "Lancer `make train` pour le régénérer depuis data/raw."
        )
    return chemin


@pytest.fixture(scope="module")
def modele():
    return joblib.load(_exiger(config.MODEL_LATEST))


@pytest.fixture(scope="module")
def preprocessor():
    return joblib.load(_exiger(config.PREPROCESSOR))


@pytest.fixture(scope="module")
def metadonnees():
    with open(_exiger(config.MODEL_METADATA)) as fichier:
        return json.load(fichier)


@pytest.fixture(scope="module")
def jeu_de_test():
    """Rejoue exactement le découpage de l'entraînement, sans rien réécrire."""
    donnees = pd.read_csv(_exiger(config.RAW_DATASET))
    X = donnees.drop(columns=[config.TARGET_COLUMN])
    y = donnees[config.TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    return X_test, y_test


class TestArtefacts:
    """Présence et nature des fichiers produits par l'entraînement."""

    def test_le_modele_est_un_classifieur_entraine(self, modele):
        assert hasattr(modele, "predict")
        assert hasattr(modele, "predict_proba")
        assert hasattr(modele, "classes_"), "le modèle n'est pas entraîné"

    def test_les_classes_sont_binaires(self, modele):
        assert sorted(modele.classes_.tolist()) == [0, 1]

    def test_le_preprocessor_expose_ses_features(self, preprocessor):
        noms = preprocessor.get_feature_names()
        assert len(noms) > 0
        assert len(noms) == len(set(noms)), "des noms de features sont dupliqués"

    def test_le_modele_et_le_preprocessor_s_accordent(self, modele, preprocessor):
        assert modele.n_features_in_ == len(preprocessor.get_feature_names())


class TestPredictions:
    """Forme et plage des sorties."""

    @pytest.fixture(scope="class")
    @staticmethod
    def predictions(modele, preprocessor, jeu_de_test):
        X_test, _ = jeu_de_test
        X = preprocessor.transform(X_test)
        return modele.predict(X), modele.predict_proba(X)

    def test_une_prediction_par_client(self, predictions, jeu_de_test):
        classes, _ = predictions
        assert classes.shape == (len(jeu_de_test[0]),)

    def test_les_predictions_sont_binaires(self, predictions):
        classes, _ = predictions
        assert set(np.unique(classes)).issubset({0, 1})

    def test_les_probabilites_sont_dans_l_intervalle_unitaire(self, predictions):
        _, probabilites = predictions
        assert probabilites.min() >= 0.0
        assert probabilites.max() <= 1.0

    def test_les_probabilites_somment_a_un(self, predictions):
        _, probabilites = predictions
        np.testing.assert_allclose(probabilites.sum(axis=1), 1.0, rtol=1e-9)

    def test_la_classe_predite_suit_la_probabilite(self, predictions):
        classes, probabilites = predictions
        np.testing.assert_array_equal(classes, (probabilites[:, 1] >= 0.5).astype(int))

    def test_le_modele_ne_predit_pas_une_seule_classe(self, predictions):
        """Un modèle qui prédit toujours 0 aurait de bonnes métriques globales."""
        classes, _ = predictions
        assert len(np.unique(classes)) == 2

    def test_une_prediction_unitaire_fonctionne(self, modele, preprocessor, jeu_de_test):
        X_test, _ = jeu_de_test
        X = preprocessor.transform(X_test.head(1))
        assert modele.predict(X).shape == (1,)


class TestDeterminisme:
    """Deux appels identiques doivent donner le même résultat."""

    def test_predictions_identiques_entre_deux_appels(self, modele, preprocessor, jeu_de_test):
        X_test, _ = jeu_de_test
        X = preprocessor.transform(X_test)
        np.testing.assert_array_equal(modele.predict(X), modele.predict(X))

    def test_probabilites_identiques_entre_deux_appels(self, modele, preprocessor, jeu_de_test):
        X_test, _ = jeu_de_test
        X = preprocessor.transform(X_test)
        np.testing.assert_array_equal(modele.predict_proba(X), modele.predict_proba(X))

    def test_le_modele_recharge_predit_a_l_identique(self, modele, preprocessor, jeu_de_test):
        X_test, _ = jeu_de_test
        X = preprocessor.transform(X_test)
        recharge = joblib.load(config.MODEL_LATEST)
        np.testing.assert_array_equal(modele.predict(X), recharge.predict(X))


class TestMetriquesPubliees:
    """
    Les chiffres du fichier de métriques doivent être ceux que le modèle
    produit réellement. C'est ce test qui empêche un README de dériver de
    la réalité du modèle.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def mesures(modele, preprocessor, jeu_de_test):
        X_test, y_test = jeu_de_test
        predictions = modele.predict(preprocessor.transform(X_test))
        return {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
        }

    @pytest.mark.parametrize("metrique", ["accuracy", "precision", "recall"])
    def test_la_metrique_publiee_est_celle_mesuree(self, metrique, mesures, metadonnees):
        publiee = metadonnees["metrics"][metrique]
        mesuree = mesures[metrique]
        assert abs(publiee - mesuree) < TOLERANCE, (
            f"{metrique} publié {publiee:.6f} contre {mesuree:.6f} mesuré. "
            "Le fichier de métriques ne correspond plus au modèle."
        )

    def test_la_taille_du_jeu_de_test_est_celle_annoncee(self, metadonnees, jeu_de_test):
        assert metadonnees["test_samples"] == len(jeu_de_test[0])

    def test_le_modele_nomme_correspond_a_l_artefact(self, metadonnees, modele):
        assert metadonnees["model_name"].replace(" ", "") == type(modele).__name__

    def test_le_desequilibre_est_bien_pris_en_compte(self, metadonnees):
        """Le projet annonce une pondération des classes : on le vérifie."""
        assert metadonnees["hyperparameters"].get("class_weight") == "balanced"
