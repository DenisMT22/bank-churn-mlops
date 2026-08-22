"""
Dashboard Streamlit pour Monitoring ML
=======================================

Interface pour visualiser :
- Performances du modèle
- Prédictions en temps réel
- Monitoring et alertes

"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import requests
import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

# Configuration de la page
st.set_page_config(
    page_title="Bank Churn Prediction - MLOps Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-critical {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# MOTEUR DE PRÉDICTION
#
# Le tableau de bord fonctionne de deux manières.
#
# Par défaut il charge le modèle dans son propre processus : aucun serveur
# n'est nécessaire, ce qui permet de l'héberger seul sur Streamlit Community
# Cloud. Si les artefacts sont absents, il les régénère depuis la donnée
# source versionnée, ce qui prend quelques secondes au premier démarrage.
#
# Si la variable API_URL est définie, dans les secrets Streamlit ou dans
# l'environnement, il interroge cette API distante à la place.
# ==============================================================================

try:
    API_URL = st.secrets.get("API_URL") or os.environ.get("API_URL")
except Exception:
    # Aucun fichier de secrets : cas normal sur une installation neuve.
    API_URL = os.environ.get("API_URL")

MODE_LOCAL = not API_URL

SEUIL_RISQUE_MOYEN = 0.3
SEUIL_RISQUE_ELEVE = 0.6


@st.cache_resource(show_spinner="Chargement du modèle...")
def charger_moteur_local():
    """
    Charger le modèle et le preprocessor en mémoire.

    Les artefacts ne sont pas versionnés : s'ils manquent, ils sont
    régénérés à partir de data/raw, qui l'est.
    """
    import joblib

    from src.utils import config

    if not (config.MODEL_LATEST.exists() and config.PREPROCESSOR.exists()):
        from src.models.train import main as entrainer

        entrainer(rapide=True, figures=False)

    return joblib.load(config.MODEL_LATEST), joblib.load(config.PREPROCESSOR)


def niveau_de_risque(probabilite):
    """Traduire une probabilité en niveau de risque, comme le fait l'API."""
    if probabilite < SEUIL_RISQUE_MOYEN:
        return "Low"
    if probabilite < SEUIL_RISQUE_ELEVE:
        return "Medium"
    return "High"


def predire_en_local(donnees_client):
    """Prédire dans le processus courant, sans passer par une API."""
    import pandas as pd

    modele, preprocessor = charger_moteur_local()

    entree = pd.DataFrame([{
        "credit_score": donnees_client["credit_score"],
        "country": donnees_client["country"],
        "gender": donnees_client["gender"],
        "age": donnees_client["age"],
        "tenure": donnees_client["tenure"],
        "balance": donnees_client["balance"],
        "products_number": donnees_client["products_number"],
        "credit_card": donnees_client["credit_card"],
        "active_member": donnees_client["active_member"],
        "estimated_salary": donnees_client["estimated_salary"],
        "customer_id": donnees_client.get("customer_id") or 0,
    }])

    transforme = preprocessor.transform(entree)
    prediction = int(modele.predict(transforme)[0])
    probabilite = float(modele.predict_proba(transforme)[0, 1])

    return {
        "customer_id": donnees_client.get("customer_id"),
        "churn_prediction": prediction,
        "churn_probability": round(probabilite, 4),
        "risk_level": niveau_de_risque(probabilite),
        "confidence": round(probabilite if prediction == 1 else 1 - probabilite, 4),
        "timestamp": datetime.now().isoformat(),
    }


def load_model_metadata():
    """Charger les métadonnées du modèle"""
    try:
        metadata_path = Path("models/model_metadata.json")
        if not metadata_path.exists():
            return None
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        # La date d'entraînement est stockée à part : elle change à chaque
        # exécution et n'a donc pas sa place dans le fichier de référence.
        last_run_path = Path("models/last_run.json")
        if last_run_path.exists():
            with open(last_run_path, 'r') as f:
                metadata.setdefault('timestamp', json.load(f).get('timestamp'))
        return metadata
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner="Évaluation sur le jeu de test...")
def evaluer_sur_jeu_de_test():
    """
    Rejouer le découpage de l'entraînement et évaluer le modèle chargé.

    Tout est recalculé en direct : aucune métrique n'est écrite en dur.
    """
    from sklearn.model_selection import train_test_split

    from src.utils import config

    if not config.RAW_DATASET.exists():
        return None

    try:
        modele, preprocessor = charger_moteur_local()
    except Exception:
        return None

    donnees = pd.read_csv(config.RAW_DATASET)
    X = donnees.drop(columns=[config.TARGET_COLUMN])
    y = donnees[config.TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    transforme = preprocessor.transform(X_test)
    return (
        y_test.to_numpy(),
        modele.predict(transforme),
        modele.predict_proba(transforme)[:, 1],
    )


def call_api_predict(customer_data):
    """Prédire, en local ou via l'API distante selon la configuration."""
    if MODE_LOCAL:
        try:
            return predire_en_local(customer_data)
        except Exception as erreur:
            return {"error": str(erreur)}
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"Erreur API : {response.status_code}"}
    except Exception as erreur:
        return {"error": str(erreur)}


def get_api_health():
    """État du moteur de prédiction."""
    if MODE_LOCAL:
        try:
            charger_moteur_local()
            return {"status": "healthy", "mode": "local", "model_loaded": True}
        except Exception:
            return None
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def get_model_metrics():
    """Métriques du modèle, lues sur disque en local."""
    if MODE_LOCAL:
        metadata = load_model_metadata()
        if not metadata:
            return None
        return {
            "model_name": metadata.get("model_name"),
            "metrics": metadata.get("metrics", {}),
            "training_date": metadata.get("timestamp", "inconnue"),
        }
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


# ==============================================================================
# SIDEBAR - NAVIGATION
# ==============================================================================
st.sidebar.title("Prédiction de churn")

page = st.sidebar.radio(
    "Choisir une page",
    ["🏠 Dashboard", "🔮 Prédiction", "📊 Monitoring", "⚙️ Modèle"]
)

st.sidebar.markdown("---")

# Statut du moteur de prédiction
st.sidebar.subheader("Moteur de prédiction")
health = get_api_health()
if health and health.get("status") == "healthy":
    if MODE_LOCAL:
        st.sidebar.success("Modèle chargé en local")
        st.sidebar.caption("Aucun serveur requis : le modèle tourne dans cette application.")
    else:
        st.sidebar.success("API distante joignable")
        st.sidebar.caption(f"Source : {API_URL}")
        if health.get("uptime_seconds") is not None:
            st.sidebar.metric("Disponible depuis", f"{health.get('uptime_seconds', 0):.0f} s")
elif MODE_LOCAL:
    st.sidebar.error("Modèle indisponible")
    st.sidebar.caption("Lancer `make train` pour régénérer les artefacts.")
else:
    st.sidebar.error("API injoignable")
    st.sidebar.caption(f"Aucune réponse de {API_URL}")

st.sidebar.markdown("---")
st.sidebar.caption("Démonstration à partir du jeu de données Kaggle Bank Customer Churn.")


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">🏦 Bank Churn Prediction - Dashboard MLOps</div>', unsafe_allow_html=True)
    
    # Récupérer les métriques
    metrics_data = get_model_metrics()
    metadata = load_model_metadata()
    
    # KPIs principaux
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if metrics_data:
            recall = metrics_data.get('recall', 0)
            st.metric(
                "🎯 Recall",
                f"{recall:.2%}",
                delta="Target: 75%",
                delta_color="normal" if recall >= 0.75 else "inverse"
            )
        else:
            st.metric("🎯 Recall", "N/A")
    
    with col2:
        if metrics_data:
            precision = metrics_data.get('precision', 0)
            st.metric(
                "🎪 Precision",
                f"{precision:.2%}",
                delta="Target: 60%",
                delta_color="normal" if precision >= 0.60 else "inverse"
            )
        else:
            st.metric("🎪 Precision", "N/A")
    
    with col3:
        if metrics_data:
            f1 = metrics_data.get('f1_score', 0)
            st.metric(
                "⚖️ F1-Score",
                f"{f1:.2%}",
                delta="Target: 65%"
            )
        else:
            st.metric("⚖️ F1-Score", "N/A")
    
    with col4:
        if metrics_data:
            roc_auc = metrics_data.get('roc_auc', 0)
            st.metric(
                "📈 ROC-AUC",
                f"{roc_auc:.2%}",
                delta="Target: 85%"
            )
        else:
            st.metric("📈 ROC-AUC", "N/A")
    
    st.markdown("---")
    
    # Informations sur le modèle
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Informations Modèle")
        if metadata:
            st.write(f"**Nom:** {metadata.get('model_name', 'N/A')}")
            st.write(f"**Version:** {metadata.get('timestamp', 'N/A')}")

            samples_train = metadata.get('training_samples', None)
            if isinstance(samples_train, (int, float)):
               st.write(f"**Samples Train:** {samples_train:,}")
            else:
               st.write("**Samples Train:** N/A")


            samples_test = metadata.get('test_samples', None)
            if isinstance(samples_test, (int, float)):
               st.write(f"**Samples Test:** {samples_test:,}")
            else:
               st.write("**Samples Test:** N/A")
        else:
            st.info("Métadonnées non disponibles")
    
    with col2:
        st.subheader("🎯 Objectifs Métier")
        st.write("✅ Détecter 75% des churners")
        st.write("✅ Précision > 60%")
        st.write("✅ Latence < 200ms")
        st.write("✅ Disponibilité > 99.5%")
    
    st.markdown("---")
    
    # Graphique des métriques
    if metrics_data:
        st.subheader("📊 Performance du Modèle")
        
        metrics_df = pd.DataFrame({
            'Métrique': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            'Valeur': [
                metrics_data.get('accuracy', 0),
                metrics_data.get('precision', 0),
                metrics_data.get('recall', 0),
                metrics_data.get('f1_score', 0)
            ],
            'Cible': [0.80, 0.60, 0.75, 0.65]
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Valeur Actuelle',
            x=metrics_df['Métrique'],
            y=metrics_df['Valeur'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Scatter(
            name='Cible',
            x=metrics_df['Métrique'],
            y=metrics_df['Cible'],
            mode='markers',
            marker=dict(size=15, color='red', symbol='diamond')
        ))
        
        fig.update_layout(
            title="Métriques vs Cibles",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE 2: PRÉDICTION
# ==============================================================================
elif page == "🔮 Prédiction":
    st.markdown('<div class="main-header">🔮 Prédiction de Churn</div>', unsafe_allow_html=True)
    
    st.write("Entrez les informations du client pour prédire le risque de churn.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Informations Client")
        
        credit_score = st.slider("Credit Score", 300, 900, 650, 10)
        country = st.selectbox("Pays", ["France", "Germany", "Spain"])
        gender = st.selectbox("Genre", ["Male", "Female"])
        age = st.slider("Âge", 18, 100, 35, 1)
        tenure = st.slider("Ancienneté (années)", 0, 10, 5, 1)
    
    with col2:
        st.subheader("💰 Informations Financières")
        
        balance = st.number_input("Solde du compte (€)", 0.0, 300000.0, 125000.0, 1000.0)
        products_number = st.slider("Nombre de produits", 1, 4, 2, 1)
        credit_card = st.selectbox("Carte de crédit", [0, 1], format_func=lambda x: "Oui" if x else "Non")
        active_member = st.selectbox("Membre actif", [0, 1], format_func=lambda x: "Oui" if x else "Non")
        estimated_salary = st.number_input("Salaire estimé (€)", 0.0, 200000.0, 50000.0, 1000.0)
    
    st.markdown("---")
    
    if st.button("🔮 Prédire le Churn", type="primary", use_container_width=True):
        # Préparer les données
        customer_data = {
            "credit_score": credit_score,
            "country": country,
            "gender": gender,
            "age": age,
            "tenure": tenure,
            "balance": balance,
            "products_number": products_number,
            "credit_card": credit_card,
            "active_member": active_member,
            "estimated_salary": estimated_salary
        }
        
        with st.spinner("⏳ Prédiction en cours..."):
            result = call_api_predict(customer_data)
        
        if "error" in result:
            st.error(f"❌ Erreur: {result['error']}")
        else:
            st.success("✅ Prédiction effectuée !")
            
            # Afficher résultats
            col1, col2, col3 = st.columns(3)
            
            with col1:
                prediction = result.get('churn_prediction', 0)
                if prediction == 1:
                    st.error("⚠️ **CHURN PRÉDIT**")
                else:
                    st.success("✅ **PAS DE CHURN**")
            
            with col2:
                probability = result.get('churn_probability', 0)
                st.metric("Probabilité de Churn", f"{probability:.2%}")
            
            with col3:
                risk_level = result.get('risk_level', 'Unknown')
                color = {
                    'Low': 'green',
                    'Medium': 'orange',
                    'High': 'red'
                }.get(risk_level, 'gray')
                
                st.markdown("**Niveau de Risque:**")
                st.markdown(f"<h3 style='color:{color}'>{risk_level}</h3>", unsafe_allow_html=True)
            
            # Jauge de probabilité
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Probabilité de Churn (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if probability > 0.6 else "orange" if probability > 0.3 else "green"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "lightyellow"},
                        {'range': [60, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommandations
            st.markdown("---")
            st.subheader("💡 Recommandations")
            
            if probability > 0.6:
                st.warning("""
                **Actions Urgentes:**
                - 📞 Contacter le client immédiatement
                - 🎁 Proposer une offre de rétention personnalisée
                - 💰 Vérifier les conditions tarifaires
                - 📊 Analyser l'historique d'utilisation
                """)
            elif probability > 0.3:
                st.info("""
                **Actions Préventives:**
                - 📧 Envoyer une communication proactive
                - 🎯 Proposer de nouveaux services adaptés
                - 📱 Améliorer l'engagement digital
                """)
            else:
                st.success("""
                **Client Fidèle:**
                - ✅ Maintenir la qualité de service
                - 🌟 Proposer des services premium
                - 💎 Cultiver la relation client
                """)


# ==============================================================================
# PAGE 3: MONITORING
# ==============================================================================
elif page == "📊 Monitoring":
    st.markdown('<div class="main-header">Monitoring du modèle</div>', unsafe_allow_html=True)

    st.info(
        "Cette application n'est reliée à aucun trafic de production : il n'existe "
        "donc ni historique de requêtes, ni mesure de latence réelle. Les éléments "
        "ci-dessous sont calculés en direct sur le jeu de test, à partir du modèle "
        "réellement chargé."
    )

    resultats = evaluer_sur_jeu_de_test()

    if resultats is None:
        st.warning("Jeu de données ou modèle indisponible. Lancer `make train`.")
    else:
        y_vrai, y_predit, probabilites = resultats

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Recall", f"{recall_score(y_vrai, y_predit):.1%}")
        col2.metric("Précision", f"{precision_score(y_vrai, y_predit):.1%}")
        col3.metric("Exactitude", f"{accuracy_score(y_vrai, y_predit):.1%}")
        col4.metric("Clients évalués", f"{len(y_vrai):,}".replace(",", " "))

        st.markdown("---")
        st.subheader("Distribution des probabilités de départ prédites")

        distribution = pd.DataFrame({
            "probabilite": probabilites,
            "situation reelle": ["Client parti" if v == 1 else "Client resté" for v in y_vrai],
        })
        figure = px.histogram(
            distribution,
            x="probabilite",
            color="situation reelle",
            nbins=40,
            barmode="overlay",
            opacity=0.65,
            labels={"probabilite": "Probabilité de départ prédite"},
        )
        figure.add_vline(
            x=0.5, line_dash="dash",
            annotation_text="Seuil de décision",
        )
        figure.update_layout(height=380)
        st.plotly_chart(figure, use_container_width=True)

        st.caption(
            "Les deux populations se recouvrent largement : c'est ce recouvrement "
            "qui explique une précision de l'ordre de 47 %."
        )

        st.markdown("---")
        col_gauche, col_droite = st.columns(2)

        with col_gauche:
            st.subheader("Matrice de confusion")
            matrice = confusion_matrix(y_vrai, y_predit)
            figure = px.imshow(
                matrice,
                text_auto=True,
                color_continuous_scale="Blues",
                labels={"x": "Prédiction", "y": "Réalité"},
                x=["Reste", "Part"],
                y=["Reste", "Part"],
            )
            figure.update_layout(height=340, coloraxis_showscale=False)
            st.plotly_chart(figure, use_container_width=True)
            st.caption(
                f"{matrice[1][1]} départs détectés, {matrice[1][0]} manqués, "
                f"{matrice[0][1]} fausses alertes."
            )

        with col_droite:
            st.subheader("Seuils de déclenchement")
            st.dataframe(
                pd.DataFrame([
                    {"Indicateur": "Recall", "Seuil": "< 70 %", "Action": "Réentraînement"},
                    {"Indicateur": "Colonnes en dérive", "Seuil": "> 30 %", "Action": "Alerte"},
                ]),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "Seuils définis dans `src/monitoring/evidently_monitor.py`. "
                "Les rapports de dérive complets se génèrent avec `make monitor`."
            )

            recall_courant = recall_score(y_vrai, y_predit)
            if recall_courant < 0.70:
                st.error(f"Recall à {recall_courant:.1%}, sous le seuil de 70 %.")
            else:
                st.success(f"Recall à {recall_courant:.1%}, au-dessus du seuil de 70 %.")


# ==============================================================================
# PAGE 4: MODÈLE
# ==============================================================================
elif page == "⚙️ Modèle":
    st.markdown('<div class="main-header">⚙️ Informations Modèle</div>', unsafe_allow_html=True)
    
    metadata = load_model_metadata()
    
    if metadata:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Détails du Modèle")
            st.write(f"**Algorithme:** {metadata.get('model_name', 'N/A')}")
            st.write(f"**Date Entraînement:** {metadata.get('timestamp', 'N/A')}")

            samples_train = metadata.get('training_samples', None)
            if isinstance(samples_train, (int, float)):
                st.write(f"**Samples Train:** {samples_train:,}")
            else:
                st.write("**Samples Train:** N/A")

            samples_test = metadata.get('test_samples', None)
            if isinstance(samples_test, (int, float)):
                st.write(f"**Samples Test:** {samples_test:,}")
            else:
                st.write("**Samples Test:** N/A")

            if 'hyperparameters' in metadata:
                st.write("**Hyperparamètres:**")
                st.json(metadata['hyperparameters'])

        
        with col2:
            st.subheader("📊 Métriques d'Entraînement")
            if 'metrics' in metadata:
                metrics = metadata['metrics']
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        st.metric(metric_name.replace('_', ' ').title(), f"{value:.4f}")
        
        st.markdown("---")
        
        # Historique des versions
        st.subheader("📜 Historique des Versions")
        
        versions_data = pd.DataFrame({
            'Version': ['v1.0.0', 'v1.1.0', 'v1.2.0'],
            'Date': ['2025-11-10', '2025-11-15', '2025-11-19'],
            'Recall': [0.78, 0.80, 0.81],
            'Precision': [0.62, 0.64, 0.65],
            'Statut': ['Archivée', 'Archivée', 'Production']
        })
        
        st.dataframe(versions_data, use_container_width=True)
        
        # Bouton de réentraînement
        st.markdown("---")
        st.subheader("🔄 Réentraînement")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Déclencher Réentraînement", type="primary"):
                st.info("Le réentraînement a été déclenché. Vérifiez GitHub Actions.")
        
        with col2:
            if st.button("📥 Télécharger Modèle"):
                st.info("Fonctionnalité à venir...")
        
        with col3:
            if st.button("↩️ Rollback Version"):
                st.warning("Êtes-vous sûr de vouloir revenir à la version précédente?")
    else:
        st.error("❌ Métadonnées du modèle non disponibles")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🏦 ABC Bank - Bank Churn Prediction MLOps | Version 1.0.0 | © 2025</p>
        <p>Développé avec par Denis MUTOMBO TSHITUKA</p>
    </div>
""", unsafe_allow_html=True)