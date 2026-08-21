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
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from pathlib import Path

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

# Configuration
API_URL = st.secrets.get("API_URL", "http://localhost:8080")


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


def call_api_predict(customer_data):
    """Appeler l'API pour une prédiction"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=customer_data,
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def get_api_health():
    """Vérifier la santé de l'API"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def get_model_metrics():
    """Récupérer les métriques du modèle"""
    try:
        response = requests.get(f"{API_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


# ==============================================================================
# SIDEBAR - NAVIGATION
# ==============================================================================
st.sidebar.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=ABC+Bank", width='stretch')
st.sidebar.title("🏦 Navigation")

page = st.sidebar.radio(
    "Choisir une page",
    ["🏠 Dashboard", "🔮 Prédiction", "📊 Monitoring", "⚙️ Modèle"]
)

st.sidebar.markdown("---")

# Statut de l'API
st.sidebar.subheader("🔌 Statut API")
health = get_api_health()
if health:
    if health.get("status") == "healthy":
        st.sidebar.success("✅ API En ligne")
        st.sidebar.metric("Uptime", f"{health.get('uptime_seconds', 0):.0f}s")
    else:
        st.sidebar.warning("⚠️ API Dégradée")
else:
    st.sidebar.error("❌ API Hors ligne")

st.sidebar.markdown("---")
st.sidebar.info("**Version:** 1.0.0\n**Environnement:** Production")


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
        
        st.plotly_chart(fig, width='stretch')


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
    
    if st.button("🔮 Prédire le Churn", type="primary", width='stretch'):
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
                
                st.markdown(f"**Niveau de Risque:**")
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
            st.plotly_chart(fig, width='stretch')
            
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
    st.markdown('<div class="main-header">📊 Monitoring & Alertes</div>', unsafe_allow_html=True)
    
    # Simuler des données de monitoring
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    monitoring_data = pd.DataFrame({
        'date': dates,
        'recall': np.random.uniform(0.72, 0.82, len(dates)),
        'precision': np.random.uniform(0.58, 0.68, len(dates)),
        'latency_ms': np.random.uniform(80, 150, len(dates)),
        'requests': np.random.randint(500, 2000, len(dates))
    })
    
    # Graphique évolution recall
    st.subheader("📈 Évolution du Recall (30 derniers jours)")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monitoring_data['date'],
        y=monitoring_data['recall'],
        mode='lines+markers',
        name='Recall',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_hline(y=0.75, line_dash="dash", line_color="green", annotation_text="Cible: 75%")
    fig.add_hline(y=0.70, line_dash="dash", line_color="red", annotation_text="Alerte: 70%")
    
    fig.update_layout(
        yaxis_title="Recall",
        yaxis_range=[0.65, 0.85],
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Métriques de latence
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Latence API")
        avg_latency = monitoring_data['latency_ms'].mean()
        st.metric("Latence Moyenne", f"{avg_latency:.0f}ms", delta=f"{avg_latency - 100:.0f}ms")
        
        fig = px.line(monitoring_data, x='date', y='latency_ms', title="Latence API (30j)")
        fig.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="SLA: 200ms")
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("📊 Volume de Requêtes")
        total_requests = monitoring_data['requests'].sum()
        st.metric("Total Requêtes (30j)", f"{total_requests:,}")
        
        fig = px.bar(monitoring_data, x='date', y='requests', title="Requêtes journalières")
        st.plotly_chart(fig, width='stretch')
    
    # Alertes
    st.markdown("---")
    st.subheader("🚨 Alertes Actives")
    
    # Simuler quelques alertes
    if monitoring_data['recall'].iloc[-1] < 0.75:
        st.warning("⚠️ Recall sous la cible : 72% < 75%")
    
    if monitoring_data['latency_ms'].iloc[-1] > 180:
        st.warning("⚠️ Latence élevée : 185ms > 180ms")
    
    if monitoring_data['recall'].iloc[-1] >= 0.75 and monitoring_data['latency_ms'].iloc[-1] < 150:
        st.success("✅ Tous les indicateurs sont au vert")


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
        
        st.dataframe(versions_data, width='stretch')
        
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