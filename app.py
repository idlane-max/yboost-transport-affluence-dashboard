
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ----------------------------
# CHARGEMENT
# ----------------------------

modele = joblib.load("modele_xgb.pkl")
colonnes_X = joblib.load("colonnes_X.pkl")

# ----------------------------
# CONFIG PAGE
# ----------------------------

st.set_page_config(
    page_title="Prédiction Affluence",
    page_icon="🚇",
    layout="wide"
)

st.set_option('client.toolbarMode', 'minimal')

# ----------------------------
# HEADER
# ----------------------------

st.title("🚇 Dashboard intelligent de prédiction d'affluence")

st.markdown("""
Analyse prédictive de fréquentation des transports publics  
avec Machine Learning optimisé
""")

col1, col2, col3 = st.columns(3)

col1.metric("MAE modèle", "3.71%")
col2.metric("RMSE", "7.78%")
col3.metric("Modèle", "XGBoost optimisé")

st.divider()

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.header("Paramètres")

# Lignes réelles

# Extraction automatique de toutes les lignes du modèle

# Traduction lisible des codes transport

transport_labels = {
    100: "Bus Ligne 72",
    101: "Bus Ligne 91",
    510: "Métro Ligne 1",
    520: "Métro Ligne 4",
    800: "Métro Ligne 13",
    900: "Tram T3"
}

codes_lignes = []

for col in colonnes_X:
    if col.startswith("CODE_STIF_TRNS_"):
        code = int(col.split("_")[-1])
        codes_lignes.append(code)

codes_lignes = sorted(codes_lignes)

ligne_map = {
    transport_labels.get(code, f"Réseau ligne {code}"): code
    for code in codes_lignes
}

ligne = st.sidebar.selectbox(
    "Ligne",
    list(ligne_map.keys())
)

heure = st.sidebar.slider(
    "Heure",
    0,
    23,
    8
)

jour = st.sidebar.selectbox(
    "Type de jour",
    [
        "Jour ouvré hors vacances",
        "Jour ouvré vacances scolaires",
        "Samedi hors vacances",
        "Samedi vacances scolaires"
    ]
)

vacances = st.sidebar.checkbox("Vacances scolaires")
ferie = st.sidebar.checkbox("Jour férié")
event = st.sidebar.checkbox("Événement local")

meteo = st.sidebar.selectbox(
    "Météo",
    [
        "Beau temps",
        "Pluie",
        "Forte pluie"
    ]
)

# ----------------------------
# ENCODAGE
# ----------------------------

donnees = {col: 0 for col in colonnes_X}

donnees["Heure"] = heure
donnees["Heure_Debut"] = heure

col_ligne = f"CODE_STIF_TRNS_{ligne_map[ligne]}"

if col_ligne in donnees:
    donnees[col_ligne] = 1

mapping_jour = {
    "Jour ouvré hors vacances": "CAT_JOUR_JOHV",
    "Jour ouvré vacances scolaires": "CAT_JOUR_JOVS",
    "Samedi hors vacances": "CAT_JOUR_SAHV",
    "Samedi vacances scolaires": "CAT_JOUR_SAVS"
}

if mapping_jour[jour] in donnees:
    donnees[mapping_jour[jour]] = 1

donnees["Vacances_Scolaires"] = int(vacances)
donnees["Jour_Ferie"] = int(ferie)
donnees["Evenement_Local"] = int(event)

mapping_meteo = {
    "Beau temps": 0,
    "Pluie": 1,
    "Forte pluie": 2
}

donnees["Meteo"] = mapping_meteo[meteo]

df_entree = pd.DataFrame([donnees])

# ----------------------------
# PRÉDICTION
# ----------------------------

prediction = modele.predict(df_entree)[0]

# ----------------------------
# KPI
# ----------------------------

st.metric(
    "Affluence prédite",
    f"{prediction:.2f}%"
)

# ----------------------------
# INTERPRÉTATION
# ----------------------------

if prediction < 5:
    niveau = "Très faible 🔵"
elif prediction < 12:
    niveau = "Faible 🟢"
elif prediction < 20:
    niveau = "Moyenne 🟡"
elif prediction < 35:
    niveau = "Élevée 🟠"
else:
    niveau = "Très élevée 🔴"

st.subheader(f"Niveau : {niveau}")

# ----------------------------
# COURBE 24H
# ----------------------------

predictions = []

for h in range(24):

    donnees_h = donnees.copy()
    donnees_h["Heure"] = h

    pred = modele.predict(
        pd.DataFrame([donnees_h])
    )[0]

    predictions.append(pred)

df_graph = pd.DataFrame({
    "Heure": range(24),
    "Affluence": predictions
})

fig = px.line(
    df_graph,
    x="Heure",
    y="Affluence",
    title=f"Prévision d'affluence sur 24h — {ligne}"
)

fig.update_layout(
    xaxis_title="Heure",
    yaxis_title="Affluence (%)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ----------------------------
# INFOS
# ----------------------------

st.info(
    """
Le modèle prend en compte :

- heure
- ligne choisie
- météo
- vacances scolaires
- jours fériés
- événements locaux
- type de journée
"""
)

# ----------------------------
# FOOTER
# ----------------------------

st.markdown("---")

st.caption(
    "Projet Ybosst IA-Data — Prédiction d'affluence transport public | Python • XGBoost • Streamlit • Azure Ready"
)
