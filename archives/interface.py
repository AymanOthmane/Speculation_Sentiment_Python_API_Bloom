import streamlit as st
import numpy as np
import time

st.title("Speculation Sentiment")
onglet = st.sidebar.radio("Navigation", ["Réplication", "Stratégie"])

if onglet == "Réplication":
    st.markdown(
        """
        <style>
        .subheader-text {
            color: orange;
            font-size: 32px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<h1 class='subheader-text'>Réplication de l'article</h1>", unsafe_allow_html=True)
    st.subheader("Inputs : ")
    start_date = st.date_input("Veuillez choisir la date de début en YYYY-MM-DD",None)
    end_date = st.date_input("Veuillez choisir la date de fin en YYYY-MM-DD",None)
    # Définir la fonction Python à exécuter
    def ma_fonction(a,b):
        # Ici, vous pouvez placer n'importe quelle logique que vous souhaitez exécuter
        # en fonction de l'entrée de l'utilisateur
        resultat = (b-a).days
        return resultat

    # Ajouter un bouton pour exécuter la fonction
    if st.button("Run"):
        # Exécuter la fonction lorsque le bouton est cliqué
        output = ma_fonction(start_date,end_date)
        st.write("Résultat:", output, " days")

elif onglet == "Stratégie":
    st.write("Stratégie réalisée")
  