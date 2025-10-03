import streamlit as st

def main():
    st.set_page_config(
        page_title="Assistant Investissement",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💰 Assistant Investissement")
    st.markdown("---")
    
    # ------------------------------
    # Barre latérale : paramètres généraux
    # ------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        profil = st.selectbox(
            "Profil d'Investissement",
            ["Prudent", "Équilibré", "Agressif"]
        )
        
        montant_dca = st.number_input("Montant DCA Mensuel (€)", min_value=50, max_value=10000, value=200, step=50)
        periode_dca = st.selectbox("Fréquence DCA", ["hebdomadaire", "mensuelle", "trimestrielle"])
    
    # ------------------------------
    # Onglets principaux
    # ------------------------------
    tab_intro, tab_portefeuille, tab_actualites, tab_chatbot = st.tabs([
        "🏦 Introduction",
        "📊 Portefeuille",
        "📰 Actualités",
        "🤖 Chatbot IA"
    ])
    
    # ------------------------------
    # 1) Page d'introduction
    # ------------------------------
    with tab_intro:
        st.header("Bienvenue sur votre assistant d'investissement")
        st.markdown("""
        - Investissez régulièrement grâce à la stratégie DCA.
        - Maximisez vos gains avec notre modèle de prédiction IA à moyen terme.
        - Le marché croît sur le long terme, mais il y a des limites à court terme.
        """)
        st.info("Découvrez votre profil investisseur pour mieux orienter vos décisions !")
        # Bouton pour rediriger vers profil (à implémenter)
        st.button("Définir mon profil investisseur")
    
    # ------------------------------
    # 2) Page Portefeuille
    # ------------------------------
    with tab_portefeuille:
        st.header("Suivi du Portefeuille")
        st.markdown("Ajouter vos positions, importer un CSV et visualiser vos performances.")
        # Ici tu pourras appeler :
        # - gestionnaire_portefeuille.ajouter_position()
        # - creer_graphique_portefeuille()
        # - creer_graphique_performance()
    
    # ------------------------------
    # 3) Page Actualités économiques
    # ------------------------------
    with tab_actualites:
        st.header("Actualités sur les actifs")
        st.markdown("Affichage des performances récentes et des tendances du marché.")
        # Ici tu pourras appeler :
        # - récupérer performances des actions / crypto
        # - tracer des graphiques interactifs
    
    # ------------------------------
    # 4) Page Chatbot IA
    # ------------------------------
    with tab_chatbot:
        st.header("Chatbot Investissement")
        st.markdown("Posez des questions sur votre portefeuille, les tendances du marché ou votre stratégie.")
        # Ici tu pourras appeler :
        # - reponse_chatbot(question, gestionnaire_portefeuille)
        question = st.text_input("Posez votre question :")
        if st.button("Demander") and question:
            st.info("Réponse IA à implémenter ici")

if __name__ == "__main__":
    main()
