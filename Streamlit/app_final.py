import streamlit as st
from Pages.Introduction import liste_valeurs
from Pages.Introduction import prediction
import matplotlib.pyplot as plt

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

        st.header("Pourquoi investir ?")
        st.markdown("""La raison est simple : vous êtes gagnat sur le long terme !
        Vous pouvez choisir des actifs juste en dessous pour vous convaincre que sur le long terme, chacune des courbes sera croissante.
        """)

        ticker = st.text_input("Entrez un ticker pour afficher son historique :", value="SPY")
        period = st.selectbox("Période :", ["1y", "5y", "10y", "20y"], index=0)

        # Faudrait se chauffer à rajouter les dates
        
        if st.button("Afficher l'historique"):
            data = liste_valeurs(ticker, period)
            if data is not None and len(data)>0:
                # Plot matplotlib
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(range(len(data)), data, label=f"{ticker} Close")
                ax.set_title(f"Historique des prix de {ticker}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Prix")
                ax.grid(True)
                ax.legend()
                
                st.pyplot(fig)
            else:
                st.warning("Aucune donnée disponible pour ce ticker.")

        forecast_days = 3
        windows_size = 20
        ticker = st.text_input("Entrez un ticker pour comparer la prédiction du modèle à la courbe réelle sur les valeurs de l'année passée :", value="SPY")
        st.line_chart(prediction(ticker, windows_size, forecast_days))

        
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
