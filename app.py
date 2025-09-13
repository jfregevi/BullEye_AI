import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import io

# Configuration de la page
st.set_page_config(
    page_title="Assistant Investissement",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration des profils d'investissement
PROFILS_INVESTISSEMENT = {
    "Prudent": {
        "tolerance_risque": 0.3,
        "seuil_reequilibrage": 0.15,
        "max_actif_unique": 0.40,
        "allocation_preferee": {"BTC": 0.20, "ETH": 0.15, "S&P500": 0.45, "Obligations": 0.20}
    },
    "Équilibré": {
        "tolerance_risque": 0.5,
        "seuil_reequilibrage": 0.20,
        "max_actif_unique": 0.50,
        "allocation_preferee": {"BTC": 0.30, "ETH": 0.20, "S&P500": 0.35, "Tesla": 0.15}
    },
    "Agressif": {
        "tolerance_risque": 0.8,
        "seuil_reequilibrage": 0.25,
        "max_actif_unique": 0.60,
        "allocation_preferee": {"BTC": 0.40, "ETH": 0.25, "Tesla": 0.20, "S&P500": 0.15}
    }
}

# Données de marché simulées
PRIX_ECHANTILLON = {
    "BTC": {"actuel": 45000, "historique": np.random.normal(45000, 5000, 252).tolist()},
    "ETH": {"actuel": 3000, "historique": np.random.normal(3000, 400, 252).tolist()},
    "S&P500": {"actuel": 4500, "historique": np.random.normal(4500, 200, 252).tolist()},
    "Tesla": {"actuel": 250, "historique": np.random.normal(250, 30, 252).tolist()},
    "Obligations": {"actuel": 100, "historique": np.random.normal(100, 5, 252).tolist()}
}

# Signaux IA (simulés)
SIGNAUX_IA = {
    "BTC": {"sentiment": "Haussier", "force": 0.8, "raison": "Adoption institutionnelle forte"},
    "ETH": {"sentiment": "Neutre", "force": 0.5, "raison": "En attente de mises à jour majeures"},
    "S&P500": {"sentiment": "Haussier", "force": 0.6, "raison": "Indicateurs de reprise économique"},
    "Tesla": {"sentiment": "Baissier", "force": -0.3, "raison": "Préoccupations concurrentielles"},
    "Obligations": {"sentiment": "Neutre", "force": 0.1, "raison": "Taux d'intérêt stables"}
}

class GestionnairePortefeuille:
    def __init__(self):
        if 'portefeuille' not in st.session_state:
            st.session_state.portefeuille = {}
        if 'historique_investissements' not in st.session_state:
            st.session_state.historique_investissements = []
    
    def ajouter_position(self, actif, montant, prix):
        if actif in st.session_state.portefeuille:
            valeur_actuelle = st.session_state.portefeuille[actif]['parts'] * st.session_state.portefeuille[actif]['prix_moyen']
            nouvelles_parts = montant / prix
            total_parts = st.session_state.portefeuille[actif]['parts'] + nouvelles_parts
            nouveau_prix_moyen = (valeur_actuelle + montant) / total_parts
            st.session_state.portefeuille[actif] = {'parts': total_parts, 'prix_moyen': nouveau_prix_moyen}
        else:
            st.session_state.portefeuille[actif] = {'parts': montant / prix, 'prix_moyen': prix}
    
    def obtenir_valeur_portefeuille(self):
        valeur_totale = 0
        for actif, position in st.session_state.portefeuille.items():
            if actif in PRIX_ECHANTILLON:
                valeur_totale += position['parts'] * PRIX_ECHANTILLON[actif]['actuel']
        return valeur_totale
    
    def obtenir_allocation_portefeuille(self):
        valeur_totale = self.obtenir_valeur_portefeuille()
        if valeur_totale == 0:
            return {}
        
        allocation = {}
        for actif, position in st.session_state.portefeuille.items():
            if actif in PRIX_ECHANTILLON:
                valeur = position['parts'] * PRIX_ECHANTILLON[actif]['actuel']
                allocation[actif] = valeur / valeur_totale
        return allocation

def creer_graphique_portefeuille(gestionnaire_portefeuille):
    allocation = gestionnaire_portefeuille.obtenir_allocation_portefeuille()
    if not allocation:
        st.info("Aucune donnée de portefeuille disponible")
        return
    
    # Créer un graphique en secteurs avec matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(allocation)))
    
    wedges, texts, autotexts = ax.pie(
        allocation.values(),
        labels=allocation.keys(),
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    
    ax.set_title("Allocation Actuelle du Portefeuille", fontsize=16, fontweight='bold')
    
    # Améliorer la lisibilité
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    st.pyplot(fig)

def creer_graphique_performance():
    # Générer des données de performance simulées
    dates = pd.date_range(start=datetime.now() - timedelta(days=252), end=datetime.now(), freq='D')
    valeurs_portefeuille = np.cumsum(np.random.normal(0, 50, len(dates))) + 10000
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, valeurs_portefeuille, linewidth=2, color='#1f77b4')
    ax.set_title("Performance du Portefeuille dans le Temps", fontsize=16, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur du Portefeuille (€)")
    ax.grid(True, alpha=0.3)
    
    # Rotation des dates pour une meilleure lisibilité
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)

def simuler_dca(montant, periode, mois, utiliser_ia=False):
    """Simuler la stratégie DCA avec amélioration IA optionnelle"""
    periodes_par_mois = {"hebdomadaire": 4, "mensuelle": 1, "trimestrielle": 0.33}
    investissements_par_mois = periodes_par_mois[periode]
    montant_par_investissement = montant / investissements_par_mois
    
    resultats = []
    total_investi = 0
    total_parts = 0
    
    for mois_num in range(mois):
        investissements_ce_mois = int(investissements_par_mois) if periode != "trimestrielle" else (1 if mois_num % 3 == 0 else 0)
        
        for _ in range(investissements_ce_mois):
            # Simuler le prix (en production, utiliser des données historiques réelles)
            prix = 45000 + np.random.normal(0, 2000)  # Exemple BTC
            
            if utiliser_ia:
                # Ajustement IA basé sur le sentiment
                multiplicateur_ia = 1 + (SIGNAUX_IA["BTC"]["force"] * 0.2)
                montant_investissement = montant_par_investissement * multiplicateur_ia
            else:
                montant_investissement = montant_par_investissement
            
            parts_achetees = montant_investissement / prix
            total_investi += montant_investissement
            total_parts += parts_achetees
            
            resultats.append({
                "mois": mois_num,
                "prix": prix,
                "investi": montant_investissement,
                "parts": parts_achetees,
                "valeur_totale": total_parts * prix
            })
    
    return pd.DataFrame(resultats)

def generer_suggestion_allocation(profil, montant_dca):
    """Générer une suggestion d'allocation alimentée par l'IA"""
    allocation_base = PROFILS_INVESTISSEMENT[profil]["allocation_preferee"]
    suggestions = {}
    
    for actif, poids_base in allocation_base.items():
        if actif in SIGNAUX_IA:
            # Ajuster selon le signal IA
            ajustement_ia = SIGNAUX_IA[actif]["force"] * 0.1
            poids_ajuste = max(0.05, min(0.6, poids_base + ajustement_ia))
            suggestions[actif] = {
                "montant": round(montant_dca * poids_ajuste, 2),
                "poids": poids_ajuste,
                "raison": SIGNAUX_IA[actif]["raison"]
            }
    
    # Normaliser les poids pour qu'ils somment à 1
    poids_total = sum(s["poids"] for s in suggestions.values())
    for actif in suggestions:
        suggestions[actif]["poids"] /= poids_total
        suggestions[actif]["montant"] = round(montant_dca * suggestions[actif]["poids"], 2)
    
    return suggestions

def creer_alertes():
    """Générer des alertes d'investissement basées sur les signaux IA et le portefeuille"""
    alertes = []
    
    for actif, signal in SIGNAUX_IA.items():
        if abs(signal["force"]) > 0.6:  # Signal fort
            type_alerte = "Opportunité" if signal["force"] > 0 else "Avertissement"
            alertes.append({
                "type": type_alerte,
                "actif": actif,
                "message": f"{actif} montre un signal {signal['sentiment']} : {signal['raison']}",
                "force": abs(signal["force"])
            })
    
    return sorted(alertes, key=lambda x: x["force"], reverse=True)

def reponse_chatbot(question, gestionnaire_portefeuille):
    """Réponses simples du chatbot basées sur des mots-clés"""
    question_min = question.lower()
    
    if "pourquoi renforcer" in question_min or "pourquoi investir" in question_min:
        actif = None
        for a in ["btc", "eth", "tesla", "s&p500"]:
            if a in question_min:
                actif = a.upper()
                if actif == "S&P500":
                    actif = "S&P500"
                break
        
        if actif and actif in SIGNAUX_IA:
            signal = SIGNAUX_IA[actif]
            return f"Selon l'analyse IA, {actif} montre un signal {signal['sentiment']} avec une force de {signal['force']:.1f}. Raison : {signal['raison']}"
    
    elif "rendement" in question_min or "performance" in question_min:
        # Calculer un rendement fictif
        rendement_fictif = np.random.uniform(-5, 15)
        return f"Votre portefeuille a généré un rendement de {rendement_fictif:.1f}% sur la période analysée. C'est {'au-dessus' if rendement_fictif > 5 else 'en dessous'} de la moyenne du marché."
    
    elif "nouvel actif" in question_min or "intéressant" in question_min:
        return "Selon l'analyse actuelle du marché, considérez la diversification dans les ETF de marchés émergents ou les actions d'énergie renouvelable. Assurez-vous toujours que cela correspond à votre profil de risque."
    
    elif "allocation" in question_min:
        valeur_totale = gestionnaire_portefeuille.obtenir_valeur_portefeuille()
        return f"La valeur actuelle de votre portefeuille est de {valeur_totale:,.2f}€. Selon votre profil, envisagez un rééquilibrage si un seul actif dépasse vos limites de risque."
    
    else:
        return "Je peux vous aider avec l'analyse de portefeuille, les suggestions d'investissement et le suivi des performances. Essayez de poser des questions sur des actifs spécifiques, les rendements ou les conseils d'allocation !"

def main():
    st.title("💰 Assistant Investissement Alimenté par l'IA")
    st.markdown("---")
    
    # Initialiser le gestionnaire de portefeuille
    gestionnaire_portefeuille = GestionnairePortefeuille()
    
    # Configuration de la barre latérale
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Profil d'investissement
        profil = st.selectbox(
            "Profil d'Investissement",
            ["Prudent", "Équilibré", "Agressif"],
            help="Choisissez votre niveau de tolérance au risque"
        )
        
        # Paramètres DCA
        st.subheader("Paramètres DCA")
        montant_dca = st.number_input("Montant DCA Mensuel (€)", min_value=50, max_value=10000, value=200, step=50)
        periode_dca = st.selectbox("Fréquence DCA", ["hebdomadaire", "mensuelle", "trimestrielle"])
        
        # Actifs disponibles
        st.subheader("Actifs Disponibles")
        actifs_disponibles = list(PRIX_ECHANTILLON.keys())
        actifs_selectionnes = st.multiselect(
            "Sélectionner les actifs à suivre",
            actifs_disponibles,
            default=list(PROFILS_INVESTISSEMENT[profil]["allocation_preferee"].keys())
        )
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Portefeuille", "📈 Simulation DCA", "🚨 Alertes", "🤖 Chatbot IA"])
    
    with tab1:
        st.header("Aperçu du Portefeuille")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Section de saisie du portefeuille
            st.subheader("Ajouter une Position")
            with st.form("ajouter_position"):
                pos_actif = st.selectbox("Actif", actifs_selectionnes)
                pos_montant = st.number_input("Montant Investi (€)", min_value=0.0, step=10.0)
                pos_prix = st.number_input("Prix d'Achat", min_value=0.0, value=float(PRIX_ECHANTILLON.get(pos_actif, {}).get('actuel', 0)))
                
                if st.form_submit_button("Ajouter Position"):
                    if pos_montant > 0 and pos_prix > 0:
                        gestionnaire_portefeuille.ajouter_position(pos_actif, pos_montant, pos_prix)
                        st.success(f"Position {pos_actif} ajoutée !")
                        st.rerun()
            
            # Upload CSV
            st.subheader("Télécharger Portefeuille CSV")
            fichier_telecharge = st.file_uploader("Choisir le fichier CSV", type="csv")
            if fichier_telecharge is not None:
                try:
                    df = pd.read_csv(fichier_telecharge)
                    st.write("Aperçu :", df.head())
                    if st.button("Importer Portefeuille"):
                        for _, row in df.iterrows():
                            if all(col in df.columns for col in ['actif', 'montant', 'prix']):
                                gestionnaire_portefeuille.ajouter_position(row['actif'], row['montant'], row['prix'])
                        st.success("Portefeuille importé avec succès !")
                        st.rerun()
                except Exception as e:
                    st.error(f"Erreur de lecture CSV : {str(e)}")
        
        with col2:
            # Métriques du portefeuille
            valeur_totale = gestionnaire_portefeuille.obtenir_valeur_portefeuille()
            st.metric("Valeur Totale du Portefeuille", f"{valeur_totale:,.2f}€")
            
            if st.session_state.portefeuille:
                total_investi = sum(pos['parts'] * pos['prix_moyen'] for pos in st.session_state.portefeuille.values())
                pnl = valeur_totale - total_investi
                pnl_pct = (pnl / total_investi * 100) if total_investi > 0 else 0
                st.metric("P&L", f"{pnl:,.2f}€", f"{pnl_pct:+.1f}%")
        
        # Visualisation du portefeuille
        if st.session_state.portefeuille:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Allocation du Portefeuille")
                creer_graphique_portefeuille(gestionnaire_portefeuille)
            with col2:
                st.subheader("Performance Historique")
                creer_graphique_performance()
            
            # Tableau du portefeuille
            st.subheader("Détails des Positions")
            donnees_portefeuille = []
            for actif, position in st.session_state.portefeuille.items():
                if actif in PRIX_ECHANTILLON:
                    prix_actuel = PRIX_ECHANTILLON[actif]['actuel']
                    valeur_actuelle = position['parts'] * prix_actuel
                    pnl = valeur_actuelle - (position['parts'] * position['prix_moyen'])
                    donnees_portefeuille.append({
                        'Actif': actif,
                        'Parts': f"{position['parts']:.4f}",
                        'Prix Moyen': f"{position['prix_moyen']:.2f}€",
                        'Prix Actuel': f"{prix_actuel:.2f}€",
                        'Valeur Actuelle': f"{valeur_actuelle:.2f}€",
                        'P&L': f"{pnl:.2f}€"
                    })
            
            if donnees_portefeuille:
                st.dataframe(pd.DataFrame(donnees_portefeuille), use_container_width=True)
    
    with tab2:
        st.header("Simulation DCA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Paramètres de Simulation")
            sim_mois = st.slider("Période de Simulation (mois)", 6, 60, 12)
            montants_comparaison = st.multiselect(
                "Comparer les montants DCA (€)",
                [100, 200, 300, 400, 500, 1000],
                default=[200, 400]
            )
        
        with col2:
            st.subheader("Amélioration IA")
            utiliser_ia = st.checkbox("Activer le DCA amélioré par IA", value=True)
            st.info("L'amélioration IA ajuste les montants d'investissement selon le sentiment du marché et les indicateurs techniques.")
        
        if st.button("Lancer la Simulation"):
            resultats = {}
            for montant in montants_comparaison:
                resultats[f"{montant}€ Standard"] = simuler_dca(montant, periode_dca, sim_mois, False)
                if utiliser_ia:
                    resultats[f"{montant}€ IA-Amélioré"] = simuler_dca(montant, periode_dca, sim_mois, True)
            
            # Tracer les résultats
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = plt.cm.tab10(np.linspace(0, 1, len(resultats)))
            
            for i, (label, data) in enumerate(resultats.items()):
                ax.plot(data['mois'], data['valeur_totale'], 
                       label=label, linewidth=2, color=colors[i], marker='o')
            
            ax.set_title("Comparaison des Stratégies DCA", fontsize=16, fontweight='bold')
            ax.set_xlabel("Mois")
            ax.set_ylabel("Valeur du Portefeuille (€)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Statistiques de résumé
            st.subheader("Résumé de la Simulation")
            donnees_resume = []
            for label, data in resultats.items():
                valeur_finale = data['valeur_totale'].iloc[-1]
                total_investi = data['investi'].sum()
                rendement_total = (valeur_finale - total_investi) / total_investi * 100
                donnees_resume.append({
                    'Stratégie': label,
                    'Valeur Finale': f"{valeur_finale:,.2f}€",
                    'Total Investi': f"{total_investi:,.2f}€",
                    'Rendement Total': f"{rendement_total:+.1f}%"
                })
            
            st.dataframe(pd.DataFrame(donnees_resume), use_container_width=True)
    
    with tab3:
        st.header("Alertes Marché & Recommandations")
        
        # Suggestion d'allocation du mois en cours
        st.subheader(f"Allocation Suggérée pour {montant_dca}€ d'Investissement")
        suggestions = generer_suggestion_allocation(profil, montant_dca)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            donnees_suggestion = []
            for actif, suggestion in suggestions.items():
                donnees_suggestion.append({
                    'Actif': actif,
                    'Montant': f"{suggestion['montant']:.2f}€",
                    'Poids': f"{suggestion['poids']:.1%}",
                    'Raison': suggestion['raison']
                })
            
            st.dataframe(pd.DataFrame(donnees_suggestion), use_container_width=True)
        
        with col2:
            # Graphique en secteurs de l'allocation suggérée
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = plt.cm.Set2(np.linspace(0, 1, len(suggestions)))
            
            wedges, texts, autotexts = ax.pie(
                [s['montant'] for s in suggestions.values()],
                labels=list(suggestions.keys()),
                autopct='%1.1f%%',
                colors=colors,
                startangle=90
            )
            
            ax.set_title("Allocation Suggérée", fontsize=14, fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            st.pyplot(fig)
        
        # Alertes
        st.subheader("Alertes Marché")
        alertes = creer_alertes()
        
        for alerte in alertes:
            type_alerte = alerte['type']
            icone = "🚨" if type_alerte == "Avertissement" else "💡"
            couleur = "red" if type_alerte == "Avertissement" else "green"
            
            st.markdown(f"""
            <div style="padding: 10px; border-left: 4px solid {couleur}; background-color: rgba(0,0,0,0.05); margin: 5px 0;">
                {icone} <strong>{type_alerte} :</strong> {alerte['message']}
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.header("Chatbot IA Investissement")
        st.markdown("Posez-moi des questions sur votre portefeuille, les conditions du marché ou les stratégies d'investissement !")
        
        # Historique des conversations
        if 'historique_chat' not in st.session_state:
            st.session_state.historique_chat = []
        
        # Afficher l'historique des conversations
        for chat in st.session_state.historique_chat:
            with st.container():
                st.markdown(f"**Vous :** {chat['question']}")
                st.markdown(f"**IA :** {chat['reponse']}")
                st.markdown("---")
        
        # Saisie de chat
        question = st.text_input("Posez votre question :", placeholder="Pourquoi renforcer BTC ce mois ?")
        
        if st.button("Demander") and question:
            reponse = reponse_chatbot(question, gestionnaire_portefeuille)
            st.session_state.historique_chat.append({"question": question, "reponse": reponse})
            st.rerun()
        
        # Questions d'exemple
        st.subheader("Questions d'Exemple")
        questions_exemple = [
            "Pourquoi renforcer BTC ce mois ?",
            "Quel est mon rendement cumulé sur 6 mois ?",
            "Quel nouvel actif semble intéressant maintenant ?",
            "Dois-je rééquilibrer mon portefeuille ?",
            "Quelles sont les perspectives du marché pour Tesla ?"
        ]
        
        for qe in questions_exemple:
            if st.button(qe, key=f"exemple_{qe}"):
                reponse = reponse_chatbot(qe, gestionnaire_portefeuille)
                st.session_state.historique_chat.append({"question": qe, "reponse": reponse})
                st.rerun()

if __name__ == "__main__":
    main()
