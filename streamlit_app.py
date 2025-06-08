import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as si

# Stratégies et profils de Greeks
strategies = {
    "Long Call": {"Delta": "positive", "Gamma": "positive", "Vega": "positive", "Theta": "negative", "Rho": "positive"},
    "Long Put": {"Delta": "negative", "Gamma": "positive", "Vega": "positive", "Theta": "negative", "Rho": "negative"},
    "Short Call": {"Delta": "negative", "Gamma": "negative", "Vega": "negative", "Theta": "positive", "Rho": "negative"},
    "Short Put": {"Delta": "positive", "Gamma": "negative", "Vega": "negative", "Theta": "positive", "Rho": "positive"},
    "Straddle (Long)": {"Delta": "neutral", "Gamma": "positive", "Vega": "positive", "Theta": "negative", "Rho": "neutral"},
    "Strangle (Long)": {"Delta": "neutral", "Gamma": "positive", "Vega": "positive", "Theta": "negative", "Rho": "neutral"},
    "Iron Condor": {"Delta": "neutral", "Gamma": "negative", "Vega": "negative", "Theta": "positive", "Rho": "neutral"},
    "Butterfly": {"Delta": "neutral", "Gamma": "positive", "Vega": "negative", "Theta": "positive", "Rho": "neutral"},
    "Calendar Spread": {"Delta": "neutral", "Gamma": "low", "Vega": "positive", "Theta": "neutral", "Rho": "neutral"},
}

# Fonction de correspondance
def suggest_strategy(prefs):
    matches = []
    for name, greeks in strategies.items():
        score = 0
        total = 0
        for greek, desired in prefs.items():
            if desired == "any":
                continue
            total += 1
            if greeks.get(greek) == desired or (desired == "neutral" and greeks.get(greek) in ["neutral", "low"]):
                score += 1
        match_score = score / total if total > 0 else 0
        matches.append((match_score, name))
    matches.sort(reverse=True)
    return matches[:3]

# Pricing Black-Scholes

def black_scholes_price(option_type, S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1)
    else:
        return 0

# Fonctions de payoff simplifiées
def payoff(strategy, S, K=100, premium=10):
    if strategy == "Long Call":
        return np.maximum(S - K, 0) - premium
    elif strategy == "Long Put":
        return np.maximum(K - S, 0) - premium
    elif strategy == "Short Call":
        return -np.maximum(S - K, 0) + premium
    elif strategy == "Short Put":
        return -np.maximum(K - S, 0) + premium
    elif strategy == "Straddle (Long)":
        return np.maximum(S - K, 0) + np.maximum(K - S, 0) - 2 * premium
    elif strategy == "Strangle (Long)":
        return np.maximum(S - (K+10), 0) + np.maximum((K-10) - S, 0) - 1.5 * premium
    elif strategy == "Iron Condor":
        payoff = np.where(S < 90, S - 90, 0)
        payoff += np.where((S >= 90) & (S < 100), 0, 0)
        payoff += np.where((S >= 100) & (S < 110), 0, 0)
        payoff += np.where(S >= 110, 110 - S, 0)
        return payoff + premium
    elif strategy == "Butterfly":
        return np.maximum(S - 90, 0) - 2*np.maximum(S - K, 0) + np.maximum(S - 110, 0)
    elif strategy == "Calendar Spread":
        return -5 * np.sin((S - K) / 10) + 5
    else:
        return np.zeros_like(S)

# Interface utilisateur
st.set_page_config(page_title="Sélecteur de stratégie Greeks", layout="centered")
st.title("📊 Sélection de stratégie par profil Greek")
st.markdown("Choisissez vos préférences pour chaque Greek :")

choices = ["positive", "negative", "neutral", "any"]
user_prefs = {}

with st.form("greeks_form"):
    for greek in ["Delta", "Gamma", "Vega", "Theta", "Rho"]:
        user_prefs[greek] = st.selectbox(f"{greek} préféré :", choices, index=3, key=greek)

    st.markdown("---")
    st.markdown("**Paramètres du modèle Black-Scholes :**")
    K = st.number_input("Prix d'exercice (K)", value=100)
    premium = st.number_input("Prime de l'option (en points)", value=10)
    S0 = st.number_input("Prix actuel de l'actif (S0)", value=100.0)
    T = st.number_input("Temps jusqu'à échéance (en années)", value=1.0)
    r = st.number_input("Taux sans risque (r)", value=0.01)
    sigma = st.number_input("Volatilité implicite (σ)", value=0.2)

    submitted = st.form_submit_button("Suggérer des stratégies")

if submitted:
    top_strats = suggest_strategy(user_prefs)

    if top_strats:
        st.subheader("🎯 Stratégies suggérées")
        names = [x[1] for x in top_strats]
        scores = [x[0] for x in top_strats]

        fig1, ax1 = plt.subplots()
        bars = ax1.barh(names, scores, color='lightcoral')
        ax1.set_xlim(0, 1.1)
        ax1.set_xlabel("Score de correspondance")
        ax1.set_title("Top 3 stratégies")
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, f"{score:.2f}", va='center')
        ax1.invert_yaxis()
        st.pyplot(fig1)

        # Graphique de payoff
        st.subheader("💹 Payoff des stratégies")
        S = np.linspace(50, 150, 500)
        fig2, ax2 = plt.subplots()
        for strat in names:
            ax2.plot(S, payoff(strat, S, K, premium), label=strat)
        ax2.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax2.set_xlabel("Prix de l'action à maturité")
        ax2.set_ylabel("Payoff")
        ax2.set_title("Profils de gains/pertes")
        ax2.legend()
        st.pyplot(fig2)

        # Affichage pricing BS
        st.subheader("💰 Valeur théorique (Black-Scholes)")
        for strat in names:
            if "Call" in strat:
                price = black_scholes_price("call", S0, K, T, r, sigma)
            elif "Put" in strat:
                price = black_scholes_price("put", S0, K, T, r, sigma)
            else:
                price = "Complexe à modéliser"
            st.markdown(f"**{strat}**: {price} €")
    else:
        st.warning("Aucune stratégie ne correspond à vos préférences.")
