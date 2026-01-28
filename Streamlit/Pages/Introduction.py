import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

# 1. Chargement des modèles avec cache pour éviter de saturer la RAM
@st.cache_resource
def load_assets():
    # Chemins relatifs vers tes fichiers dans le repo
    model_path = "Streamlit/Pages/modele_gru.h5"
    scaler_x_path = "Streamlit/Pages/scalers_X.pkl"
    scaler_y_path = "Streamlit/Pages/scaler_y.pkl"
    
    model = load_model(model_path, compile=False)
    scalers_X = joblib.load(scaler_x_path)
    scaler_y = joblib.load(scaler_y_path)
    return model, scalers_X, scaler_y

model, scalers_X, scaler_y = load_assets()

@st.cache_data
def liste_valeurs(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        historical_data = stock.history(period=period)
        return list(historical_data['Close'])
    except:
        return []

def create_windows(df, window_size, feature_cols):
    X = []
    # On s'assure qu'on a assez de lignes pour créer au moins une fenêtre
    if len(df) < window_size:
        return np.array([])
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size][feature_cols].values
        X.append(window)
    return np.array(X)

@st.cache_data
def prediction2(ticker, window_size=20, forecast_days=14):
    # 1️⃣ Téléchargement des données
    # On prend une date fixe pour ton test ou pd.Timestamp.today() pour du temps réel
    end_date = pd.Timestamp("2025-05-15")  
    start_date = end_date - pd.Timedelta(days=400)
    
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty or len(df) < window_size + 20: 
        return [None, None, None]

    # 2️⃣ Indicateurs techniques
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean()
    
    # Sécurité division par zéro pour le RSI
    rs = roll_up / roll_down.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_14'] = df['RSI_14'].fillna(50) # Valeur neutre si calcul impossible
    
    df.dropna(inplace=True)

    # 3️⃣ Scaling (La partie délicate corrigée)
    feature_cols = ["Open","High","Low","Close","Volume","SMA_10","EMA_10","RSI_14"]
    
    X = create_windows(df, window_size, feature_cols)
    if X.size == 0: return [None, None, None]
    
    # IMPORTANT: dtype=np.float64 pour ne pas arrondir les scales
    X_scaled = np.zeros_like(X, dtype=np.float64)

    for i in range(len(feature_cols)):
        feature_slice = X[:, :, i] 
        # On force en (Total_points, 1) pour le scaler Scikit-Learn
        flattened = feature_slice.reshape(-1, 1)
        scaled_flattened = scalers_X[i].transform(flattened)
        # On remet en (Nb_fenetres, Window_size)
        X_scaled[:, :, i] = scaled_flattened.reshape(feature_slice.shape)

    # 4️⃣ Prédiction
    # On prend la dernière fenêtre (la plus proche de la date de prédiction)
    current_window = X_scaled[-1].reshape(1, window_size, len(feature_cols))
    pred_scaled = model.predict(current_window, verbose=0)[0,0]
    
    # On repasse en prix réel
    pred_final = float(scaler_y.inverse_transform([[pred_scaled]])[0,0])

    # 5️⃣ Valeurs réelles pour le tableau final
    current_price = float(df['Close'].iloc[-1])
    # Prix d'il y a 14 jours ouvrés
    real_value_past = float(df['Close'].iloc[-forecast_days]) if len(df) >= forecast_days else None

    return [pred_final, real_value_past, current_price]
