import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime
import joblib

def liste_valeurs(ticker, period):
    if not isinstance(ticker, str):
        print("Un ticker est un code d'actif. Veuillez recommencer")
        return None
    
    stock = yf.Ticker(ticker)  # utiliser la variable, pas une chaîne littérale
    historical_data = stock.history(period=period)
    return list(historical_data['Close'])

# Charger les fichiers sauvegardés
model = load_model("Streamlit/Pages/modele_gru.h5", compile=False)
scalers_X = joblib.load("Streamlit/Pages/scalers_X.pkl")
scaler_y = joblib.load("Streamlit/Pages/scaler_y.pkl")

def create_windows(df, window_size=20, feature_cols=None):
    """
    Prépare les fenêtres multivariées pour la prédiction.
    """
    X = []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i+window_size][feature_cols].values
        X.append(window)
    return np.array(X)

import yfinance as yf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Charger ton modèle (pré-entraîné sur AAPL)
model = load_model("Streamlit/Pages/modele_gru.h5", compile=False)

def prediction(ticker, window_size=20, forecast_days=252):
    """
    Prédit le prix de l'actif sur forecast_days jours ouvrés à partir du ticker.
    """
    # 1️⃣ Télécharger données historiques
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=3*365)  # prendre 3 ans pour stabilité
    df = yf.download(ticker, start=start_date, end=end_date).dropna()
    
    # 2️⃣ Ajouter indicateurs techniques
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / roll_down
    df['RSI_14'] = 100 - (100 / (1 + RS))
    df.dropna(inplace=True)
    
    feature_cols = ["Open","High","Low","Close","Volume","SMA_10","EMA_10","RSI_14"]
    num_features = len(feature_cols)
    
    # 3️⃣ Préparer la dernière fenêtre
    last_window = df[feature_cols].iloc[-window_size:].values
    last_window = last_window.reshape(1, window_size, num_features)
    
    # 4️⃣ Boucle de prédiction
    predictions = []
    current_window = last_window.copy()
    
    for _ in range(forecast_days):
        pred = model.predict(current_window, verbose=0)[0,0]
        predictions.append(pred)
        
        # Mise à jour de la fenêtre
        next_row = current_window[0,1:,:].copy()
        new_row = current_window[0,-1,:].copy()
        new_row[3] = pred  # Close prédit
        next_row = np.vstack([next_row, new_row])
        current_window = next_row.reshape(1, window_size, num_features)
    
    # 5️⃣ Générer les dates
    start_forecast = df.index[-1] + pd.Timedelta(days=1)
    forecast_dates = pd.bdate_range(start=start_forecast, periods=forecast_days)
    
    return pd.Series(predictions, index=forecast_dates, name=f"{ticker}_pred")
    


