import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

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

def prediction(ticker, window_size=20, forecast_days=365):
    # 1️⃣ Télécharger données historiques
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=3*365)  # prendre 3 ans pour être sûr
    df = yf.download(ticker, start=start_date, end=end_date).dropna()
    
    # 2️⃣ Ajouter les indicateurs techniques
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
    
    # 3️⃣ Préparer la dernière fenêtre
    last_window = df[feature_cols].iloc[-window_size:].values
    last_window_scaled = np.zeros_like(last_window)
    for i in range(len(feature_cols)):
        last_window_scaled[:,i] = scalers_X[i].transform(last_window[:,i].reshape(-1,1)).flatten()
    
    predictions = []
    last_input = last_window_scaled.copy().reshape(1, window_size, len(feature_cols))
    
    # 4️⃣ Boucle pour prédiction jour par jour
    for _ in range(forecast_days):
        pred_scaled = model.predict(last_input, verbose=0)
        pred = scaler_y.inverse_transform(pred_scaled)[0,0]
        predictions.append(pred)
        
        # Mise à jour de la fenêtre pour le prochain jour
        next_row_scaled = last_input[0][1:,:].copy()  # décaler la fenêtre
        next_row_scaled = np.vstack([next_row_scaled, last_input[0][-1,:]])  # placeholder pour la nouvelle ligne
        next_row_scaled[-1,3] = pred_scaled  # mettre le close prédit à la place
        last_input = next_row_scaled.reshape(1, window_size, len(feature_cols))
    
    # 5️⃣ Créer les dates correspondantes
    start_forecast = df.index[-1] + pd.Timedelta(days=1)
    forecast_dates = pd.date_range(start=start_forecast, periods=forecast_days, freq='B')  # jours ouvrés
    
    return pd.Series(predictions, index=forecast_dates, name=f"{ticker}_pred")
    


