import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

def liste_valeurs(ticker, period):
    if not isinstance(ticker, str):
        print("Un ticker est un code d'actif. Veuillez recommencer")
        return None
    
    stock = yf.Ticker(ticker)  # utiliser la variable, pas une chaîne littérale
    historical_data = stock.history(period=period)
    return list(historical_data['Close'])

def prediction():
    


