import yfinance as yf
import matlplotlib.pyplot as plt
import numpy as np

stock = yf.Ticker("AAPL")
historical_data = stock.history(period="1y")  # Last 1 year of data
print(historical_data)
