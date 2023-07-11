import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import yfinance as yf

# Set the random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Get the stock ticker symbol from the user
ticker = input("Enter the stock ticker symbol: ")

# Define the date range
start_date = '2012-01-01'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

# Fetch the historical stock data from Yahoo Finance
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Sort the data by date in ascending order
stock_data.sort_index(ascending=True, inplace=True)

# Extract the 'Close' column as the time series data
data = stock_data['Close'].values.reshape(-1, 1)

# INTELLECTUAL PROPERTY...
