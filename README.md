# STOCKPREDICTION

# Stock Price Prediction with LSTM

This script uses the Long Short-Term Memory (LSTM) model, a type of recurrent neural network (RNN), to predict stock prices. LSTM is effective for analyzing time series data like stock prices because it can capture long-term dependencies and patterns.

## How it Works

1. The user is prompted to enter a stock ticker symbol.
2. Historical stock data is fetched from Yahoo Finance.
3. The data is preprocessed by scaling it between 0 and 1.
4. The dataset is split into training and test sets.
5. Input sequences and corresponding labels are created for the LSTM model.
6. The LSTM model is defined and trained on the training data.
7. Predictions are made on the test data.
8. The predictions are inverse transformed to obtain the actual stock prices.
9. The root mean squared error (RMSE) is calculated to evaluate the model's performance.
10. The results are plotted to visualize the training data, actual data, and predicted prices.
11. The predicted prices for the test data are printed.
12. The script also provides the predicted price for the next day.

## LSTM (Long Short-Term Memory)

LSTM is a type of recurrent neural network (RNN) architecture that can effectively capture long-term dependencies and patterns in sequential data. It is particularly useful for analyzing time series data like stock prices, where historical information is crucial for making accurate predictions. The LSTM model used in this script has multiple layers of LSTM cells and a dense output layer for predicting the stock prices.

## Why This Script is Amazing

- Uses state-of-the-art deep learning technique (LSTM) for stock price prediction.
- Fetches historical stock data from Yahoo Finance, ensuring up-to-date information.
- Provides accurate predictions and evaluates the model's performance using RMSE.
- Visualizes the results through a clear and informative plot.
- Offers the predicted price for the next day, enabling users to make informed decisions.

## How it Works

1. The user is prompted to enter a stock ticker symbol.
2. Historical stock data is fetched from Yahoo Finance.
3. The data is preprocessed by scaling it between 0 and 1.
4. The dataset is split into training and test sets.
5. Input sequences and corresponding labels are created for the LSTM model.
6. The LSTM model is defined and trained on the training data.
7. Predictions are made on the test data.
8. The predictions are inverse transformed to obtain the actual stock prices.
9. The root mean squared error (RMSE) is calculated to evaluate the model's performance.
10. The results are plotted to visualize the training data, actual data, and predicted prices.
11. The predicted prices for the test data are printed.
12. The script also provides the predicted price for the next day.

## LSTM (Long Short-Term Memory)

LSTM is a type of recurrent neural network (RNN) architecture that can effectively capture long-term dependencies and patterns in sequential data. It is particularly useful for analyzing time series data like stock prices, where historical information is crucial for making accurate predictions. The LSTM model used in this script has multiple layers of LSTM cells and a dense output layer for predicting the stock prices.

## Usage

1. Install the required dependencies by running: `pip install -r requirements.txt`.
2. Run the script: `LSTM.py`.
3. Enter the stock ticker symbol when prompted.
4. View the predicted stock prices, RMSE, and plot of the results.
5. The predicted price for the next day will also be displayed.

