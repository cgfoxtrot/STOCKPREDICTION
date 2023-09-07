import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import random
import tensorflow as tf

# Set the random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load the data from the CSV file
df = pd.read_csv('data.csv')

# Sort the data by date in ascending order
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Extract the 'Close' column as the time series data
data = df['Close'].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split the data into training and test sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Convert the time series data into input-output pairs
def create_dataset(data, window_size):
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        Y.append(data[i+window_size])
    return np.array(X), np.array(Y)

window_size = 100
train_X, train_Y = create_dataset(train_data, window_size)
test_X, test_Y = create_dataset(test_data, window_size)

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_X, train_Y, epochs=3, batch_size=32)

# Make predictions on the test data
predictions = model.predict(test_X)
predictions = scaler.inverse_transform(predictions)

# Calculate the root mean squared error
rmse = np.sqrt(np.mean((predictions - scaler.inverse_transform(test_Y))**2))
print('Root Mean Squared Error:', rmse)

# Plot the results
train = df[:train_size+window_size]
test = df[train_size+window_size:]
test['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model prediciton results - TESLA shares')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
plt.plot(train['Date'], train['Close'], color='red', label='Training Data')
plt.plot(test['Date'], test['Close'], color='blue', label='Actual Data')
plt.plot(test['Date'], test['Predictions'], color='green', label='Predictions')
plt.xlabel('Date')
plt.savefig('prediction.png')
plt.legend()

# Print the prices of the predictions
for date, pred_price in zip(test['Date'], test['Predictions']):
    print(f"Price on {date.date()}: {pred_price}")

plt.show()


# Get the last 60 days of data
last_60_days = df['Close'].values[-60:].reshape(-1, 1)

# Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

# Create an empty list
X_test = []
# Append the past 60 days
X_test.append(last_60_days_scaled)

# Convert the X_test data into numpy array
X_test = np.array(X_test)

# Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Get predicted scaled price
pred_price = model.predict(X_test)
# Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(f'Price for tomorrow: {pred_price[0][0]}')
