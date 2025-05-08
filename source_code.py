import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime

def fetch_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df[['Close']]

def preprocess_data(data, sequence_length=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(data, predictions, title='Stock Price Prediction'):
    plt.figure(figsize=(12,6))
    plt.plot(data, label='Actual Price')
    plt.plot(range(len(data)-len(predictions), len(data)), predictions, label='Predicted Price', color='red')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Parameters
    ticker = 'AAPL'
    start_date = '2015-01-01'
    end_date = datetime.date.today().strftime('%Y-%m-%d')
    sequence_length = 60
    epochs = 20

    # Load and preprocess data
    df = fetch_stock_data(ticker, start_date, end_date)
    data = df.values
    X, y, scaler = preprocess_data(data, sequence_length)

    # Build and train model
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)

    # Predict next day price
    last_sequence = data[-sequence_length:]
    last_scaled = scaler.transform(last_sequence)
    X_test = last_scaled.reshape((1, sequence_length, 1))
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    print(f"Predicted {ticker} closing price for next day: ${predicted_price[0][0]:.2f}")

    # Plot predictions on last 100 days
    recent_data = data[-(100+sequence_length):]
    X_recent, y_recent, _ = preprocess_data(recent_data, sequence_length)
    y_pred = model.predict(X_recent)
    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))

    plot_predictions(recent_data[sequence_length:], y_pred_rescaled)

