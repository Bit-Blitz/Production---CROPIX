import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

# --- Configuration ---
DATA_FILE = 'central_india_weekly_crop_prices.csv'
MODEL_FILE = 'lstm_model.keras'
CROP_TO_VISUALIZE = 'Rice' # Change this to see plots for other crops

# Model parameters
N_STEPS = 8  # How many past weeks of data to use for a prediction
N_FEATURES = 8 # Number of crops we are tracking

def load_and_preprocess_data(file_path):
    """Loads and preprocesses the crop price data."""
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'")
        print("Please run the data generation script first.")
        return None, None, None
        
    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    return df, scaled_data, scaler

def create_sequences(data, n_steps, n_features):
    """Converts the time series data into sequences for the LSTM model."""
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix, :], data[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def build_and_train_model(X_train, y_train, n_steps, n_features, model_path):
    """Builds and trains the LSTM model, then saves it."""
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
        Dense(n_features)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    print("\n--- Training LSTM Model ---")
    model.fit(X_train, y_train, epochs=200, verbose=1)
    
    model.save(model_path)
    print(f"Model saved to '{model_path}'")
    return model

def make_predictions(model, start_data, n_forecast_weeks, scaler):
    """Makes future predictions week by week."""
    current_batch = start_data.reshape((1, N_STEPS, N_FEATURES))
    forecast = []
    
    for _ in range(n_forecast_weeks):
        current_pred = model.predict(current_batch, verbose=0)[0]
        forecast.append(current_pred)
        # Update the batch to include the new prediction for the next step
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        
    # Inverse transform the forecast back to original price scale
    forecast_prices = scaler.inverse_transform(forecast)
    return forecast_prices

def plot_forecast(df_historical, df_forecast, crop_name):
    """Plots the historical data and the forecast."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 8))
    
    plt.plot(df_historical.index, df_historical[crop_name], label='Historical Context', color='blue')
    plt.plot(df_forecast.index, df_forecast[crop_name], label='Forecasted Prices', linestyle='--', color='darkorange')
    
    # Add a vertical line to show where the forecast starts
    plt.axvline(df_historical.index[-1], color='red', linestyle='--', label='Forecast Start')

    plt.title(f'Future Forecast for {crop_name}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (INR per Quintal)', fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    df, scaled_data, scaler = load_and_preprocess_data(DATA_FILE)
    
    if df is not None:
        # Create sequences from the entire dataset for training
        X, y = create_sequences(scaled_data, N_STEPS, N_FEATURES)
        
        # --- Model Training or Loading ---
        if not os.path.exists(MODEL_FILE):
            model = build_and_train_model(X, y, N_STEPS, N_FEATURES, MODEL_FILE)
        else:
            print(f"Loading existing model from '{MODEL_FILE}'")
            model = load_model(MODEL_FILE)
            
        # --- Make a Future Forecast ---
        # Get user input for forecast duration
        while True:
            try:
                weeks_to_forecast = int(input("Enter the number of weeks to forecast ahead: "))
                if weeks_to_forecast > 0:
                    break
                else:
                    print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter an integer.")

        # Prepare the last known data as the starting point for the forecast
        last_known_data = scaled_data[-N_STEPS:]
        
        # Generate the forecast
        forecast_values = make_predictions(model, last_known_data, weeks_to_forecast, scaler)
        
        # Create a dataframe for the forecast with future dates
        last_date = df.index[-1]
        future_dates = pd.to_datetime([last_date + timedelta(weeks=i) for i in range(1, weeks_to_forecast + 1)])
        df_forecast = pd.DataFrame(forecast_values, index=future_dates, columns=df.columns)

        print("\n--- Forecasted Prices ---")
        print(df_forecast)
        
        # Plot the forecast for the selected crop
        plot_forecast(df, df_forecast, CROP_TO_VISUALIZE)
