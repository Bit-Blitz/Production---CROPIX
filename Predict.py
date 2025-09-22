import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import json
import joblib
import requests
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- INITIALIZATION ---
load_dotenv()
print("Starting CHUNK-BASED Weather Forecaster...")

MODEL_DIR = 'Trained_models/LSTM' # Points to the new model directory
MODEL_PATH = os.path.join(MODEL_DIR, 'LSTM_model.keras')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.gz')
FEATURES_PATH = os.path.join(MODEL_DIR, 'features.json')

WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
SEQUENCE_LENGTH = 24

# --- FUNCTION DEFINITIONS ---

def load_artifacts():
    """Loads the chunk-based model and its tools."""
    print("Loading chunk-based model and preprocessing tools...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        with open(FEATURES_PATH, 'r') as f:
            features = json.load(f)
        print("Artifacts loaded successfully.")
        return model, scaler, features
    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e}. Please run 'train_chunk_model.py' first.")
        return None, None, None

def fetch_api_forecast(city, days):
    """Fetches the direct forecast from WeatherAPI.com for 1-10 days."""
    print(f"Fetching direct {days}-day forecast from API for {city}...")
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&days={days}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()['forecast']['forecastday']
    
    print("\n--- Direct API Forecast Summary ---")
    for day_data in data:
        date = day_data['date']
        day_info = day_data['day']
        print(f"{date}: Avg Temp: {day_info['avgtemp_c']:.1f}°C, "
              f"Precipitation: {day_info['totalprecip_mm']:.2f}mm, "
              f"Avg Humidity: {day_info['avghumidity']:.0f}% "
              f"({day_info['condition']['text']})")

def fetch_historical_data(city):
    """Fetches the last 24 hours of history for any city to seed the model."""
    print(f"Fetching last 24 hours of history for '{city}'...")
    date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    url = f"http://api.weatherapi.com/v1/history.json?key={WEATHER_API_KEY}&q={city}&dt={date}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    hourly_data = data['forecast']['forecastday'][0]['hour']
    
    if len(hourly_data) < SEQUENCE_LENGTH:
        raise ValueError(f"Could not retrieve enough historical data.")
    
    location = data['location']
    for hour in hourly_data:
        hour['latitude'], hour['longitude'] = location.get('lat'), location.get('lon')
    
    return hourly_data

def preprocess_for_prediction(api_data, features_list, scaler):
    """Prepares API data for the generalized model."""
    df = pd.DataFrame(api_data)
    df.rename(columns={'lon': 'longitude', 'lat': 'latitude'}, inplace=True)
    df['condition_text'] = df['condition'].apply(lambda x: x['text'])
    
    processed_df = pd.DataFrame(columns=features_list)
    df_encoded = pd.get_dummies(df, columns=['wind_dir', 'condition_text'])
    
    for col in df_encoded.columns:
        if col in processed_df.columns:
            processed_df[col] = df_encoded[col]

    processed_df = processed_df.reindex(columns=features_list, fill_value=0)
    return scaler.transform(processed_df)

@tf.function
def predict_step(model, input_tensor):
    return model(input_tensor, training=False)

def run_chunk_forecast(initial_sequence, days_to_predict, scaler, model, features):
    """Runs a fast, chunk-based iterative forecast with more robust logic."""
    print(f"Starting {days_to_predict}-day fast, chunk-based forecast...")
    all_predictions_unscaled = []
    current_input_sequence = initial_sequence

    target_indices = [features.index(col) for col in ['temp_c', 'precip_mm', 'humidity']]

    for day in range(days_to_predict):
        input_tensor = tf.convert_to_tensor(np.array([current_input_sequence]), dtype=tf.float32)
        
        predicted_chunk_scaled_targets = predict_step(model, input_tensor)[0].numpy()
        
        if np.any(np.isnan(predicted_chunk_scaled_targets)) or np.any(np.isinf(predicted_chunk_scaled_targets)):
            print(f"  - Day {day + 1}: Model instability detected. Using previous day's data as fallback.")
            predicted_chunk_scaled_targets = current_input_sequence[:, target_indices]

        # --- FIX: Construct the next input sequence more intelligently ---
        # 1. Take the last known hour of the previous day as a baseline for non-target features.
        last_known_hour_all_features = current_input_sequence[-1, :]
        
        # 2. Create a new 24-hour block by repeating this baseline.
        next_input_sequence = np.tile(last_known_hour_all_features, (SEQUENCE_LENGTH, 1))
        
        # 3. Inject the newly predicted target values into this new block.
        for i, idx in enumerate(target_indices):
            next_input_sequence[:, idx] = predicted_chunk_scaled_targets[:, i]
        
        # 4. Unscale the results for this day to store them.
        predicted_chunk_unscaled = scaler.inverse_transform(next_input_sequence)
        
        for hour in range(SEQUENCE_LENGTH):
            all_predictions_unscaled.append({
                'temp_c': predicted_chunk_unscaled[hour, features.index('temp_c')],
                'precip_mm': predicted_chunk_unscaled[hour, features.index('precip_mm')],
                'humidity': predicted_chunk_unscaled[hour, features.index('humidity')]
            })
            
        # 5. The newly constructed sequence becomes the input for the next day.
        current_input_sequence = next_input_sequence
        print(f"  - Day {day + 1} of {days_to_predict} predicted.")

    print("\n--- Chunk-Based Model Forecast Summary ---")
    current_date = datetime.now()
    for i, result in enumerate(all_predictions_unscaled):
        if i % 24 == 0:
            day = i // 24
            forecast_date = (current_date + timedelta(days=day)).strftime('%Y-%m-%d')
            temp_str = f"{result['temp_c']:.1f}" if not np.isnan(result['temp_c']) else "N/A"
            precip_str = f"{result['precip_mm']:.2f}" if not np.isnan(result['precip_mm']) else "N/A"
            hum_str = f"{result['humidity']:.0f}" if not np.isnan(result['humidity']) else "N/A"
            print(f"Day {day+1} ({forecast_date}): Temp: {temp_str}°C, "
                  f"Precipitation: {precip_str}mm, "
                  f"Humidity: {hum_str}%")

if __name__ == "__main__":
    model, scaler, features = load_artifacts()

    if not all([model, scaler, features]):
        print("Exiting.")
    elif not WEATHER_API_KEY:
        print("\nERROR: API key not found. Please ensure WEATHER_API_KEY is in your .env file.")
    else:
        try:
            city_input = input("Enter the city for weather prediction: ")
            days_input = int(input("Enter number of days to forecast: "))
            
            if days_input <= 10:
                fetch_api_forecast(city_input, days_input)
            else:
                seed_data = fetch_historical_data(city_input)
                initial_seq = preprocess_for_prediction(seed_data, features, scaler)
                run_chunk_forecast(initial_seq, days_input, scaler, model, features)

        except ValueError:
            print("\nInvalid input. Please enter a whole number for days.")
        except requests.exceptions.HTTPError as e:
            print(f"\nAPI Error: Could not fetch data. Reason: {e.response.text}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

