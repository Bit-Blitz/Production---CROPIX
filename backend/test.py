import os
import joblib
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
import warnings

# Suppress all warnings for a cleaner output
warnings.filterwarnings("ignore")

# Suppress TensorFlow C++ and Python logs by setting the log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# Suppress absl warnings
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
# Paths to the saved models based on the provided directory structure
CROP_YIELD_MODEL_PATH = 'Trained_models/CROP_YIELD_MODEL.joblib'
FERTILIZER_MODEL_PATH = 'Trained_models/fertilizer_recommendation_model.joblib'
SOIL_CROP_MODEL_PATH = 'Trained_models/Soil_crop_recom.joblib'
DISEASE_MODEL_PATH = 'Trained_models/CNN/Disease_Detection_model[CNN].h5'
DISEASE_CLASSES_PATH = 'Trained_models/CNN/disease_classes.npy'
MARKET_MODEL_PATH = 'Trained_models/lstm_model.keras'
MARKET_DATA_PATH = 'Datasets/central_india_weekly_crop_prices.csv' # Required for the scaler

# Load environment variables for the Weather API
load_dotenv()
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')


# --- HELPER FUNCTIONS ---

def check_file_exists(path):
    """Checks if a file exists and prints an error message if it doesn't."""
    if not os.path.exists(path):
        print(f"\n[Error] Required file not found: {path}")
        print("Please ensure the file is in the correct directory.")
        return False
    return True

def get_float_input(prompt):
    """Continuously prompts for a float input until a valid number is entered."""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("[Error] Invalid input. Please enter a valid number.")

def get_int_input(prompt):
    """Continuously prompts for an integer input until a valid number is entered."""
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("[Error] Invalid input. Please enter a valid whole number.")

# --- MODEL INFERENCE FUNCTIONS ---

def predict_crop_yield():
    """Handles the crop yield prediction workflow."""
    print("\n--- Crop Yield Prediction (XGBoost) ---")
    if not check_file_exists(CROP_YIELD_MODEL_PATH):
        return

    try:
        model = joblib.load(CROP_YIELD_MODEL_PATH)
        
        # Get user input for all required features
        crop = input("Enter Crop Name (e.g., Rice, Wheat): ")
        crop_year = get_int_input("Enter Crop Year (e.g., 2020): ")
        season = input("Enter Season (e.g., Kharif, Rabi): ")
        area = get_float_input("Enter Area in Hectares: ")
        annual_rainfall = get_float_input("Enter Annual Rainfall (in mm): ")
        fertilizer = get_float_input("Enter total Fertilizer usage (in tonnes): ")
        pesticide = get_float_input("Enter total Pesticide usage (in tonnes): ")

        # Create a DataFrame with all features in the correct order for prediction
        # The order must match the training script's feature order
        input_data = pd.DataFrame([[
            crop,
            crop_year,
            season,
            area,
            annual_rainfall,
            fertilizer,
            pesticide
        ]], columns=['Crop', 'Crop_Year', 'Season', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'])
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        print("\n--- Prediction Result ---")
        print(f"Predicted Crop Yield: {prediction:.2f} tonnes/hectare")
        print(f"Estimated Total Production for the Area: {prediction * area:.2f} tonnes")

    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")


def predict_disease():
    """Handles the plant disease detection workflow."""
    print("\n--- Plant Disease Detection (CNN) ---")
    if not check_file_exists(DISEASE_MODEL_PATH) or not check_file_exists(DISEASE_CLASSES_PATH):
        return

    try:
        model = load_model(DISEASE_MODEL_PATH)
        class_names = np.load(DISEASE_CLASSES_PATH, allow_pickle=True)

        img_path = input("Enter the full path to the plant image file: ")
        if not check_file_exists(img_path):
            return
            
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale

        # Predict
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100

        print("\n--- Prediction Result ---")
        print(f"Predicted Disease: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")


def recommend_fertilizer():
    """Handles the fertilizer recommendation workflow."""
    print("\n--- Fertilizer Recommendation (Random Forest) ---")
    if not check_file_exists(FERTILIZER_MODEL_PATH):
        return

    try:
        model = joblib.load(FERTILIZER_MODEL_PATH)

        # Get user input
        crop = input("Enter Crop Name (e.g., Rice, Wheat): ")
        current_n = get_float_input("Enter current Nitrogen (N) level in soil (kg/hectare): ")
        current_p = get_float_input("Enter current Phosphorus (P) level in soil (kg/hectare): ")
        current_k = get_float_input("Enter current Potassium (K) level in soil (kg/hectare): ")

        # Create DataFrame
        input_data = pd.DataFrame([[crop, current_n, current_p, current_k]], 
                                  columns=['Crop', 'Current_N', 'Current_P', 'Current_K'])
        
        # Predict
        prediction = model.predict(input_data)[0]

        print("\n--- Recommendation Result ---")
        print(f"Recommended Nitrogen (N) to add: {prediction[0]:.2f} kg/hectare")
        print(f"Recommended Phosphorus (P) to add: {prediction[1]:.2f} kg/hectare")
        print(f"Recommended Potassium (K) to add: {prediction[2]:.2f} kg/hectare")

    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")


def forecast_market_prices():
    """Handles a simplified market price forecast for a specific crop."""
    print("\n--- Market Price Forecast (LSTM) ---")
    if not check_file_exists(MARKET_MODEL_PATH) or not check_file_exists(MARKET_DATA_PATH):
        return

    try:
        model = load_model(MARKET_MODEL_PATH)
        df_historical = pd.read_csv(MARKET_DATA_PATH, index_col='Date', parse_dates=True)

        # --- Get User Input ---
        print("Available crops for forecasting:")
        print(", ".join(df_historical.columns))
        while True:
            crop_name = input("Enter the crop name you want to forecast: ")
            if crop_name in df_historical.columns:
                break
            else:
                print(f"[Error] Invalid crop name. Please choose from the list above.")

        weeks_to_forecast = get_int_input("Enter the number of weeks to forecast ahead: ")

        # --- Scale Data and Predict ---
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df_historical)

        n_steps = 8 # This must match the model's training configuration
        n_features = df_historical.shape[1]

        # Prepare the last known data from the historical dataset as the starting point
        last_known_data = scaled_data[-n_steps:]
        current_batch = last_known_data.reshape((1, n_steps, n_features))
        forecast = []

        # Make predictions week by week
        for _ in range(weeks_to_forecast):
            current_pred = model.predict(current_batch, verbose=0)[0]
            forecast.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        # Inverse transform to get actual price values
        forecast_prices = scaler.inverse_transform(forecast)

        # Create a DataFrame for the full forecast
        last_historical_date = df_historical.index[-1]
        future_dates = pd.to_datetime([last_historical_date + pd.Timedelta(weeks=i) for i in range(1, weeks_to_forecast + 1)])
        df_forecast = pd.DataFrame(forecast_prices, index=future_dates, columns=df_historical.columns)

        print(f"\n--- Forecasted Prices for {crop_name} (INR per Quintal) ---")
        # Display only the requested crop's forecast
        print(df_forecast[[crop_name]].round(2))

    except Exception as e:
        print(f"\nAn error occurred during forecasting: {e}")


def recommend_soil_crop():
    """Handles the soil-based crop recommendation workflow."""
    print("\n--- Soil-Based Crop Recommendation (KNN) ---")
    if not check_file_exists(SOIL_CROP_MODEL_PATH):
        return
        
    try:
        model = joblib.load(SOIL_CROP_MODEL_PATH)
        
        # Get user input
        n = get_float_input("Enter Nitrogen (N) level in soil: ")
        p = get_float_input("Enter Phosphorus (P) level in soil: ")
        k = get_float_input("Enter Potassium (K) level in soil: ")
        temp = get_float_input("Enter average temperature (°C): ")
        humidity = get_float_input("Enter average humidity (%): ")
        ph = get_float_input("Enter soil pH value: ")
        rainfall = get_float_input("Enter average rainfall (mm): ")

        # Predict
        input_data = np.array([[n, p, k, temp, humidity, ph, rainfall]])
        prediction = model.predict(input_data)[0]

        print("\n--- Recommendation Result ---")
        print(f"The recommended crop for these conditions is: {prediction.capitalize()}")

    except Exception as e:
        print(f"\nAn error occurred during prediction: {e}")


def get_weather_forecast():
    """Fetches and displays the weather forecast using an API."""
    print("\n--- Weather Forecast (API) ---")
    if not WEATHER_API_KEY:
        print("\n[Error] WEATHER_API_KEY not found in .env file.")
        print("Please create a .env file and add your API key.")
        return

    try:
        city = input("Enter the city for the weather forecast: ")
        days = int(input("Enter number of days to forecast (1-10): "))
        
        if not 1 <= days <= 10:
            print("[Error] Please enter a number of days between 1 and 10.")
            return

        url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&days={days}"
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()['forecast']['forecastday']

        print(f"\n--- Weather Forecast for {city} ---")
        for day_data in data:
            date = day_data['date']
            day_info = day_data['day']
            print(f"{date}: "
                  f"Min: {day_info['mintemp_c']:.1f}°C, "
                  f"Max: {day_info['maxtemp_c']:.1f}°C, "
                  f"Avg: {day_info['avgtemp_c']:.1f}°C, "
                  f"Humidity: {day_info['avghumidity']:.0f}%, "
                  f"Precip Chance: {day_info['daily_chance_of_rain']}%, "
                  f"Weather: {day_info['condition']['text']}")

    except requests.exceptions.HTTPError as e:
        print(f"\nAPI Error: Could not fetch data. Server responded with: {e.response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"\nNetwork Error: Could not connect to the weather service. {e}")
    except (ValueError, KeyError):
        print("\n[Error] Invalid input or unexpected API response.")


# --- MAIN CLI ---

def main():
    """Main function to run the command-line interface."""
    while True:
        print("\n=============================================")
        print("                 CROPIX Interface              ")
        print("==============================================")
        print("1. Predict Crop Yield")
        print("2. Detect Plant Disease from Image")
        print("3. Recommend Fertilizer")
        print("4. Forecast Crop Market Prices")
        print("5. Recommend Crop based on Soil")
        print("6. Get Weather Forecast")
        print("7. Exit")
        print("---------------------------------------------")

        choice = input("Enter your choice (1-7): ")

        if choice == '1':
            predict_crop_yield()
        elif choice == '2':
            predict_disease()
        elif choice == '3':
            recommend_fertilizer()
        elif choice == '4':
            forecast_market_prices()
        elif choice == '5':
            recommend_soil_crop()
        elif choice == '6':
            get_weather_forecast()
        elif choice == '7':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")
        
        input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()

