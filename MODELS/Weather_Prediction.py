import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# --- INITIALIZATION ---
load_dotenv()
print("Starting API Weather Forecaster...")

# --- CONFIGURATION ---
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')

# --- FUNCTION DEFINITIONS ---

def fetch_api_forecast(city, days):
    """Fetches the direct forecast from WeatherAPI.com for 1-10 days."""
    print(f"\nFetching direct {days}-day forecast from API for {city}...")
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&days={days}"
    
    response = requests.get(url)
    response.raise_for_status() # This will raise an error for bad responses (e.g., 401, 404)
    data = response.json()['forecast']['forecastday']
    
    print("\n--- Direct API Forecast Summary ---")
    for day_data in data:
        date = day_data['date']
        day_info = day_data['day']
        # Updated print statement to make the weather type more explicit
        print(f"{date}: "
              f"Min: {day_info['mintemp_c']:.1f}°C, "
              f"Max: {day_info['maxtemp_c']:.1f}°C, "
              f"Mean: {day_info['avgtemp_c']:.1f}°C, "
              f"Humidity: {day_info['avghumidity']:.0f}%, "
              f"Precip Chance: {day_info['daily_chance_of_rain']}%, "
              f"Type: {day_info['condition']['text']}")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    if not WEATHER_API_KEY:
        print("\nERROR: API key not found. Please ensure WEATHER_API_KEY is in your .env file.")
    else:
        try:
            city_input = input("Enter the city for weather prediction: ")
            days_input = int(input("Enter number of days to forecast (1-10): "))
            
            # Validate the number of days
            if 1 <= days_input <= 10:
                fetch_api_forecast(city_input, days_input)
            else:
                print("\nError: Please enter a number of days between 1 and 10.")

        except ValueError:
            print("\nInvalid input. Please enter a whole number for days.")
        except requests.exceptions.HTTPError as e:
            print(f"\nAPI Error: Could not fetch data. Reason: {e.response.text}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

