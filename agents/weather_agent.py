import requests

def get_weather(latitude: float, longitude: float):
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m"
    )
    weather_data = response.json().get("current", {})
    return {
        "temperature": weather_data.get("temperature_2m", "N/A"),
        "response": f"The current temperature is {weather_data.get('temperature_2m', 'N/A')}°C with wind speed {weather_data.get('wind_speed_10m', 'N/A')} m/s.",
    }
