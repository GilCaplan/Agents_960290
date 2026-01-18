import json
import pandas as pd
import os


class WeatherService:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.weather_cache = {}
        self.load_weather_data()

    def load_weather_data(self):
        """Loads weather JSONs and sorts them by date."""
        if not os.path.exists(self.data_dir):
            print(f"WARNING: Directory {self.data_dir} not found.")
            return

        # Load only .json files that are NOT source books
        files = [f for f in os.listdir(self.data_dir) if f.endswith('.json') and not f.startswith("source_")]

        print(f"--- Loading Weather Files from {self.data_dir} ---")
        for f in files:
            # Create a clean key (e.g., 'haifa_technion.json' -> 'haifa_technion')
            city_key = f.replace(".json", "").lower()

            try:
                file_path = os.path.join(self.data_dir, f)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    df = pd.DataFrame(data)

                    # 1. Parse Dates Correctly (Handle Day/Month/Year formats)
                    df['date'] = pd.to_datetime(df['date'], dayfirst=True)

                    # 2. Sort by Date (Oldest -> Newest)
                    df = df.sort_values(by='date')

                    self.weather_cache[city_key] = df
                    latest_date = df.iloc[-1]['date']
                    print(f"Loaded '{city_key}': {len(df)} records. Latest data point: {latest_date}")
            except Exception as e:
                print(f"Error loading {f}: {e}")

    def get_current_weather(self, city: str):
        """
        Finds the weather file for the city and returns the LATEST data point.
        """
        search_key = city.lower()
        df = None

        # 1. Try Exact Match
        if search_key in self.weather_cache:
            df = self.weather_cache[search_key]

        # 2. Try Partial Match (e.g., User: "Haifa" -> File: "haifa_technion")
        if df is None:
            for key in self.weather_cache:
                if search_key in key:
                    df = self.weather_cache[key]
                    break

        # 3. Fallback: If we have data but couldn't match the name, use the first file.
        if df is None and len(self.weather_cache) > 0:
            print(f"DEBUG: Could not match city '{city}'. Using first available weather file.")
            df = list(self.weather_cache.values())[0]

        if df is None:
            return "Weather data not available."

        # 4. Get the LATEST row (The most recent date in the file)
        latest = df.iloc[-1]

        return (
            f"Location: {latest.get('sname', 'Unknown')}\n"
            f"Date/Time: {latest.get('date')}\n"
            f"Temperature: {latest.get('TD')}°C\n"
            f"Humidity: {latest.get('RH')}%\n"
            f"Rainfall: {latest.get('Rain')} mm\n"
            f"Wind Speed: {latest.get('WS')} m/s"
        )