import pandas as pd
import numpy as np
from datetime import date, timedelta
import random

random.seed(42)
np.random.seed(42)

FESTIVALS = {
    "2024-01-15": ("Makar Sankranti", "major"),
    "2024-02-13": ("Maha Shivaratri", "major"),
    "2024-03-25": ("Holi", "major"),
    "2024-04-09": ("Ram Navami", "major"),
    "2024-04-14": ("Baisakhi", "minor"),
    "2024-05-23": ("Buddha Purnima", "minor"),
    "2024-07-17": ("Guru Purnima", "major"),
    "2024-08-19": ("Raksha Bandhan", "minor"),
    "2024-08-26": ("Janmashtami", "major"),
    "2024-09-07": ("Ganesh Chaturthi", "major"),
    "2024-10-02": ("Navratri Start", "major"),
    "2024-10-12": ("Dussehra", "major"),
    "2024-10-31": ("Diwali", "major"),
    "2024-11-15": ("Dev Uthani Ekadashi", "minor"),
    "2024-12-30": ("New Year Eve", "minor"),
}

WEATHERS = {
    "winter": ["Cold", "Cold", "Pleasant", "Foggy"],
    "spring": ["Pleasant", "Pleasant", "Warm", "Pleasant"],
    "summer": ["Hot", "Hot", "Very Hot", "Cloudy"],
    "monsoon": ["Rainy", "Rainy", "Cloudy", "Stormy", "Rainy"],
    "autumn": ["Pleasant", "Pleasant", "Cool", "Cloudy"],
}

WEATHER_IMPACT = {
    "Cold": 0.85, "Foggy": 0.70, "Pleasant": 1.00, "Warm": 0.95,
    "Hot": 0.80, "Very Hot": 0.65, "Cloudy": 0.90,
    "Rainy": 0.55, "Stormy": 0.30, "Cool": 1.05,
}

def get_season(month):
    if month in [12, 1, 2]: return "winter"
    if month in [3, 4, 5]: return "spring"
    if month in [6, 7]: return "summer"
    if month in [8, 9]: return "monsoon"
    return "autumn"

rows = []
start = date(2023, 1, 1)
end = date(2024, 12, 31)
d = start

while d <= end:
    ds = str(d)
    day_name = d.strftime("%A")
    month = d.month
    season = get_season(month)
    weather = random.choice(WEATHERS[season])
    weather_mult = WEATHER_IMPACT[weather]

    base = 12000
    if day_name in ["Saturday", "Sunday"]:
        base *= 1.5
    elif day_name == "Monday":
        base *= 1.2

    festival_name = "None"
    festival_type = "none"
    festival_mult = 1.0
    if ds in FESTIVALS:
        festival_name, festival_type = FESTIVALS[ds]
        festival_mult = 3.5 if festival_type == "major" else 2.0

    footfall = int(base * weather_mult * festival_mult * np.random.uniform(0.92, 1.08))
    footfall = max(3000, footfall)

    entry_rate = 420 + random.randint(-30, 60)
    wait_time = round(footfall / (entry_rate * 6))
    if festival_type == "major":
        wait_time = int(wait_time * 1.4)
    wait_time = max(5, wait_time)

    if footfall > 50000: level = "High"
    elif footfall > 25000: level = "Medium"
    else: level = "Low"

    rows.append({
        "date": ds,
        "day": day_name,
        "month": month,
        "season": season,
        "festival_name": festival_name,
        "festival_type": festival_type,
        "weather": weather,
        "is_weekend": 1 if day_name in ["Saturday", "Sunday"] else 0,
        "footfall": footfall,
        "wait_time_min": wait_time,
        "crowd_level": level,
        "entry_rate": entry_rate,
    })
    d += timedelta(days=1)

df = pd.DataFrame(rows)
df.to_csv("pilgrimage_data.csv", index=False)
print(f"Generated {len(df)} records")
print(df.head(10).to_string())
print("\nCrowd level distribution:")
print(df["crowd_level"].value_counts())
