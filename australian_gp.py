import os
import fastf1 as f1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable data caching to speed up the data retrieving process
cache_folder = "cache_folder"
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

f1.Cache.enable_cache(cache_folder)

# Get data from the 2024 australian grand prix
session = f1.get_session(2024, "Australia", "R") # "R" represents race
session.load()
print("Cache enabled and race data loaded successfully")

# Extract the data of interest for each lap
laps = session.laps
race_df = laps[["Driver", "LapTime"]].copy()
# It's a good practice to use .copy() when selecting a subset of columns

print(f"\nNumber of rows before dropping NA LapTimes: {len(race_df)}")

# Drop rows where "LapTime" is missing
race_df = race_df.dropna(subset=["LapTime"])
print(f"Number of rows after dropping NA LapTimes: {len(race_df)}")

# Turn LapTime to seconds
race_df["LapTime (s)"] = race_df["LapTime"].dt.total_seconds()

# Input of the 2025 quali data for this specific race
quali_df = pd.DataFrame({
    # It's a must to enter the driver's via their three-letter code, any other format will not work, as the api strictly stores drivers this way
    "Driver": ["NOR", "PIA", "VER", "RUS", "TSU", "ALB", "LEC", "HAM", "GAS", "SAI", "ALO", "STR"],
    "QualifyingTime (s)": [75.096, 75.180, 75.481, 75.546, 75.670, 75.737, 75.755, 75.973, 75.980, 76.062, 76.4, 76.5]
})

# Merge 2024 race data with 2025 qualifying data
merged_df = quali_df.merge(race_df, left_on="Driver", right_on="Driver")

# Predict the LapTime based on the QualifyingTime
X = merged_df[["QualifyingTime (s)"]]
y = merged_df["LapTime (s)"]

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing, check data sources")

# Train gradient boosting model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=39)
model.fit(X_train, y_train)

# Predict using 2025 quali times
predicted_lap_times = model.predict(quali_df[["QualifyingTime (s)"]]) 
quali_df["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race times
quali_df = quali_df.sort_values(by="PredictedRaceTime (s)")

print("\n🏁 2025 Australian Grand Prix Prediction 🏁 \n")
print(quali_df[["Driver", "PredictedRaceTime (s)"]])

# Evaluate Model
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")