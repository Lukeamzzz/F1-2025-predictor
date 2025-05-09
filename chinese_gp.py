import os
import fastf1 as f1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

cache_folder = "cache_folder"
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

f1.Cache.enable_cache(cache_folder)

# Get data from the 2024 chinese grand prix
session = f1.get_session(2024, "China", "R") # "R" represents race
session.load()
print("Cache enabled and race data loaded successfully")

# Extract the data of interest
race_df = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
# It's a good practice to use .copy() when selecting a subset of columns

# Remove the rows that contain null values
race_df = race_df.dropna()

# Covert times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    race_df[f"{col} (s)"] = race_df[col].dt.total_seconds()

# Get the average sector times in seconds per driver
# When grouping the data by driver, the "Driver" column becomes the index of the resulting df
# ".reset_index()" restores "Driver" as a regular column
sector_times = race_df.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()

# Input of the 2025 quali data for this specific race
quali_df = pd.DataFrame({
    "Driver": ["PIA", "RUS", "NOR", "VER", "HAM", "LEC", "HAD", "ANT", "TSU", "ALB", 
               "OCO", "HUL", "ALO", "STR", "SAI", "GAS", "BEA", "DOO", "BOR", "LAW"],

    "QualifyingTime (s)": [90.641, 90.723, 90.793, 90.817, 90.927, 91.021, 91.079, 91.103, 91.638, 91.706,
                           91.625, 91.632, 91.688, 91.773, 91.840, 91.992, 92.018, 92.092, 92.141, 92.174]
})

# Merge qualifying data with sector times
merged_df = quali_df.merge(sector_times, left_on="Driver", right_on="Driver", how="left")

# Predict the average laptime by driver based on the quali time and the sectors
# We replace NaN values with 0 instead of dropping rows to handle partial data, as a driver might have missing sector times 
# due to a crash, red flag, or incomplete runs
X = merged_df[["QualifyingTime (s)", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].fillna(0)
y = race_df.groupby("Driver")["LapTime (s)"].mean().reset_index()["LapTime (s)"]

# Train gradient boosting model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=38)
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=38)
model.fit(X_train, y_train)

# Predict race times based on X (quali time and sectors time)
predicted_lap_times = model.predict(X)
quali_df["PredictedRaceTime (s)"] = predicted_lap_times

# Rank drivers by predicted race times
quali_df = quali_df.sort_values(by="PredictedRaceTime (s)")

print("\n 🏁 2025 Chinese Grand Prix Prediction 🏁 \n")
print(quali_df[["Driver", "PredictedRaceTime (s)"]])

# Evaluate model
# The model predicts lap times based on the X_test features (quali time and sectors time)
# We compare the prediction (y_pred) with the real lap times (y_test), MEA masures how far off our predictions are from the real laptimes
y_pred = model.predict(X_test)
print(f"\n🔍 Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")