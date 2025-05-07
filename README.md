# 🏎️ 2025 F1 Predictor

This project uses **machine learning, FastF1 API data, and historical F1 race results** to predict race outcomes for the 2025 Formula 1 season.

## 🚀 Project Overview
This repository contains a **Gradient Boosting Machine Learning Model** that predicts race results based on past performance, qualifying times, and other structured F1 data. The model leverages:
- FastF1 API for historical race data, such as the results of the previous season (2024).
- 2025 qualifying session results.
- Over the course of the season I will be adding additional data to improve the model.
- Feature engineering techniques to improve predictions.

## 📊 Data Sources
- **FastF1 API**: Fetches lap times, race results, and telemetry data.
- **2025 Qualifying Data**: Used for prediction.
- **Historical F1 Results**: Processed from FastF1 for training the model.

## 🏁 How It Works
1. **Data Collection**: Extract relevant data using the FastF1 API.
2. **Preprocessing & Feature Engineering**: Converts lap times, normalizes driver names, and structures race data.
3. **Model Training**: A **Gradient Boosting Regressor** is trained using 2024 race results.
4. **Prediction**: The model predicts race times for 2025 and ranks drivers accordingly.
5. **Evaluation**: Model performance is measured using Mean Absolute Error **(MAE)**.

### Dependencies
- `fastf1`
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`


## 🔧 Usage
Run the prediction script for a specific race:
```bash
python australian_gp.py
```
Expected output:
```
🏁 2025 Australian Grand Prix Prediction 🏁
Driver: NOR, Predicted Race Time (s): 82.71s
Driver: LEC, Predicted Race Time (s): 83.07s
...
🔍 Model Error (MAE): 3.47 seconds
```

## 📈 Model Performance
The Mean Absolute Error (MAE) is used to evaluate how well the model predicts race times. Lower MAE values indicate more accurate predictions.

## 📌 Future Improvements
- Incorporate **weather conditions** as a feature
- Add **pit stop strategies** into the model
- Explore **deep learning** models for improved accuracy

## 📜 License
This project is licensed under the MIT License.
