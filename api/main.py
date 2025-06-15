import csv
import joblib
import chardet
import numpy as np

from fastapi import FastAPI
from typing import List, Dict
from pydantic import BaseModel
from datetime import datetime, timedelta

app = FastAPI()

# Load model dan scaler
model = joblib.load("gold_price_model.pkl")
scaler_features = joblib.load("scaler_features.pkl")  # Scaler untuk fitur X
scaler_target = joblib.load("scaler_target.pkl")  # Scaler untuk fitur X

# Load data from CSV
data = []

with open('data/FX_IDC_XAUIDRG_1D.csv', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

# Now open the file with the detected encoding
with open('data/FX_IDC_XAUIDRG_1D.csv', 'r', encoding=encoding) as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# Fungsi prediksi
def get_last_sequence(data):
    last_close = float(data[-1]["close"])
    scaled = scaler_features.transform([[last_close]])
    return scaled.flatten()

def predict_future(model, last_sequence, days=365):
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(days):
        pred_scaled = model.predict(current_seq.reshape(1, -1))[0]
        predictions.append(pred_scaled)
        current_seq = np.array([pred_scaled])  # Update sequence
    
    predictions_actual = scaler_target.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()
    
    return predictions_actual

@app.get("/predict_next_year")
def predict_next_year():
    last_sequence = get_last_sequence(data)
    predictions = predict_future(model, last_sequence, days=365)
    
    # Parsing tanggal
    last_date_str = data[-1]["time"]
    last_date = datetime.strptime(last_date_str, "%m/%d/%Y")
    
    dates = [(last_date + timedelta(days=i)).isoformat() for i in range(1, 366)]
    
    return [
        {"date": date, "predicted_price": float(price)}
        for date, price in zip(dates, predictions)
    ]
# Model for data
class GoldPrice(BaseModel):
    time: str
    close: str
    open: str
    
# Ganti semua instance date.fromisoformat() dengan:
# datetime.strptime(item['Date'], "%m/%d/%Y").date()
    
@app.get("/")
def root():
    return {"message": "Gunakan /gold_prices/ atau /gold_prices/{tanggal}"}

@app.get("/gold_prices/", response_model=List[GoldPrice])
def read_gold_prices(start_date: datetime = None, end_date: datetime = None):
    if start_date and end_date:
        filtered_data = [item for item in data if start_date <= datetime.fromisoformat(item['time']) <= end_date]
    elif start_date:
        filtered_data = [item for item in data if datetime.fromisoformat(item['time']) >= start_date]
    elif end_date:
        filtered_data = [item for item in data if datetime.fromisoformat(item['time']) <= end_date]
    else:
        filtered_data = data
    return filtered_data

@app.get("/gold_prices/{date_str}", response_model=Dict[str, str])
def read_gold_price_by_date(date_str: str):
    # Ganti '/' dengan '-' jika diperlukan
    search_date = date_str.replace("-", "/")
    for item in data:
        if item['time'] == search_date:
            return item
    return {"error": "Data not found"}

# ---------------------------------------------------------------------------------------------------------------

@app.get("/gold_prices/langsa", response_model=Dict[str, str])
def read_langsa_gold_price(date_str: str):
    # Ganti '/' dengan '-' jika diperlukan
    search_date = date_str.replace("-", "/")
    for item in data:
        if item['Date'] == search_date:
            return item
    return {"error": "Data not found"}

@app.get("/gold_prices/banda", response_model=Dict[str, str])
def read_banda_gold_price(date_str: str):
    # Ganti '/' dengan '-' jika diperlukan
    search_date = date_str.replace("-", "/")
    for item in data:
        if item['Date'] == search_date:
            return item
    return {"error": "Data not found"}

@app.get("/gold_prices/lhok", response_model=Dict[str, str])
def read_lhok_gold_price(date_str: str):
    # Ganti '/' dengan '-' jika diperlukan
    search_date = date_str.replace("-", "/")
    for item in data:
        if item['Date'] == search_date:
            return item
    return {"error": "Data not found"}