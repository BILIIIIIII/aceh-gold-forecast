from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import csv
from datetime import datetime
import chardet

app = FastAPI()

# Load data from CSV
data = []



with open('data/harga_emas_2013_2025_volatility.csv', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

# Now open the file with the detected encoding
with open('data/harga_emas_2013_2025_volatility.csv', 'r', encoding=encoding) as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# Model for data
class GoldPrice(BaseModel):
    Date: str
    USD: str
    IDR: str
    
# Ganti semua instance date.fromisoformat() dengan:
# datetime.strptime(item['Date'], "%m/%d/%Y").date()
    
@app.get("/")
def root():
    return {"message": "Gunakan /gold_prices/ atau /gold_prices/{tanggal}"}

@app.get("/gold_prices/", response_model=List[GoldPrice])
def read_gold_prices(start_date: datetime = None, end_date: datetime = None):
    if start_date and end_date:
        filtered_data = [item for item in data if start_date <= datetime.fromisoformat(item['Date']) <= end_date]
    elif start_date:
        filtered_data = [item for item in data if datetime.fromisoformat(item['Date']) >= start_date]
    elif end_date:
        filtered_data = [item for item in data if datetime.fromisoformat(item['Date']) <= end_date]
    else:
        filtered_data = data
    return filtered_data

@app.get("/gold_prices/{date_str}", response_model=Dict[str, str])
def read_gold_price_by_date(date_str: str):
    # Ganti '/' dengan '-' jika diperlukan
    search_date = date_str.replace("-", "/")
    for item in data:
        if item['Date'] == search_date:
            return item
    return {"error": "Data not found"}