from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime
import csv
import chardet

app = FastAPI()

# Load data dari CSV
data = []

# Deteksi encoding file CSV
with open('data/harga_emas_2013_2025_volatility.csv', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']

# Muat data ke dalam list
with open('data/harga_emas_2013_2025_volatility.csv', 'r', encoding=encoding) as f:
    reader = csv.DictReader(f)
    for row in reader:
        data.append(row)

# Model untuk data harga emas
class GoldPrice(BaseModel):
    Date: str
    USD: str
    IDR: str

# Fungsi untuk proses filter dan kalikan IDR
def process_gold_prices(data_list, start_date, end_date, multiplier):
    """Filter data berdasarkan rentang tanggal dan kalikan nilai IDR."""
    filtered = []
    
    # Filter berdasarkan rentang tanggal
    for item in data_list:
        item_datetime = datetime.strptime(item['Date'], "%m/%d/%Y")
        
        if start_date and end_date:
            if start_date <= item_datetime <= end_date:
                filtered.append(item)
        elif start_date:
            if item_datetime >= start_date:
                filtered.append(item)
        elif end_date:
            if item_datetime <= end_date:
                filtered.append(item)
        else:
            filtered = data_list
    
    # Kalikan nilai IDR dengan faktor tertentu
    result = []
    for item in filtered:
        new_item = item.copy()
        try:
            new_item['IDR'] = str(float(new_item['IDR']) * multiplier)
        except ValueError:
            new_item['IDR'] = '0'  # Nilai default jika konversi gagal
        result.append(new_item)
    
    return result

# Endpoint utama
@app.get("/")
def root():
    return {"message": "Gunakan /gold_prices/ atau /gold_prices/{tanggal}"}

@app.get("/gold_prices/global", response_model=List[GoldPrice])
def read_gold_prices_global(start_date: datetime = None, end_date: datetime = None):
    """Tampilkan semua data harga emas (tanpa modifikasi IDR)."""
    return process_gold_prices(data, start_date, end_date, multiplier=1.0)

@app.get("/gold_prices/global/{date_str}", response_model=Dict[str, str])
def read_gold_price_global_by_date(date_str: str):
    """Tampilkan harga emas pada tanggal tertentu (tanpa modifikasi IDR)."""
    search_date = date_str.replace("-", "/")
    for item in data:
        if item['Date'] == search_date:
            return item
    return {"error": "Data tidak ditemukan"}

# --- ENDPOINT UNTUK BANDA ACEH ---
@app.get("/gold_prices/banda_aceh", response_model=List[GoldPrice])
def read_gold_prices_banda_aceh(start_date: datetime = None, end_date: datetime = None):
    """Tampilkan harga emas di Banda Aceh (IDR dikalikan 3.33)."""
    return process_gold_prices(data, start_date, end_date, multiplier=3.33)

@app.get("/gold_prices/banda_aceh/{date_str}", response_model=Dict[str, str])
def read_gold_price_banda_aceh_by_date(date_str: str):
    """Tampilkan harga emas di Banda Aceh pada tanggal tertentu (IDR dikalikan 3.33)."""
    search_date = date_str.replace("-", "/")
    for item in data:
        if item['Date'] == search_date:
            new_item = item.copy()
            try:
                new_item['IDR'] = str(float(item['IDR']) * 3.33)
            except ValueError:
                new_item['IDR'] = '0'
            return new_item
    return {"error": "Data tidak ditemukan"}

# --- ENDPOINT UNTUK LHOKSEUMAWE ---
@app.get("/gold_prices/lhokseumawe", response_model=List[GoldPrice])
def read_gold_prices_lhokseumawe(start_date: datetime = None, end_date: datetime = None):
    """Tampilkan harga emas di Lhokseumawe (IDR dikalikan 3.0)."""
    return process_gold_prices(data, start_date, end_date, multiplier=3.0)

@app.get("/gold_prices/lhokseumawe/{date_str}", response_model=Dict[str, str])
def read_gold_price_lhokseumawe_by_date(date_str: str):
    """Tampilkan harga emas di Lhokseumawe pada tanggal tertentu (IDR dikalikan 3.0)."""
    search_date = date_str.replace("-", "/")
    for item in data:
        if item['Date'] == search_date:
            new_item = item.copy()
            try:
                new_item['IDR'] = str(float(item['IDR']) * 3.0)
            except ValueError:
                new_item['IDR'] = '0'
            return new_item
    return {"error": "Data tidak ditemukan"}

# --- ENDPOINT UNTUK LANGSA ---
@app.get("/gold_prices/langsa", response_model=List[GoldPrice])
def read_gold_prices_langsa(start_date: datetime = None, end_date: datetime = None):
    """Tampilkan harga emas di Langsa (IDR dikalikan 3.0)."""
    return process_gold_prices(data, start_date, end_date, multiplier=3.3)

@app.get("/gold_prices/langsa/{date_str}", response_model=Dict[str, str])
def read_gold_price_langsa_by_date(date_str: str):
    """Tampilkan harga emas di Langsa pada tanggal tertentu (IDR dikalikan 3.0)."""
    search_date = date_str.replace("-", "/")
    for item in data:
        if item['Date'] == search_date:
            new_item = item.copy()
            try:
                new_item['IDR'] = str(float(item['IDR']) * 3.3)
            except ValueError:
                new_item['IDR'] = '0'
            return new_item
    return {"error": "Data tidak ditemukan"}