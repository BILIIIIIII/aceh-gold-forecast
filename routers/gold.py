# api/routers/gold.py

from fastapi import APIRouter, Query
from typing import Optional, List
from datetime import datetime

# --- UBAH BAGIAN IMPORT INI ---
# Impor variabel data yang sudah jadi dari data_loader
from core.data_loader import historical_data, forecast_data
# -----------------------------

from utils.response import create_success_response, create_error_response, create_meta

router = APIRouter()

@router.get("/")
def root():
    return create_success_response({
        "message": "Gunakan /gold_prices/ (data historis) atau /forecast/ (data prediksi)"
    })

@router.get("/gold_prices/")
def read_gold_prices(
    time: Optional[str] = Query(default=None),
    sort: Optional[str] = Query(default=None, regex="^(asc|desc)$"),
    limit: Optional[int] = Query(default=100, ge=0, le=1000),
    offset: int = Query(default=0, ge=0)
):
    try:
        # --- GUNAKAN `historical_data` ---
        filtered_data = historical_data

        if time:
            filtered_data = [item for item in filtered_data if time.lower() in item['time'].lower()]

        if sort:
            reverse = sort == "desc"
            filtered_data = sorted(
                filtered_data,
                key=lambda x: datetime.strptime(x['time'], "%m/%d/%Y"),
                reverse=reverse
            )

        total = len(filtered_data)
        paginated_data = filtered_data[offset:offset + limit] if limit else filtered_data[offset:]
        
        gold_prices = [{"time": item["time"], "close": item["close"], "open": item["open"]} for item in paginated_data]
        meta = create_meta(total=total, limit=len(paginated_data), offset=offset)
        
        return create_success_response(gold_prices, meta)
    except Exception as e:
        return create_error_response(f"Gagal mengambil data harga emas: {str(e)}", "FETCH_ERROR")


@router.get("/gold_prices/{date_str}")
def read_gold_price_by_date(date_str: str):
    try:
        search_date = date_str.replace("-", "/")
        # --- GUNAKAN `historical_data` ---
        for item in historical_data:
            if item['time'] == search_date:
                return create_success_response({"time": item["time"], "close": item["close"], "open": item["open"]})
        return create_error_response("Data tidak ditemukan untuk tanggal yang ditentukan", "NOT_FOUND")
    except Exception as e:
        return create_error_response(f"Gagal mengambil harga emas: {str(e)}", "FETCH_ERROR")


@router.get("/forecast", summary="Get 30-Day Gold Price Forecast")
def read_forecast():
    if not forecast_data:
        return create_error_response("Data prediksi tidak tersedia.", "NOT_FOUND")
    
    return create_success_response(forecast_data)