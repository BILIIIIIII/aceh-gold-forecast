# api/routers/gold.py

from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime, timedelta

from core.data_loader import data
from utils.response import create_success_response, create_error_response, create_meta

router = APIRouter()

@router.get("/")
def root():
    return create_success_response({
        "message": "Gunakan /gold_prices/ atau /gold_prices/{tanggal}"
    })

@router.get("/gold_prices/")
def read_gold_prices(
    time: Optional[str] = Query(default=None),
    sort: Optional[str] = Query(default=None, regex="^(asc|desc)$"),
    limit: Optional[int] = Query(default=100, ge=0, le=1000),
    offset: int = Query(default=0, ge=0)
):
    try:
        filtered_data = data

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

        if limit == 0 or limit is None:
            paginated_data = filtered_data[offset:]
            actual_limit = len(paginated_data)
        else:
            paginated_data = filtered_data[offset:offset + limit]
            actual_limit = limit

        gold_prices = [
            {
                "time": item["time"],
                "close": item["close"],
                "open": item["open"]
            } for item in paginated_data
        ]

        meta = create_meta(total=total, limit=actual_limit, offset=offset)
        return create_success_response(gold_prices, meta)

    except Exception as e:
        return create_error_response(f"Failed to fetch gold prices: {str(e)}", "FETCH_ERROR")

@router.get("/gold_prices/{date_str}")
def read_gold_price_by_date(date_str: str):
    try:
        search_date = date_str.replace("-", "/")
        for item in data:
            if item['time'] == search_date:
                return create_success_response({
                    "time": item["time"],
                    "close": item["close"],
                    "open": item["open"]
                })

        return create_error_response("Data not found for the specified date", "NOT_FOUND")
    except Exception as e:
        return create_error_response(f"Failed to fetch gold price: {str(e)}", "FETCH_ERROR")
