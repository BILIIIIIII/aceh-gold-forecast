# api/models/schema.py

from pydantic import BaseModel
from typing import List

class GoldPrice(BaseModel):
    time: str
    close: str
    open: str

class PredictionItem(BaseModel):
    date: str
    predicted_price: float

class PaginatedGoldPrices(BaseModel):
    items: List[GoldPrice]

class PredictionList(BaseModel):
    predictions: List[PredictionItem]
