# api/models/response.py

from typing import Optional, Dict, Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class Meta(BaseModel):
    total: int
    limit: int
    offset: int
    has_next: bool
    has_prev: bool

class ApiResponse(BaseModel, Generic[T]):
    status: str
    data: T
    meta: Optional[Meta] = None
    error: Optional[Dict[str, str]] = None
