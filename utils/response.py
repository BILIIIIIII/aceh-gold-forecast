# api/utils/response.py

from typing import Optional
from models.response import Meta

def create_success_response(data, meta: Optional[Meta] = None):
    return {
        "status": "success",
        "data": data,
        "meta": meta,
        "error": None
    }

def create_error_response(error_message: str, error_code: str = "GENERAL_ERROR"):
    return {
        "status": "error",
        "data": None,
        "meta": None,
        "error": {error_code: error_message}
    }

def create_meta(total: int, limit: int, offset: int) -> Meta:
    return Meta(
        total=total,
        limit=limit,
        offset=offset,
        has_next=offset + limit < total,
        has_prev=offset > 0
    )
