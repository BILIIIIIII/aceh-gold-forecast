# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import gold
from core.config import settings

api = FastAPI()

# CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
api.include_router(gold.router)