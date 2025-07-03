# api/core/config.py

class Settings:
    ALLOWED_ORIGINS = [
        "http://localhost",
        "http://localhost:8080",
        "http://localhost:3000",
        "https://aceh-gold-forecast.vercel.app"
    ]

settings = Settings()
