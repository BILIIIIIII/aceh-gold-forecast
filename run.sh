#!/bin/bash

FILE="data/FX_IDC_XAUIDRG_1D.csv"

# Cek apakah file CSV sudah ada
if [ ! -f "$FILE" ]; then
    echo "[INFO] File $FILE tidak ditemukan. Menjalankan scraper..."
    python scraper/scrape.py
else
    echo "[INFO] File $FILE sudah ada. Melewati proses scraping..."
fi

# Jalankan API
uvicorn api.main:api --reload
