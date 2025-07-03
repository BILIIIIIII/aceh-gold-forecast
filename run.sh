#!/bin/bash
FILE="data/FX_IDC_XAUIDRG_1D.csv"

# Cek apakah file CSV sudah ada
# Ingat: di Railway, file ini akan selalu tidak ada di setiap deployment baru!
if [ ! -f "$FILE" ]; then
    echo "[INFO] File $FILE tidak ditemukan. Menjalankan scraper..."
    # Pastikan Python environment-nya sudah aktif (Nixpacks akan mengaktifkan venv-nya)
    python scraper/scrape.py
else
    echo "[INFO] File $FILE sudah ada. Melewati proses scraping..."
fi

# Jalankan API
# PENTING:
# 1. Hapus --reload (tidak cocok untuk produksi)
# 2. Gunakan $PORT yang disediakan oleh Railway
uvicorn api.main:api --host 0.0.0.0 --port $PORT