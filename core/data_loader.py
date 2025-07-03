import csv
import json
from pathlib import Path

current_file_path = Path(__file__).resolve()
for parent in current_file_path.parents:
    if (parent / "api").is_dir():
        project_root = parent
        break

if project_root is None:
    raise FileNotFoundError("Tidak dapat menemukan direktori root project. Pastikan folder 'api' ada.")

# Path lengkap ke file data Anda
HISTORICAL_DATA_PATH = project_root / "data" / "FX_IDC_XAUIDRG_1D.csv"
FORECAST_DATA_PATH = project_root / "data" / "forecast_data_all_models.json"


def load_historical_data(path: Path = HISTORICAL_DATA_PATH):
    if not path.is_file():
        print(f"Peringatan: File data historis tidak ditemukan di '{path}'")
        return []

    data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def load_forecast_data(path: Path = FORECAST_DATA_PATH):
    if not path.is_file():
        print(f"Peringatan: File data prediksi tidak ditemukan di '{path}'")
        return []
        
    with open(path, 'r') as f:
        data = json.load(f)
    return data

print(f"Mencoba memuat data dari root: {project_root}")
historical_data = load_historical_data()
forecast_data = load_forecast_data()