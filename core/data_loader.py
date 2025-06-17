# api/core/data_loader.py

import csv
import chardet

def load_gold_data(path: str = "data/FX_IDC_XAUIDRG_1D.csv"):
    with open(path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        print(f"Detected encoding: {encoding}")

    data = []
    with open(path, 'r', encoding=encoding) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    return data

data = load_gold_data()
