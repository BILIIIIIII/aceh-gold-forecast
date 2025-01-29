import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ==================================================
# 1) BACA DATA & PREPROCESSING
# ==================================================
data = pd.read_csv('data/harga_emas_2013_2025_volatility.csv')

def clean_numeric_column(column):
    # ganti '.' dengan kosong lalu ',' dengan '.'
    # agar bisa dikonversi ke float
    return pd.to_numeric(column.str.replace('.', '', regex=False)
                                .str.replace(',', '.', regex=False))

data['USD'] = clean_numeric_column(data['USD'])
data['USD Volatility'] = clean_numeric_column(data['USD Volatility'])
data['IDR'] = clean_numeric_column(data['IDR'])
data['IDR Volatility'] = clean_numeric_column(data['IDR Volatility'])

data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

# Imputasi (misal untuk Volatility) jika ada yang kosong
imputer = SimpleImputer(strategy='mean')
data[['USD Volatility', 'IDR Volatility']] = imputer.fit_transform(data[['USD Volatility', 'IDR Volatility']])

# Feature engineering sederhana: Year, Month, Day
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# Feature & Target
features = ['Year', 'Month', 'Day', 'USD Volatility', 'IDR Volatility']
target = 'USD'

X = data[features]
y = data[target]

# Split data (dengan shuffle=False untuk time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

# ==================================================
# 2) DEFINISI FUNGSI EVALUASI (MAE, MSE, R2, MAPE)
# ==================================================
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Menghitung MAPE.
    Note: Pastikan y_true tidak ada yang bernilai 0 (bisa menyebabkan error).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    accuracy = 100 - mape  # Definisi sederhana: 100% - MAPE
    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "MAPE": mape,
        "Accuracy": accuracy
    }

# ==================================================
# 3) INISIALISASI & TRAINING MODEL DENGAN EVAL_SET
#    (AGAR BISA PLOT EPOCH VS ERROR)
# ==================================================
# XGBoost
xgb_model = XGBRegressor(
    random_state=42,
    n_estimators=100,
    eval_metric='rmse'  # penting untuk track error
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)
xgb_evals_result = xgb_model.evals_result()

# LightGBM
lgbm_model = LGBMRegressor(
    random_state=42,
    n_estimators=100
)
# LightGBM eval_metric default nya 'l2' -> MSE
lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='l2'
)
# Untuk LightGBM, kita bisa mengambil 'evals_result_' setelah fitting
lgbm_evals_result = lgbm_model.evals_result_

# ==================================================
# 4) PREDIKSI & EVALUASI
# ==================================================
xgb_predictions = xgb_model.predict(X_test)
lgbm_predictions = lgbm_model.predict(X_test)

xgb_results = evaluate_model(y_test, xgb_predictions, "XGBoost")
lgbm_results = evaluate_model(y_test, lgbm_predictions, "LightGBM")

results_df = pd.DataFrame([xgb_results, lgbm_results])

# ==================================================
# 5) VISUALISASI
# ==================================================
sns.set_style("whitegrid")

# -----------------------------
# 5A) Chart data actual dari 2013
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['USD'], label='Actual USD Price', color='blue')
plt.title('Actual USD Price (2013 - 2025)')
plt.xlabel('Date')
plt.ylabel('USD Price')
plt.legend()
plt.show()

# -----------------------------
# 5B) Visualisasi Training vs Actual (contoh: X_train saja)
#     Kita bisa bandingkan actual vs predicted di TRAINING set
# -----------------------------
# Prediksi training
train_pred_xgb = xgb_model.predict(X_train)
train_pred_lgbm = lgbm_model.predict(X_train)

plt.figure(figsize=(10, 5))
plt.plot(data.loc[X_train.index, 'Date'], y_train, label='Actual (Train)', color='blue')
plt.plot(data.loc[X_train.index, 'Date'], train_pred_xgb, label='XGB Pred (Train)', color='red', alpha=0.7)
plt.plot(data.loc[X_train.index, 'Date'], train_pred_lgbm, label='LGBM Pred (Train)', color='green', alpha=0.7)
plt.title('Train Set: Actual vs Predicted (2 Models)')
plt.xlabel('Date')
plt.ylabel('USD Price')
plt.legend()
plt.show()

# -----------------------------
# 5C) Chart epoch vs error (validation) untuk setiap model
#     Dari evals_result
# -----------------------------
# XGBoost
plt.figure(figsize=(10,5))
epochs = range(len(xgb_evals_result['validation_0']['rmse']))
plt.plot(epochs, xgb_evals_result['validation_0']['rmse'], label='XGB Train RMSE')
plt.plot(epochs, xgb_evals_result['validation_1']['rmse'], label='XGB Test RMSE')
plt.title('XGBoost - Epoch vs RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.show()

# LightGBM
# lgbm_evals_result => dict: {'training': {'l2': [...]}, 'valid_1': {'l2': [...]} }
plt.figure(figsize=(10,5))
epochs_lgbm = range(len(lgbm_evals_result['training']['l2']))
plt.plot(epochs_lgbm, lgbm_evals_result['training']['l2'], label='LGBM Train L2(MSE)')
plt.plot(epochs_lgbm, lgbm_evals_result['valid_1']['l2'], label='LGBM Test L2(MSE)')
plt.title('LightGBM - Epoch vs L2(MSE)')
plt.xlabel('Epoch')
plt.ylabel('L2(MSE)')
plt.legend()
plt.show()

# -----------------------------
# 5D) Perbandingan hasil METRIC (MSE, MAPE, dsb) dari 2 Model (Bar Chart)
# -----------------------------
# MSE
plt.figure(figsize=(6,4))
plt.bar(results_df['Model'], results_df['MSE'], color=['red','green'])
plt.title("Comparison of MSE Among Models")
plt.xlabel("Model")
plt.ylabel("MSE")
plt.show()

# MAPE
plt.figure(figsize=(6,4))
plt.bar(results_df['Model'], results_df['MAPE'], color=['red','green'])
plt.title("Comparison of MAPE Among Models")
plt.xlabel("Model")
plt.ylabel("MAPE (%)")
plt.show()

# ==================================================
# 6) FORECAST TAHUN 2026
# ==================================================
# Kita asumsikan pembuatan data Future (contoh bulanan)
future_dates_2026 = pd.date_range(start='2026-01-01', end='2026-12-31', freq='MS')
future_data_2026 = pd.DataFrame({'Date': future_dates_2026})
future_data_2026['Year'] = future_data_2026['Date'].dt.year
future_data_2026['Month'] = future_data_2026['Date'].dt.month
future_data_2026['Day'] = future_data_2026['Date'].dt.day

# Karena kita tidak punya volatility 2026, kita asumsikan = rata2 training
mean_usd_vol = data['USD Volatility'].mean()
mean_idr_vol = data['IDR Volatility'].mean()
future_data_2026['USD Volatility'] = mean_usd_vol
future_data_2026['IDR Volatility'] = mean_idr_vol

# Fitur untuk prediksi
X_future_2026 = future_data_2026[features]

# Prediksi
future_xgb = xgb_model.predict(X_future_2026)
future_lgbm = lgbm_model.predict(X_future_2026)

# Buat dataframe untuk menampung hasil
future_data_2026['XGB_Pred'] = future_xgb
future_data_2026['LGBM_Pred'] = future_lgbm

# -----------------------------
# 6A) Plot perbandingan 2 model untuk 2026
# -----------------------------
plt.figure(figsize=(10,5))
plt.plot(future_data_2026['Date'], future_data_2026['XGB_Pred'], label='XGBoost')
plt.plot(future_data_2026['Date'], future_data_2026['LGBM_Pred'], label='LightGBM')
plt.title("Forecast USD Price 2026 (Comparison of 2 Models)")
plt.xlabel("Date")
plt.ylabel("Predicted USD Price")
plt.legend()
plt.show()

# ==================================================
# 7) PERBANDINGAN FORECAST 2024 (ACTUAL vs 2 MODEL)
# ==================================================
# 7A) Siapkan data 2024 (aktual)
data_2024 = data[data['Year'] == 2024].copy()
X_2024 = data_2024[features]
y_2024 = data_2024[target]

# Lakukan prediksi
pred_xgb_2024 = xgb_model.predict(X_2024)
pred_lgbm_2024 = lgbm_model.predict(X_2024)

# 7B) Plot Actual vs Predicted (2 model)
plt.figure(figsize=(10, 5))
plt.plot(data_2024['Date'], y_2024, label='Actual 2024', color='blue')
plt.plot(data_2024['Date'], pred_xgb_2024, label='XGBoost 2024 Pred', color='red')
plt.plot(data_2024['Date'], pred_lgbm_2024, label='LightGBM 2024 Pred', color='green')
plt.title("Comparison of Actual vs Forecast (2024)")
plt.xlabel("Date")
plt.ylabel("USD Price")
plt.legend()
plt.show()

# (Opsional) 7C) Evaluasi Khusus 2024
results_2024 = []
results_2024.append(evaluate_model(y_2024, pred_xgb_2024, "XGB_2024"))
results_2024.append(evaluate_model(y_2024, pred_lgbm_2024, "LGBM_2024"))
df_2024 = pd.DataFrame(results_2024)
print("\n=== EVALUATION FOR 2024 DATA ===")
print(df_2024)

# ==================================================
# 8) TAMPILKAN HASIL EVALUASI DI KONSOL
# ==================================================
print("\n=== EVALUATION RESULTS (Test Set) ===")
print(results_df)