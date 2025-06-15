# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# Configure page settings
st.set_page_config(page_title="Gold Price Forecasting", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .model-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
    }
    .metric-box {
        background-color: #e9ecef;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        margin: 0.5rem;
        min-width: 180px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and preprocess the data"""
    df = pd.read_csv("data/FX_IDC_XAUIDRG_1D.csv", parse_dates=["time"])
    df = df[df["Volume"] > 0]
    df.rename(columns={
        "time": "Date", "open": "Open", 
        "high": "High", "low": "Low", 
        "close": "Close"
    }, inplace=True)
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    df = df[["Close"]]
    
    data = df.copy()
    data["Target"] = data["Close"].shift(-1)
    data.dropna(inplace=True)
    
    base_date = data.index[int(len(data)*0.8)]
    train = data[data.index < base_date]
    valid = data[data.index >= base_date]
    
    return {
        'df': df,
        'data': data,
        'train': train,
        'valid': valid,
        'base_date': base_date
    }

@st.cache_resource
def train_models(_data):
    """Train all models and return results"""
    X_train = _data['train'][["Close"]].values
    y_train = _data['train']["Target"].values
    X_valid = _data['valid'][["Close"]].values
    y_valid = _data['valid']["Target"].values
    
    models_cfg = {
        'XGBoost': (
            XGBRegressor(random_state=42, n_jobs=-1),
            {'model__booster': ['gblinear','gbtree'], 
             'model__n_estimators': [50,100,200], 
             'model__learning_rate': [0.01,0.1],
             'model__objective': ['reg:squarederror']}
        ),
        'RandomForest': (
            RandomForestRegressor(random_state=42),
            {'model__n_estimators': [100,200], 
             'model__max_depth': [5,10,20], 
             'model__min_samples_split': [2,5]}
        ),
        'Linear': (
            LinearRegression(),
            {'model__fit_intercept': [True, False]}
        )
    }
    
    results = {}
    
    for name, (model, params) in models_cfg.items():
        pipeline = Pipeline([("scaler", MinMaxScaler()), ("model", model)])
        
        if len(X_train) < 3:
            pipeline.fit(X_train, y_train)
        else:
            splits = min(5, len(X_train)-1)
            tscv = TimeSeriesSplit(n_splits=splits)
            
            searcher = RandomizedSearchCV(
                pipeline, params, cv=tscv, n_jobs=-1, verbose=0,
                n_iter=20, random_state=42
            )
            searcher.fit(X_train, y_train)
            pipeline = searcher.best_estimator_
        
        # Predictions
        train_preds = pipeline.predict(X_train)
        valid_preds = pipeline.predict(X_valid)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, train_preds)
        valid_mae = mean_absolute_error(y_valid, valid_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        valid_rmse = np.sqrt(mean_squared_error(y_valid, valid_preds))
        train_r2 = pipeline.score(X_train, y_train)
        valid_r2 = pipeline.score(X_valid, y_valid)
        
        # Generate plots
        fig_train, ax_train = plt.subplots(figsize=(10, 4))
        ax_train.plot(_data['train'].index[-100:], y_train[-100:], label='Actual')
        ax_train.plot(_data['train'].index[-100:], train_preds[-100:], label='Predicted')
        ax_train.set_title(f'{name} - Training Data (Last 100 Days)')
        ax_train.set_xlabel('Date')
        ax_train.set_ylabel('Price')
        ax_train.legend()
        plt.tight_layout()
        train_plot = fig_train
        
        fig_valid, ax_valid = plt.subplots(figsize=(10, 4))
        ax_valid.plot(_data['valid'].index, y_valid, label='Actual')
        ax_valid.plot(_data['valid'].index, valid_preds, label='Predicted')
        ax_valid.set_title(f'{name} - Validation Data')
        ax_valid.set_xlabel('Date')
        ax_valid.set_ylabel('Price')
        ax_valid.legend()
        plt.tight_layout()
        valid_plot = fig_valid
        
        results[name] = {
            'model': pipeline,
            'metrics': {
                'train_mae': train_mae,
                'valid_mae': valid_mae,
                'train_rmse': train_rmse,
                'valid_rmse': valid_rmse,
                'train_r2': train_r2,
                'valid_r2': valid_r2
            },
            'plots': {
                'train': train_plot,
                'valid': valid_plot
            }
        }
    
    return results

def display_model_tab(name, model_data):
    """Display content for each model tab"""
    metrics = model_data['metrics']
    
    st.markdown(f"### {name} Model Performance")
    
    # Metrics display
    st.markdown("#### Evaluation Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.write("Training MAE:", f"{metrics['train_mae']:.4f}")
        st.write("Validation MAE:", f"{metrics['valid_mae']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.write("Training RMSE:", f"{metrics['train_rmse']:.4f}")
        st.write("Validation RMSE:", f"{metrics['valid_rmse']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.write("Training R²:", f"{metrics['train_r2']:.4f}")
        st.write("Validation R²:", f"{metrics['valid_r2']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Plot displays
    st.markdown("#### Visualizations")
    st.markdown("##### Training Data Comparison")
    st.pyplot(model_data['plots']['train'])
    
    st.markdown("##### Validation Data Comparison")
    st.pyplot(model_data['plots']['valid'])

def main():
    st.title("📈 Gold Price Forecasting Dashboard")
    
    # Load data
    data = load_and_process_data()
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Model Analysis"])
    
    if page == "Home":
        st.header("🏠 Home")
        
        st.markdown("## 📊 Actual Gold Price Data")
        st.line_chart(data['df']['Close'])
        
        st.markdown("### Data Overview")
        st.write(data['df'].describe())
        
        st.markdown(f"### 📅 Data Period")
        st.write(f"Start: {data['df'].index.min().date()}")
        st.write(f"End: {data['df'].index.max().date()}")
        
        st.markdown(f"### 📁 Training/Validation Split")
        st.write(f"Training End Date: {data['base_date'].date()}")
        st.write(f"Training Samples: {len(data['train'])}")
        st.write(f"Validation Samples: {len(data['valid'])}")
    
    elif page == "Model Analysis":
        st.header("🧮 Model Analysis")
        
        # Train models
        model_results = train_models(data)
        
        st.markdown("## 📈 Model Comparisons")
        tab1, tab2, tab3 = st.tabs(["XGBoost", "RandomForest", "Linear"])
        
        with tab1:
            display_model_tab("XGBoost", model_results['XGBoost'])
        
        with tab2:
            display_model_tab("RandomForest", model_results['RandomForest'])
        
        with tab3:
            display_model_tab("Linear", model_results['Linear'])

if __name__ == "__main__":
    main()