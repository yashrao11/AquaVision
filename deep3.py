# app.py - Academic Gold Standard Groundwater Prediction System
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import shap
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import geopy.distance
import pydeck as pdk

# ==============================
# 1. DATA ENGINEERING CORE
# ==============================
def handle_missing_data(df):
    """Three-stage missing data handling with academic rigor"""
    # Stage 1: Critical column validation
    critical_cols = ['WSE', 'LATITUDE', 'LONGITUDE']
    df = df.dropna(subset=critical_cols, how='any')
    
    # Stage 2: Type-specific imputation
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.median()))
    
    # Stage 3: Categorical handling
    for col in ['BASIN_NAME', 'WELL_USE']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
    
    return df

def engineer_temporal_features(df):
    """Create publication-quality temporal features"""
    df['Year'] = df['MSMT_DATE'].dt.year
    df['Month'] = df['MSMT_DATE'].dt.month
    df['DayOfYear'] = df['MSMT_DATE'].dt.dayofyear
    df['fourier_sin'] = np.sin(2 * np.pi * df['DayOfYear']/365)
    df['fourier_cos'] = np.cos(2 * np.pi * df['DayOfYear']/365)
    
    # Academic lag features with EWMA
    for lag in [1, 7, 30]:
        df[f'WSE_lag_{lag}'] = df.groupby('Station_Code')['WSE'].transform(
            lambda x: x.shift(lag).ewm(span=lag).mean()
        )
    return df

def calculate_proximity(df):
    """Geospatial feature engineering with Haversine distance"""
    stations = df[['Station_Code', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
    station_coords = stations.set_index('Station_Code')[['LATITUDE', 'LONGITUDE']].to_dict('index')
    
    def get_proximity(row):
        base = (row['LATITUDE'], row['LONGITUDE'])
        return np.mean([
            geopy.distance.distance(base, (station_coords[code]['LATITUDE'], 
                                         station_coords[code]['LONGITUDE'])).km
            for code in station_coords if code != row['Station_Code']
        ])
    
    df['proximity_score'] = df.apply(get_proximity, axis=1)
    return df

# ==============================
# 2. MACHINE LEARNING ENGINE
# ==============================
class AcademicPredictor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.model = xgb.XGBRegressor(
            n_estimators=1500,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            objective='reg:squarederror'
        )
    
    def train(self, X, y):
        """Journal-quality training with TS cross-validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        X_scaled = self.scaler.fit_transform(X)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            self.model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)],
                          early_stopping_rounds=50,
                          verbose=False)
            
            val_pred = self.model.predict(X_val)
            st.write(f"Fold {fold+1} MAE: {mean_absolute_error(y_val, val_pred):.4f}")
    
    def predict(self, X):
        """Peer-reviewed prediction method"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

# ==============================
# 3. ACADEMIC VISUALIZATION
# ==============================
def create_3d_map(df):
    """Nature-quality geospatial visualization"""
    layer = pdk.Layer(
        "HexagonLayer",
        data=df,
        get_position=['LONGITUDE', 'LATITUDE'],
        radius=1000,
        elevation_scale=50,
        extruded=True,
        pickable=True,
        auto_highlight=True
    )
    
    return pdk.Deck(
        map_style='mapbox://styles/mapbox/satellite-v9',
        initial_view_state=pdk.ViewState(
            latitude=df['LATITUDE'].mean(),
            longitude=df['LONGITUDE'].mean(),
            zoom=6,
            pitch=50
        ),
        layers=[layer],
        tooltip={
            "html": "<b>Station:</b> {Station_Code}<br><b>WSE:</b> {WSE:.2f}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    )

def plot_shap_summary(model, X):
    """SHAP analysis for academic explainability"""
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X)
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.tight_layout()
    return fig

# ==============================
# 4. DATA PIPELINE & UI
# ==============================
@st.cache_data(ttl=3600, show_spinner="🔬 Processing data...")
def load_and_process():
    """Reproducible research pipeline"""
    try:
        # Column normalization
        column_mapping = {
            'STATION': 'Station_Code',
            'MSMT_DATE': 'Date',
            'WLM_RPE': 'WSE',
            'ELEV': 'Elevation'
        }
        
        stations = pd.read_csv("gwl-stations.csv").rename(columns=column_mapping)
        daily = pd.read_csv("gwl-daily.csv", parse_dates=['Date']).rename(columns=column_mapping)
        
        # Merge datasets
        merged = pd.merge(daily, stations, on='Station_Code', how='left')
        
        # Academic processing pipeline
        return (
            merged.pipe(handle_missing_data)
                  .pipe(engineer_temporal_features)
                  .pipe(calculate_proximity)
        )
    
    except Exception as e:
        st.error(f"Academic Integrity Alert: {str(e)}")
        sys.exit(1)

def main():
    # Configure academic UI
    st.set_page_config(
        page_title="HydroScholar AI",
        layout="wide",
        page_icon="📚",
        initial_sidebar_state="expanded"
    )
    
    # Load data
    df = load_and_process()
    
    # Academic dashboard
    st.title("📚 HydroScholar: Groundwater Research Platform")
    
    with st.sidebar:
        st.header("Research Parameters")
        station = st.selectbox("Select Monitoring Well", df['Station_Code'].unique())
        forecast_days = st.slider("Prediction Horizon (Days)", 30, 365, 90)
    
    # Main research interface
    tab1, tab2, tab3 = st.tabs(["Geospatial Analysis", "Model Insights", "Research Data"])
    
    with tab1:
        st.pydeck_chart(create_3d_map(df))
        st.plotly_chart(px.line(df, x='Date', y='WSE', color='Station_Code',
                            title="Temporal Dynamics"))
    
    with tab2:
        model = AcademicPredictor()
        features = ['WSE', 'Elevation', 'proximity_score', 'fourier_sin', 'fourier_cos']
        X = df[features]
        y = df['WSE']
        
        if st.button("Train Model"):
            with st.spinner("Conducting Research..."):
                model.train(X, y)
                joblib.dump(model, 'academic_model.joblib')
                
                st.plotly_chart(plot_shap_summary(model, X))
                st.success("Model Training Published!")
    
    with tab3:
        st.dataframe(df, use_container_width=True)
        st.download_button("Export Research Data", df.to_csv(), "hydro_data.csv")

if __name__ == "__main__":
    main()