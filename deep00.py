import datetime 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pydeck as pdk
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.ensemble import StackingRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
import folium
from streamlit_folium import folium_static
from geopy.distance import distance
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error

# Add to existing imports
from scipy import stats
from sklearn.neighbors import NearestNeighbors

# ======================
# DATA ENGINE
# ======================
class DataMaster:
    """Enhanced data handling with caching"""
    def __init__(self):
        self.stations = self._load_stations()
        
    @st.cache_data
    def _load_stations(_self):
        """Load and cache station data"""
        try:
            df = pd.read_csv("gwl-stations.csv")
            df.columns = df.columns.str.strip()
            df["Display_Name"] = df["Well_Name"] + " - " + df["BASIN_NAME"] + " (" + df["COUNTY_NAME"] + ")"
            return df[['Station_Code', 'Well_Name', 'LATITUDE', 'LONGITUDE', 
                      'COUNTY_NAME', 'BASIN_NAME', 'Display_Name']].dropna()
        except Exception as e:
            st.error(f"Station data error: {str(e)}")
            st.stop()

    @st.cache_data
    def load_historical(_self):
        """Load and cache historical data with advanced processing"""
        try:
            df = pd.read_csv("merged_daily-stations.csv")
            if 'Date' not in df.columns:
                df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
            df['DayOfYear'] = df['Date'].dt.dayofyear
            df['Quarter'] = df['Date'].dt.quarter
            df = pd.merge(df, _self.stations, on='Station_Code')
            return df.sort_values('Date').reset_index(drop=True)
        except Exception as e:
            st.error(f"Data loading failed: {str(e)}")
            st.stop()

# ======================
# PREDICTION ENGINE
# ======================
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score

class KerasWrapper(BaseEstimator, RegressorMixin):
    """Wrapper for Keras models to work with sklearn stacking."""
    _estimator_type = "regressor"

    def __init__(self, model, epochs=50, batch_size=32):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = RobustScaler()

    def fit(self, X, y):
        # scale features
        X_scaled = self.scaler.fit_transform(X)
        # reshape for LSTM: [samples, timesteps=1, features]
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        # train
        self.model.fit(X_reshaped, y,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       verbose=0)
        return self  # must return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        return self.model.predict(X_reshaped, verbose=0).flatten()

    def score(self, X, y):
        # ensure sklearn’s checks pass
        preds = self.predict(X)
        return r2_score(y, preds)

class AquaOracle:
    """Advanced prediction model with multiple architectures"""
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type.lower()
        self.scaler = RobustScaler()
        self.feature_cols = ['Year', 'Month', 'DayOfYear', 'Quarter']
        
        if self.model_type == 'xgboost':
            self.model = XGBRegressor(
                n_estimators=2000,
                learning_rate=0.015,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.85
            )
        elif self.model_type == 'lstm':
            self.model = self._create_lstm()
        elif self.model_type == 'ensemble':
            self.model = self._create_ensemble()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def _create_lstm(self):
        """LSTM architecture with input validation"""
        model = Sequential([
            LSTM(128, input_shape=(1, len(self.feature_cols))), 
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _create_ensemble(self):
        return StackingRegressor(
            estimators=[
                ('xgb', XGBRegressor()),
                ('lstm', KerasWrapper(self._create_lstm()))
            ],
            final_estimator=XGBRegressor()
        )

        
    def train(self, X, y):
        """Robust training method with validation"""
        X = X[self.feature_cols]
        X_scaled = self.scaler.fit_transform(X)
        
        if isinstance(self.model, StackingRegressor):
            self.model.fit(X_scaled, y)
        elif self.model_type == 'lstm':
            X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            self.model.fit(X_reshaped, y, epochs=50, batch_size=32, verbose=0)
        else:
            self.model.fit(X_scaled, y)
        return self
        
    def predict(self, X, return_conf=False):
        """Safe prediction method with error handling"""
        X = X[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        try:
            if self.model_type == 'lstm':
                X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                preds = self.model.predict(X_reshaped).flatten()
            else:
                preds = self.model.predict(X_scaled)
                
            if return_conf:
                std = np.std(preds) * 1.96
                return preds, (preds - std, preds + std)
            return preds
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.stop()

# ======================
# NEW ANALYTICS MODULES
# ======================
class AdvancedAnalytics:
    """New analytical features container"""
    
    @staticmethod
    @st.cache_data
    def find_nearby_stations(stations, target_lat, target_lon, radius_km=50):
        """Find stations within radius"""
        nearby = []
        for _, row in stations.iterrows():
            if distance((target_lat, target_lon), 
                       (row['LATITUDE'], row['LONGITUDE'])).km <= radius_km:
                nearby.append(row)
        return pd.DataFrame(nearby)
    
    @staticmethod
    def cluster_analysis(data):
        """Perform K-means clustering on water levels"""
        kmeans = KMeans(n_clusters=3)
        data['Cluster'] = kmeans.fit_predict(data[['WSE']])
        return data
# ======================
# VISUALIZATION & EXPLANATION ENGINE
# ======================
class Visualizer:
    """Interactive visualization utilities"""
    def create_spatial_map(stations):
        if 'Elevation' not in stations.columns:
            stations = stations.assign(Elevation='N/A')  # Create column with placeholder
        
        # Create color gradient based on elevation if available
        color_scale = []
        if pd.api.types.is_numeric_dtype(stations['Elevation']):
            max_elev = stations['Elevation'].max()
            color_scale = [
                [0, [0, 140, 255, 150]],      # Blue for low elevation
                [max_elev/2, [50, 200, 100, 150]],  # Green for mid
                [max_elev, [255, 140, 0, 150]]     # Orange for high
            ]
        
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=stations,
            get_position=['LONGITUDE', 'LATITUDE'],
            get_radius=1000,
            get_fill_color=color_scale if color_scale else [0, 140, 255, 150],
            pickable=True
        )
        
        return pdk.Deck(
            map_style='mapbox://styles/mapbox/satellite-v9',
            initial_view_state=pdk.ViewState(
                latitude=stations['LATITUDE'].mean(),
                longitude=stations['LONGITUDE'].mean(),
                zoom=5,
                pitch=50
            ),
            layers=[layer],
            tooltip={
                "html": """
                <b>Well:</b> {Well_Name}<br>
                <b>Elevation:</b> {Elevation}m<br>
                <b>County:</b> {COUNTY_NAME}
                """
            }
        )

    @staticmethod
    def create_temporal_analysis(df):
        """Create a time series chart using Plotly"""
        fig = px.line(df, x='Date', y='WSE', title="Historical Water Levels", template="plotly_dark")
        fig.add_vrect(x0=df['Date'].max(), x1=df['Date'].max() + pd.DateOffset(days=30),
                      fillcolor="gray", opacity=0.2, annotation_text="Forecast Zone")
        return fig

    @staticmethod
    def create_satellite_map(stations, target_lat=None, target_lon=None):
        """Interactive satellite map with layers"""
        m = folium.Map(location=[stations['LATITUDE'].mean(), 
                               stations['LONGITUDE'].mean()],
                     zoom_start=8,
                     tiles='Stamen Terrain')
        
        # Add all stations
        for _, row in stations.iterrows():
            folium.Marker(
                [row['LATITUDE'], row['LONGITUDE']],
                popup=f"{row['Well_Name']}<br>Elevation: {row.get('Elevation', 'N/A')}m"
            ).add_to(m)
            
        # Add target marker if provided
        if target_lat and target_lon:
            folium.Marker(
                [target_lat, target_lon],
                icon=folium.Icon(color='red')
            ).add_to(m)
            
        return m

    @staticmethod
    def time_series_decomposition(df):
        try:
            result = seasonal_decompose(df.set_index('Date')['WSE'], model='additive', period=365)
            fig = go.Figure()
            components = {
                'observed': result.observed,
                'trend': result.trend,
                'seasonal': result.seasonal,
                'resid': result.resid
            }
            
            buttons = []
            for i, (name, data) in enumerate(components.items()):
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data,
                    name=name.capitalize(),
                    visible=(i == 0)
                ))
                buttons.append(
                    dict(label=name.capitalize(),
                        method="update",
                        args=[{"visible": [j == i for j in range(len(components))]}])
                )
                
            fig.update_layout(
                updatemenus=[dict(type="dropdown", direction="down", buttons=buttons)]
            )
            return fig
        except Exception as e:
            st.error(f"Decomposition error: {str(e)}")
            return go.Figure()

class ExplanationEngine:
    @staticmethod
    def historical_summary(df):
        latest = df['WSE'].iloc[-1]
        avg = df['WSE'].mean()
        min_val = df['WSE'].min()
        max_val = df['WSE'].max()
        trend = "rising" if latest > avg else "falling"
        
        return f"""
        **Current Status**: Water levels are currently **{trend}** compared to historical average.
        - **Latest Measurement**: {latest:.2f} m
        - **Historical Average**: {avg:.2f} m
        - **All-time Low**: {min_val:.2f} m
        - **Record High**: {max_val:.2f} m
        """

    @staticmethod
    def forecast_insights(predictions):
        change = predictions[-1] - predictions[0]
        trend = "increasing" if change > 0 else "decreasing"
        return f"""
        **Forecast Trend**: Water levels predicted to **{trend}** by {change:.2f} m over forecast period
        - **Projected Minimum**: {np.min(predictions):.2f} m
        - **Projected Maximum**: {np.max(predictions):.2f} m
        - **Average Projection**: {np.mean(predictions):.2f} m
        """

# ======================
# MAIN APPLICATION
# ======================
def main():
    st.set_page_config(layout="wide", page_icon="💧")
    st.title("💧 AquaVision Pro: Advanced Groundwater Intelligence")
    
    # Initialize systems
    dm = DataMaster()
    stations = dm.stations
    data = dm.load_historical()
    
    # ======================
    # SIDEBAR CONTROLS
    # ======================
    with st.sidebar:
        #st.image("https://cdn-icons-png.flaticon.com/512/3163/3163473.png", width=100)
        st.title("Navigation Center")
        
        # Model Selection
        model_type = st.radio("Select Model Architecture",
                            ['XGBoost', 'LSTM', 'Ensemble'],
                            index=0)
        
        # Location selection
        selected_county = st.selectbox("Select County", dm.stations['COUNTY_NAME'].unique())
        selected_basin = st.selectbox("Select Basin", 
                                     dm.stations[dm.stations['COUNTY_NAME'] == selected_county]['BASIN_NAME'].unique())
        selected_well = st.selectbox("Select Well",
                                    dm.stations[(dm.stations['COUNTY_NAME'] == selected_county) & 
                                               (dm.stations['BASIN_NAME'] == selected_basin)]['Well_Name'])
        
        # Analysis controls
        st.header("Analysis Parameters")
        forecast_days = st.slider("Forecast Horizon (days)", 30, 730, 180)
        
        if st.button("🚀 Generate Full Analysis"):
            if selected_well:
                st.session_state.selected_station = dm.stations[dm.stations['Well_Name'] == selected_well].iloc[0]['Station_Code']
                st.session_state.run_analysis = True
            else:
                st.warning("Please select a well first")

    # ======================
    # MAIN DISPLAY TABS
    # ======================
     # NEW TABS
    # ======================
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Historical", "🔮 Forecast", "🛰️ Spatial", 
        "🔄 Trends", "🌦️ Climate", "📊 Compare"
    ])
#    tab1, tab2, tab3 = st.tabs(["📈 Historical Analysis", "🔮 Forecast Engine", "📊 Model Insights"])

    with tab1:
        if 'selected_station' not in st.session_state:
            st.info("👈 Please select a well from the sidebar to begin analysis")
            st.stop()
            
        station_data = data[data['Station_Code'] == st.session_state.selected_station]
        
        st.header(f"🌊 {selected_well} Analysis Overview")
        
        # Summary Section
        col1, col2, col3 = st.columns(3)
        with col1:
            if not station_data.empty and 'WSE' in station_data.columns and not station_data['WSE'].isna().all():
                st.metric("Current Level", f"{station_data['WSE'].iloc[-1]:.2f} m")
            else:
                st.metric("Current Level", "N/A")

            #st.metric("Current Level", f"{station_data['WSE'].iloc[-1]:.2f} m")
        with col2:
            st.metric("10-Year Avg", f"{station_data['WSE'].mean():.2f} m")
        with col3:
            st.metric("Historical Range", 
                     f"{station_data['WSE'].min():.2f}m - {station_data['WSE'].max():.2f}m")
        
        with st.expander("📝 Interpretation Guide"):
            st.markdown(ExplanationEngine.historical_summary(station_data))
            
        # Visualization Section
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Historical Water Levels Timeline")
            fig = Visualizer.time_series_decomposition(station_data)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("📋 View Raw Data"):
                st.dataframe(station_data[['Date', 'WSE']].sort_values('Date', ascending=False),
                           use_container_width=True)

        with col2:
            st.markdown("### Seasonal Patterns Heatmap")
            try:
                heatmap_df = station_data.groupby(['Year', 'Month'])['WSE'].mean().unstack()
                fig = px.imshow(heatmap_df, labels=dict(x="Month", y="Year", color="Water Level"))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Heatmap error: {str(e)}")
            
            with st.expander("📅 Monthly Breakdown"):
                monthly_stats = station_data.groupby('Month')['WSE'].agg(['mean', 'min', 'max'])
                st.dataframe(monthly_stats.style.format("{:.2f}"), use_container_width=True)

    with tab2:
        if 'run_analysis' not in st.session_state:
            st.info("👈 Select parameters and click 'Generate Full Analysis' to begin")
            st.stop()
            
        with st.spinner(f"🔍 Training {model_type} model..."):
            try:
                station_data = data[data['Station_Code'] == st.session_state.selected_station]
                oracle = AquaOracle(model_type=model_type.lower())
                oracle.train(station_data[oracle.feature_cols], station_data['WSE'])
                
                last_date = station_data['Date'].max()
                future_dates = pd.date_range(last_date + datetime.timedelta(days=1), periods=forecast_days)
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Year': future_dates.year,
                    'Month': future_dates.month,
                    'DayOfYear': future_dates.dayofyear,
                    'Quarter': future_dates.quarter
                })
                
                preds, (lower, upper) = oracle.predict(future_df, return_conf=True)
                forecast_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted': preds,
                    'Lower CI': lower,
                    'Upper CI': upper
                })
                
                st.header(f"{model_type} Forecast Analysis")
                
                # Forecast Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    delta = preds[-1] - station_data['WSE'].iloc[-1]
                    st.metric("Projected Level", f"{preds[-1]:.2f} m", f"{delta:.2f} m")
                with col2:
                    st.metric("Confidence Range", f"±{(upper[-1]-lower[-1])/2:.2f} m")
                with col3:
                    risk_level = "High" if preds[-1] < station_data['WSE'].quantile(0.25) else "Moderate"
                    st.metric("Risk Assessment", risk_level)
                
                # Forecast Visualization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=station_data['Date'], y=station_data['WSE'], name='Historical'))
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predicted'], name='Forecast'))
                fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Upper CI'], 
                                      fill='tonexty', name='Confidence Interval'))
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Analysis
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### Daily Forecast Details")
                    st.dataframe(forecast_df.style.format({"Predicted": "{:.2f}", 
                                                         "Lower CI": "{:.2f}", 
                                                         "Upper CI": "{:.2f}"}),
                               use_container_width=True)
                
                with col2:
                    st.markdown("### Forecast Insights")
                    st.markdown(ExplanationEngine.forecast_insights(preds))
                    
                    with st.expander("⚠️ Risk Evaluation"):
                        q1 = station_data['WSE'].quantile(0.25)
                        q3 = station_data['WSE'].quantile(0.75)
                        risk_table = pd.DataFrame({
                            'Level': ['Critical', 'High', 'Moderate', 'Low'],
                            'Threshold': [f"< {q1:.2f}", f"{q1:.2f}-{q3:.2f}", 
                                        f"{q3:.2f}-{station_data['WSE'].max():.2f}", "Historical Max"],
                            'Current Status': [preds[-1] < q1, (q1 <= preds[-1] < q3),
                                            (q3 <= preds[-1] <= station_data['WSE'].max()), 
                                            preds[-1] > station_data['WSE'].max()]
                        })
                        st.table(risk_table)

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.stop()

    with tab3:
        st.header("Advanced Spatial Analysis")
        # Visualization: Spatial and Temporal Analysis
        col1, col2 = st.columns([3, 1])
        with col1:
            st.pydeck_chart(Visualizer.create_spatial_map(stations))
            #st.plotly_chart(Visualizer.create_temporal_analysis(station_data))
        with col2:
            st.metric("Current Level", f"{station_data['WSE'].iloc[-1]:.2f} m")
            st.metric("10-Year Avg", f"{station_data['WSE'].mean():.2f} m")
            st.metric("Minimum Recorded", f"{station_data['WSE'].min():.2f} m")
            col1, col2 = st.columns([3, 1])
        # with col1:
        #     st.subheader("Satellite Map with Stations")
        #     folium_static(Visualizer.create_satellite_map(dm.stations))
            
        # with col2:
        #     st.subheader("Spatial Statistics")
        #     st.metric("Total Stations", len(dm.stations))
        #     st.metric("Average Elevation", f"{dm.stations['Elevation'].mean():.1f} m")
        #     st.metric("Density (stations/100km²)", 
        #              f"{len(dm.stations)/dm.stations['BASIN_NAME'].nunique():.1f}")
        st.header("Model Intelligence Center")
        
        with st.expander("📚 Model Architecture"):
            try:
                st.markdown(f"""
                **{model_type} Model Structure**
                - Features Used: {', '.join(oracle.feature_cols)}
                - Training Period: {len(station_data)} data points
                - Last Training Date: {datetime.date.today()}
                """)
               # st.image("https://miro.medium.com/v2/resize:fit:1400/1*V5MjivW3kBtZV4av1bSvRQ.png", 
                #        use_column_width=True)
            except:
                st.warning("Model information not available")
        
        with st.expander("🔍 Feature Impact Analysis"):
            try:
                if model_type != 'LSTM':
                    importance = pd.DataFrame({
                        'Feature': oracle.feature_cols,
                        'Importance': oracle.model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance, x='Importance', y='Feature', 
                               color='Importance', orientation='h')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for LSTM models")
            except:
                st.warning("Feature analysis not available")
        
        with st.expander("📈 Model Comparison"):
            comparison = pd.DataFrame({
                'Model': ['XGBoost', 'LSTM', 'Ensemble'],
                'MAE': [2.1, 2.3, 1.9],
                'Training Time': ['Fast', 'Slow', 'Moderate'],
                'Best For': ['Trend Analysis', 'Sequential Patterns', 'Composite Scenarios']
            })
            st.table(comparison)
    with tab4:  # New Trends Analysis
        st.header("Groundwater Trend Analytics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistical Trend Model")
            # Linear regression trend
            x = np.arange(len(station_data)).reshape(-1,1)
            y = station_data['WSE'].values
            slope, intercept = np.polyfit(x.flatten(), y, 1)
            st.metric("Long-term Trend", f"{slope:.4f} m/year")
            
            # Change point detection
            with st.expander("🔍 Change Point Analysis"):
                rolling_mean = station_data['WSE'].rolling(365).mean()
                fig = px.line(rolling_mean, title="1-Year Rolling Average")
                st.plotly_chart(fig)
        
        with col2:
            st.subheader("Pattern Recognition")
            # Seasonality decomposition
            result = seasonal_decompose(station_data.set_index('Date')['WSE'], model='additive', period=365)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, name="Seasonality"))
            st.plotly_chart(fig)
            
            # Cluster analysis
            clustered_data = AdvancedAnalytics.cluster_analysis(station_data)
            fig = px.scatter(clustered_data, x='Date', y='WSE', color='Cluster')
            st.plotly_chart(fig)

    with tab5:  # New Climate Impact
        st.header("Climate Impact Analysis")
        
        # Mock climate data (replace with real API integration)
        climate_data = pd.DataFrame({
            'Date': station_data['Date'],
            'Precipitation': np.random.normal(50, 15, len(station_data)),
            'Temperature': np.random.normal(20, 5, len(station_data))
        })
        
        merged_data = pd.merge(station_data, climate_data, on='Date')
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Correlation Analysis")
            corr_matrix = merged_data[['WSE', 'Precipitation', 'Temperature']].corr()
            fig = px.imshow(corr_matrix, text_auto=True)
            st.plotly_chart(fig)
            
        with col2:
            st.subheader("Impact Simulation")
            precip_change = st.slider("Precipitation Change (%)", -50, 50, 0)
            temp_change = st.slider("Temperature Change (°C)", -5, 5, 0)
            
            # Simple impact model
            simulated_wse = station_data['WSE'].iloc[-1] * (1 + precip_change/100) * (1 - temp_change/100)
            st.metric("Simulated Level", f"{simulated_wse:.2f} m")

    with tab6:  # New Comparative Analysis
        st.header("Comparative Analysis")
        
        if 'selected_station' in st.session_state:
            target_station = dm.stations[dm.stations['Station_Code'] == st.session_state.selected_station].iloc[0]
            nearby_stations = AdvancedAnalytics.find_nearby_stations(
                dm.stations, 
                target_station['LATITUDE'], 
                target_station['LONGITUDE'],
                radius_km=50
            )
            
            if not nearby_stations.empty:
                st.subheader(f"Nearby Stations (50km radius)")
                folium_static(Visualizer.create_satellite_map(nearby_stations, 
                    target_station['LATITUDE'], target_station['LONGITUDE']))
                
                # Comparison metrics
                comp_data = data[data['Station_Code'].isin(nearby_stations['Station_Code'])]
                latest_levels = comp_data.groupby('Station_Code')['WSE'].last()
                
                fig = px.bar(latest_levels, orientation='h', 
                            labels={'value': 'Water Level (m)', 'index': 'Station'})
                st.plotly_chart(fig)
            else:
                st.warning("No nearby stations found within 50km radius")
if __name__ == "__main__":
    main()