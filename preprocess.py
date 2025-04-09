import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
import dask.dataframe as dd
import joblib
import holidays
import pygeohash as gh
from tqdm import tqdm
import geopandas as gpd  # Added missing import
from sklearn.neighbors import BallTree  # Added missing import
from geopy.distance import great_circle  # Added missing import

def create_spatiotemporal_features(df):
    """Create advanced spatiotemporal features"""
    # Temporal features
    df['date'] = pd.to_datetime(df['MSMT_DATE'])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    
    # Fourier transformations for seasonality
    for period in [365, 182, 90]:
        df[f'sin_{period}'] = np.sin(2 * np.pi * df['day_of_year']/period)
        df[f'cos_{period}'] = np.cos(2 * np.pi * df['day_of_year']/period)
    
    # Holiday effects
    us_holidays = holidays.US()
    df['is_holiday'] = df['date'].apply(lambda x: x in us_holidays)
    
    # Lag features with expanding windows
    for lag in [1, 7, 30, 90, 365]:
        df[f'lag_{lag}'] = df.groupby('STATION')['WSE'].shift(lag)
        df[f'rolling_mean_{lag}'] = df.groupby('STATION')['WSE'].transform(
            lambda x: x.rolling(lag, min_periods=1).mean())
    
    # Rate of change features
    df['daily_change'] = df.groupby('STATION')['WSE'].diff()
    df['weekly_change'] = df.groupby('STATION')['WSE'].diff(7)
    
    return df

def process_monthly_data(df):
    """Integrate monthly data with different frequencies"""
    monthly = pd.read_csv('gwl-monthly.csv', parse_dates=['MSMT_DATE'])
    monthly = monthly.rename(columns={'MSMT_DATE': 'date', 'STATION': 'STATION'})
    
    # Monthly aggregations
    monthly['month'] = monthly['date'].dt.month
    monthly['quarter'] = monthly['date'].dt.quarter
    
    # Merge with daily data
    df['month'] = df['date'].dt.month
    df = df.merge(monthly, on=['STATION', 'month'], how='left', suffixes=('', '_monthly'))
    
    return df

def geospatial_features(df, stations):
    """Create advanced geospatial features"""
    # Geohash encoding
    df['geohash'] = df.apply(lambda x: gh.encode(x['LATITUDE'], x['LONGITUDE'], precision=7), axis=1)
    
    # Distance to nearest neighbor using geopandas and BallTree
    stations_gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations.LONGITUDE, stations.LATITUDE)
    )
    tree = BallTree(stations_gdf[['LATITUDE', 'LONGITUDE']].values, leaf_size=40)
    _, indices = tree.query(df[['LATITUDE', 'LONGITUDE']].values, k=2)
    df['dist_to_nearest'] = df.apply(
        lambda row: great_circle(
            (row['LATITUDE'], row['LONGITUDE']),
            (stations_gdf.iloc[indices[row.name,1]]['LATITUDE'],
             stations_gdf.iloc[indices[row.name,1]]['LONGITUDE'])
        ).meters, axis=1
    )
    
    # Elevation differentials
    df['elevation_diff'] = df['ELEVATION'] - df.groupby('geohash')['ELEVATION'].transform('mean')
    
    return df

def feature_selection(df, target='WSE', top_n=30):
    """Select most important features using mutual information"""
    X = df.select_dtypes(include=np.number).dropna()
    y = X.pop(target)
    
    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    selected_features = mi_scores.sort_values(ascending=False).index[:top_n]
    
    return selected_features

def main():
    # Load data with Dask for out-of-core processing
    ddf = dd.read_csv('gwl-daily.csv', parse_dates=['MSMT_DATE'])
    stations = pd.read_csv('gwl-stations.csv')
    
    # Process in chunks
    processed_chunks = []
    for partition in tqdm(ddf.partitions, total=ddf.npartitions):
        pdf = partition.compute()
        pdf = create_spatiotemporal_features(pdf)
        pdf = process_monthly_data(pdf)
        pdf = geospatial_features(pdf, stations)
        processed_chunks.append(pdf)
    
    # Combine and save
    final_df = pd.concat(processed_chunks)
    selected_features = feature_selection(final_df)
    final_df[selected_features].to_parquet('processed_data.parquet')
    
    # Save feature processor
    preprocessor = make_pipeline(
        ColumnTransformer([
            ('num', RobustScaler(), final_df.select_dtypes(include=np.number).columns),
            ('cat', OneHotEncoder(), ['geohash', 'BASIN_NAME'])
        ])
    )
    joblib.dump(preprocessor, 'preprocessor.joblib')

if __name__ == "__main__":
    main()
