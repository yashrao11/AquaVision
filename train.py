import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import optuna
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

# Define feature list (ensure this matches your preprocessing)
FEATURES = ['Year', 'Month', 'Day', 'fourier_sin', 'fourier_cos',
            'WSE_lag_1', 'WSE_lag_7', 'WSE_lag_30', 'Elevation', 'Well_Depth']

def create_hybrid_model():
    """Create a hybrid CNN-LSTM model"""
    model = Sequential([
        Conv1D(128, 3, activation='relu', input_shape=(30, len(FEATURES))),
        LSTM(256, return_sequences=True, dropout=0.3),
        LSTM(128, dropout=0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='reg:squarederror')
    return model

def objective(trial):
    """Optuna optimization for XGBoost"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'tree_method': 'gpu_hist' if tf.config.list_physical_devices('GPU') else 'hist'
    }
    
    model = xgb.XGBRegressor(**params)
    scores = []
    tscv = TimeSeriesSplit(n_splits=3)
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        scores.append(mean_absolute_error(y_val, preds))
    
    return np.mean(scores)

def train_ensemble(X, y):
    """Train stacked ensemble model"""
    base_models = [
        ('xgb', xgb.XGBRegressor(tree_method='gpu_hist')),
        ('lgb', lgb.LGBMRegressor(device='gpu')),
        ('lstm', tf.keras.wrappers.scikit_learn.KerasRegressor(
            build_fn=create_hybrid_model, epochs=50, batch_size=1024, verbose=0))
    ]
    
    ensemble = StackingRegressor(
        estimators=base_models,
        final_estimator=RidgeCV(),
        cv=TimeSeriesSplit(n_splits=3)
    )
    
    ensemble.fit(X, y)
    return ensemble

def main():
    # Load processed data
    df = pd.read_parquet('processed_data.parquet')
    X, y = df.drop(columns=['WSE']), df['WSE']
    
    # Optimize XGBoost
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, timeout=3600)
    
    # Train best XGBoost
    best_xgb = xgb.XGBRegressor(**study.best_params)
    best_xgb.fit(X, y)
    
    # Train hybrid ensemble
    ensemble = train_ensemble(X, y)
    
    # Save models and features
    joblib.dump({
        'xgb': best_xgb,
        'ensemble': ensemble,
        'features': X.columns.tolist()
    }, 'model_artifacts.joblib')

if __name__ == "__main__":
    main()
