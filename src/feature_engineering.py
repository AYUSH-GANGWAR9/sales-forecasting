# src/feature_engineering.py
import pandas as pd
import numpy as np

def create_datetime_features(df, date_col='date'):
    df = df.copy()
    dt = pd.DatetimeIndex(df[date_col])
    df['month'] = dt.month
    df['day'] = dt.day
    df['dayofweek'] = dt.dayofweek
    df['weekofyear'] = dt.isocalendar().week.astype(int)
    df['is_weekend'] = (dt.dayofweek >= 5).astype(int)
    return df

def create_lags(df, target_col='sales', lags=[1,7,14,28]):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    return df

def create_rollings(df, target_col='sales', windows=[7,14,30]):
    df = df.copy()
    for w in windows:
        df[f'roll_mean_{w}'] = df[target_col].shift(1).rolling(w).mean()
        df[f'roll_std_{w}'] = df[target_col].shift(1).rolling(w).std()
    return df

def drop_na_for_model(df):
    return df.dropna().reset_index(drop=True)
