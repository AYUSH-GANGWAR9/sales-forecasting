# src/model_prophet.py
from prophet import Prophet
import pandas as pd
import joblib

def fit_prophet(df, date_col='date', target_col='sales', weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False):
    m = Prophet(weekly_seasonality=weekly_seasonality, yearly_seasonality=yearly_seasonality, daily_seasonality=daily_seasonality)
    dfp = df[[date_col, target_col]].rename(columns={date_col:'ds', target_col:'y'})
    m.fit(dfp)
    return m

def forecast_prophet(m, periods, freq='D'):
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return forecast  # contains yhat, yhat_lower, yhat_upper

def save_prophet(m, path):
    joblib.dump(m, path)

def load_prophet(path):
    return joblib.load(path)
