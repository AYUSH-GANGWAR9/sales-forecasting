# src/model_arima.py
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

def fit_sarima(train_series, order=(1,1,1), seasonal_order=(0,1,1,12), exog=None):
    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, exog=exog, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res

def forecast_sarima(res, steps, exog_future=None):
    pred = res.get_forecast(steps=steps, exog=exog_future)
    mean = pred.predicted_mean
    conf_int = pred.conf_int()
    return mean, conf_int

def save_model(res, path):
    joblib.dump(res, path)

def load_model(path):
    return joblib.load(path)
