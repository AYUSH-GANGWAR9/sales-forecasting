# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_preprocessing import load_csv, ensure_ts_format
from src.model_prophet import fit_prophet, forecast_prophet
from src.model_lstm import train_lstm, forecast_lstm
from src.model_arima import fit_sarima, forecast_sarima
from src.ensemble import weighted_ensemble
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sales Forecasting", layout="wide")
st.title("Sales Forecasting — ARIMA / Prophet / LSTM / Ensemble")

uploaded = st.file_uploader("Upload sales CSV (must have date,sales columns)", type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded, parse_dates=['date'])
    df = ensure_ts_format(df, date_col='date', target_col='sales')
    st.write("Data sample:", df.head())

    periods = st.number_input("Forecast horizon (days)", value=30, min_value=1, max_value=365)
    model_choice = st.multiselect("Select models to run", ['prophet', 'arima', 'lstm'], default=['prophet','lstm'])

    if st.button("Run Forecast"):
        st.info("Training / Forecasting — this may take a while for LSTM.")
        results = {}
        if 'prophet' in model_choice:
            m = fit_prophet(df)
            fc = forecast_prophet(m, periods=periods)
            prophet_pred = fc.tail(periods)['yhat'].values
            results['prophet'] = prophet_pred

        if 'arima' in model_choice:
            # quick auto-diff: simple SARIMA on aggregated series
            ts = df.set_index('date')['sales']
            res = fit_sarima(ts, order=(1,1,1), seasonal_order=(0,1,1,7))
            mean, ci = forecast_sarima(res, steps=periods)
            results['arima'] = mean.values

        if 'lstm' in model_choice:
            series = df['sales'].values
            seq_len = st.slider("LSTM sequence length", min_value=7, max_value=60, value=30)
            model, scaler = train_lstm(series, seq_len=seq_len, epochs=20, batch_size=16, model_path='lstm_best.h5')
            preds = forecast_lstm(model, scaler, series, steps=periods, seq_len=seq_len)
            results['lstm'] = preds

        # Ensemble
        if len(results) > 1:
            weights = {k: 1.0/len(results) for k in results.keys()}
            ensemble_preds = weighted_ensemble(results, weights=weights)
            results['ensemble'] = ensemble_preds

        # Build forecast DataFrame
        last_date = df['date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq='D')
        out = pd.DataFrame({'date': future_dates})
        for k, v in results.items():
            out[k] = np.round(v, 2)
        st.success("Done")
        st.write(out)

        # Plot
        st.line_chart(out.set_index('date'))

        csv = out.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast CSV", data=csv, file_name='forecast.csv', mime='text/csv')
