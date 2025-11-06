# src/model_lstm.py
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def create_sequences(values, seq_len=30):
    X, y = [], []
    for i in range(len(values)-seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    return np.array(X), np.array(y)

def build_lstm(input_shape, units=64, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(series, seq_len=30, epochs=50, batch_size=32, model_path='lstm_model.h5'):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1,1))
    X, y = create_sequences(scaled.flatten(), seq_len=seq_len)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model = build_lstm((X.shape[1], X.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ck = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    model.fit(X, y, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[es, ck], verbose=2)
    joblib.dump(scaler, model_path + '.scaler')
    return model, scaler

def forecast_lstm(model, scaler, history_series, steps=14, seq_len=30):
    """
    history_series: np.array of raw values (not scaled) with length >= seq_len
    """
    preds = []
    hist = history_series.copy().tolist()
    for _ in range(steps):
        last_seq = np.array(hist[-seq_len:]).reshape(-1,1)
        scaled_seq = scaler.transform(last_seq).reshape(1, seq_len, 1)
        p = model.predict(scaled_seq, verbose=0)[0,0]
        # invert scale
        inv = scaler.inverse_transform([[p]])[0,0]
        preds.append(inv)
        hist.append(inv)
    return np.array(preds)
