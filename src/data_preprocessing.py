# src/data_preprocessing.py
import pandas as pd

def load_csv(path, date_col='date', parse_dates=True):
    if parse_dates:
        df = pd.read_csv(path, parse_dates=[date_col])
    else:
        df = pd.read_csv(path)
    df = df.sort_values(date_col).reset_index(drop=True)
    return df

def ensure_ts_format(df, date_col='date', target_col='sales'):
    """
    Ensure we have date and sales columns and clean basic issues.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, target_col]].dropna()
    df = df.groupby(date_col).sum().reset_index()  # aggregate if duplicates
    df = df.sort_values(date_col).reset_index(drop=True)
    return df
