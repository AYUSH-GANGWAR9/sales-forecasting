# src/ensemble.py
import numpy as np
import pandas as pd

def weighted_ensemble(preds_dict, weights=None):
    """
    preds_dict: {'arima': arr, 'prophet': arr, 'lstm': arr}
    weights: dict with same keys or None -> equal weights
    """
    keys = list(preds_dict.keys())
    if weights is None:
        weights = {k: 1.0/len(keys) for k in keys}
    assert set(keys) == set(weights.keys())
    arrs = np.stack([preds_dict[k] * weights[k] for k in keys], axis=0)
    combined = arrs.sum(axis=0)
    return combined
