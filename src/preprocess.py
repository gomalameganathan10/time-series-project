# src/preprocess.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_series(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1))
    return scaled, scaler

def create_dataset(data, window_size=20):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)
