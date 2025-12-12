# src/data.py
import numpy as np
import pandas as pd

def generate_synthetic_series(n_points=500, seed=42):
    np.random.seed(seed)
    t = np.arange(n_points)
    series = np.sin(0.02 * t) + 0.1 * np.random.randn(n_points)
    return pd.DataFrame({'value': series})

if __name__ == "__main__":
    print(generate_synthetic_series().head())

