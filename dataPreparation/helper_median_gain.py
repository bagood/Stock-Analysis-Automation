import numpy as np
import pandas as pd
import yfinance as yf

def _generate_median_gain(data: pd.DataFrame, target_column: str, rolling_window: int) -> (np.array, float):
    median_close = data[target_column].rolling(rolling_window).quantile(0.4)
    median_gain = 100 * (median_close - data[target_column].values) / data[target_column].values
    threshold = np.nanquantile(median_gain, 0.85)
    
    return (median_gain, threshold)

def _bin_median_gain(threshold: float, val: float) -> str:
    if np.isnan(val):
        return val
    if val >= threshold:
        return 'High Gain'
    else:
        return 'Low Gain'

def _generate_all_median_gain(data: pd.DataFrame, target_column: str, rolling_window: int) -> pd.DataFrame:
    column_name = f'Median Gain {rolling_window}dd'
    median_gain, threshold = _generate_median_gain(data, target_column, rolling_window)
    data[column_name] = [_bin_median_gain(threshold, val) for val in median_gain]
    
    return data