import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from curl_cffi import requests

def _download_stock_data(emiten: str, start_date: str, end_date: str) -> pd.DataFrame: 
    """
    (Internal Helper) Downloads historical stock data from Yahoo Finance for a given ticker.

    This function fetches daily 'Open', 'High', 'Low', 'Close', and 'Volume' data.
    It automatically appends the '.JK' suffix, which is standard for tickers
    on the Jakarta Stock Exchange (IDX). It also performs basic data cleaning
    by removing non-essential columns and standardizing the date format.

    Args:
        emiten (str): The stock ticker symbol (e.g., 'BBCA').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
                          If empty, the download will start from the earliest available date.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.
                        If empty, the download will go up to the most recent date.

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned historical stock data,
                      or None if the download fails.
    """
    session = requests.Session(impersonate="chrome123")
    ticker = yf.Ticker(f"{emiten}.JK", session=session)

    start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.strptime('2021-01-01', '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
    data = ticker.history(start=start, end=end)

    columns_to_drop = ['Dividends', 'Stock Splits', 'Capital Gains']
    for col in columns_to_drop:
        if col in data.columns:
            data.drop(columns=[col], inplace=True)

    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.date

    return data
    
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