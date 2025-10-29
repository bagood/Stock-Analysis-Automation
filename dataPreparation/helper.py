import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from curl_cffi import requests

from dataPreparation.helper_linear_trend import _generate_all_linreg_gradients
from dataPreparation.helper_median_gain import _generate_all_median_gain
from dataPreparation.helper_max_loss import _generate_all_max_loss

def _download_stock_data(emiten: str, start_date: str, end_date: str) -> pd.DataFrame: 
    """
    (Internal Helper) Downloads historical stock data from Yahoo Finance for a given emiten

    This function fetches daily 'Open', 'High', 'Low', 'Close', and 'Volume' data
    It automatically appends the '.JK' suffix, which is standard for emitens
    on the Jakarta Stock Exchange (IDX). It also performs basic data cleaning
    by removing non-essential columns and standardizing the date format

    Args:
        emiten (str): The stock emiten symbol (e.g., 'BBCA')
        start_date (str): The start date for the data in 'YYYY-MM-DD' format
                          If empty, the download will start from the earliest available date
        end_date (str): The end date for the data in 'YYYY-MM-DD' format
                        If empty, the download will go up to the most recent date

    Returns:
        pd.DataFrame: A DataFrame containing the cleaned historical stock data,
                      or None if the download fails
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
    
    try:
        data['Date'] = data['Date'].dt.date
    except:
        pass

    return data

def _generate_labels_based_on_label_type(data, target_column, rolling_windows, label_type):
    if label_type in 'linear_trend':
        for window in rolling_windows:
            data = _generate_all_linreg_gradients(data, target_column, window)
            
        data.dropna(subset=[f'Linear Trend {window}dd' for window in rolling_windows], inplace=True)

    elif label_type == 'median_gain':
        for window in rolling_windows:
            data = _generate_all_median_gain(data, target_column, window)

        data.dropna(subset=[f'Median Gain {window}dd' for window in rolling_windows], inplace=True)
    
    elif label_type == 'max_loss':
        for window in rolling_windows:
            data = _generate_all_max_loss(data, target_column, window)

        data.dropna(subset=[f'Max Loss {window}dd' for window in rolling_windows], inplace=True)
    
    return data