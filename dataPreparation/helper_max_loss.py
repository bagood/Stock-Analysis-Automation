import numpy as np
import pandas as pd
import yfinance as yf

def _generate_max_loss(data: pd.DataFrame, target_column: str, rolling_window: int) -> (np.array, float):
    """
    (Internal Helper) Calculates the max loss of a target column based on a rolling window

    Args:
        data (pd.DataFrame): A pandas dataframe containing the daily price data
        target_column (str): The column name wished to be used as target data
        rolling_windows (int): The amount of upcoming target data used for creating the label

    Returns:
        np.array: A collection of max loss based on a rolling window
        float: The threshold for the quantile 0.6
    """
    min_close = data[target_column].rolling(rolling_window).min()
    max_loss = 100 * (min_close - data[target_column].values) / data[target_column].values
    threshold = np.nanquantile(max_loss, 0.6)
    
    return (max_loss, threshold)

def _bin_max_loss(threshold: float, val: float) -> str:
    """
    (Internal Helper) Perform binning for the max loss based on the threshold to create the labels for model development

    Args:
        threshold (float): The threshold for the quantile 0.6
        val (float): The max loss based on a rolling window for the label
        
    Returns:
        str: Labels for developing the model
    """
    if np.isnan(val):
        return val
    if val >= threshold:
        return 'Low Risk'
    else:
        return 'High Risk'

def _generate_all_max_loss(data: pd.DataFrame, target_column: str, rolling_window: int) -> pd.DataFrame:
    """
    (Internal Helper) Generates a max loss label for each day based on a rolling window

    Args:
        data (pd.DataFrame): The input DataFrame containing stock data
        target_column (str): The name of the column to analyze
        rolling_window (int): The number of future days to look at for the max loss

    Returns:
        pd.DataFrame: The DataFrame with the new future max loss column added
    """
    column_name = f'Max Loss {rolling_window}dd'
    max_loss, threshold = _generate_max_loss(data, target_column, rolling_window)
    data[column_name] = [_bin_max_loss(threshold, val) for val in max_loss]
    data['Threshold ' + column_name] = threshold
    
    return data