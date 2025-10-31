import numpy as np
import pandas as pd

def _generate_median_gain(data: pd.DataFrame, target_column: str, rolling_window: int) -> (np.array, float):
    """
    (Internal Helper) Calculates the median gain of a target column based on a rolling window

    Args:
        data (pd.DataFrame): A pandas dataframe containing the daily price data
        target_column (str): The column name wished to be used as target data
        rolling_windows (int): The amount of upcoming target data used for creating the label

    Returns:
        np.array: The median gain for all target data
        float: The threshold for the quantile 0.9
    """
    median_close = data[target_column].rolling(rolling_window).quantile(0.4)
    median_gain = 100 * (median_close - data[target_column].values) / data[target_column].values
    threshold = np.nanquantile(median_gain, 0.9)
    
    return (median_gain, threshold)

def _bin_median_gain(threshold: float, val: float) -> str:
    """
    (Internal Helper) Perform binning for the median gain based on the threshold to create the labels for model development

    Args:
        threshold (float): The threshold for the quantile 0.9
        val (float): The median gain for the upcoming rolling window target variable
        
    Returns:
        str: Labels for developing the model
    """
    if np.isnan(val):
        return val
    if val >= threshold:
        return 'High Gain'
    else:
        return 'Low Gain'

def _generate_all_median_gain(data: pd.DataFrame, target_column: str, rolling_window: int) -> pd.DataFrame:
    """
    (Internal Helper) Generates a median gain label for each day based on a rolling window

    Args:
        data (pd.DataFrame): The input DataFrame containing stock data
        target_column (str): The name of the column to analyze
        rolling_window (int): The number of future days to look at for the median gain

    Returns:
        pd.DataFrame: The DataFrame with the new future median gain column added
    """
    column_name = f'Median Gain {rolling_window}dd'
    median_gain, threshold = _generate_median_gain(data, target_column, rolling_window)
    data[column_name] = [_bin_median_gain(threshold, val) for val in median_gain]
    data['Threshold ' + column_name] = threshold
    
    return data