import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
    
def _generate_linreg_gradient(target_data: np.array) -> float:
    """
    (Internal Helper) Calculates the slope of a numpy arrays using linear regression

    Args:
        target_data (np.array): A numpy array of numerical data

    Returns:
        float: The calculated slope (gradient) of the regression line
    """
    X = np.arange(len(target_data)).reshape(-1, 1)
    y = target_data - target_data[0]

    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    linreg_gradient = model.coef_[0]
    
    return linreg_gradient

def _bin_linreg_gradients(val: float) -> str:
    """
    (Internal Helper) Classifies a slope value into a categorical trend direction

    Args:
        val (float): The slope value from the linear regression calculation

    Returns:
        str: 'Up Trend' for positive or zero slopes, 'Down Trend' for negative slopes, or NaN if the input is NaN
    """
    if np.isnan(val):
        return val
    if val < 0:
        return 'Down Trend'
    else:
        return 'Up Trend'

def _generate_all_linreg_gradients(data: pd.DataFrame, target_column: str, rolling_window: int) -> pd.DataFrame:
    """
    (Internal Helper) Generates a future trend label for each day based on a rolling window

    Args:
        data (pd.DataFrame): The input DataFrame containing stock data
        target_column (str): The name of the column to analyze
        rolling_window (int): The number of future days to look at for the trend

    Returns:
        pd.DataFrame: The DataFrame with the new future trend column added
    """
    column_name = f'Linear Trend {rolling_window}dd'    
    target_data = data[target_column].values

    linreg_gradients = [
        _generate_linreg_gradient(target_data[i+1 : i+1+rolling_window])
        for i in range(len(target_data) - rolling_window)
    ]

    padding = [np.nan] * rolling_window
    full_gradient_list = linreg_gradients + padding

    data[column_name] = full_gradient_list
    data[column_name] = data[column_name].apply(_bin_linreg_gradients)

    data['Threshold ' + column_name] = np.nan()
    
    return data