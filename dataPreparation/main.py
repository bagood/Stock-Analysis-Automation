import logging
import pandas as pd

from technicalIndicators.main import generate_all_technical_indicators
from dataPreparation.helper import _download_stock_data
from dataPreparation.helper_linear_trend import _generate_all_linreg_gradients
from dataPreparation.helper_median_gain import _generate_all_median_gain
from dataPreparation.helper_max_loss import _generate_all_max_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def prepare_data_for_modelling(emiten: str, start_date: str, end_date: str, target_column: str, label_type: str, rolling_windows: list, download: bool = True) -> pd.DataFrame:
    """
    Orchestrates the full data preparation pipeline for a machine learning model

    This function serves as the main controller, executing a sequence of steps:
    1. Downloads or loads historical stock data
    2. Generates a comprehensive set of technical indicators to be used as model features
    3. Creates the target variable(s)
    4. Cleans the final dataset by removing any rows with missing values (NaNs)
       that result from the indicator and trend calculations

    Args:
        emiten (str): The stock ticker symbol
        start_date (str): The start date for the data ('YYYY-MM-DD')
        end_date (str): The end date for the data ('YYYY-MM-DD')
        target_column (str): The price column (e.g., 'Close') to use for label g
        rolling_windows (list): A list of integers for the future trend windows (e.g., [5, 10])
        download (bool): If True, downloads data from Yahoo Finance. If False, loads a local dummy file

    Returns:
        pd.DataFrame: A clean, feature-rich DataFrame ready for model training and evaluation
    """
    logging.info(f"Starting Data Preparation Pipeline for Ticker: {emiten}")

    if download:
        logging.info(f"Downloading stock data for ticker {emiten}.JK")
        data = _download_stock_data(emiten, start_date, end_date)
 
    else:
        logging.info("Loading data from local 'dummy_data.csv' file")
        data = pd.read_csv('dataPreparation/dummy_data.csv')
    
    logging.info("Generating technical indicators as features")
    data = generate_all_technical_indicators(data)

    if label_type == 'linear_trend':
        for window in rolling_windows:
            logging.info(f"Generating the {window}dd linear trend as target variables")
            data = _generate_all_linreg_gradients(data, target_column, window)
            
        logging.info('Dropping rows containing missing target variables')
        data.dropna(subset=[f'Linear Trend {window}dd' for window in rolling_windows], inplace=True)

    elif label_type == 'median_gain':
        for window in rolling_windows:
            logging.info(f"Generating the {window}dd median gain as target variables")
            data = _generate_all_median_gain(data, target_column, window)

        logging.info('Dropping rows containing missing target variables')
        data.dropna(subset=[f'Median Gain {window}dd' for window in rolling_windows], inplace=True)
    
    elif label_type == 'max_loss':
        for window in rolling_windows:
            logging.info(f"Generating the {window}dd max loss as target variables")
            data = _generate_all_max_loss(data, target_column, window)

        logging.info('Dropping rows containing missing target variables')
        data.dropna(subset=[f'Max Loss {window}dd' for window in rolling_windows], inplace=True)
    
    logging.info(f"Succesfully Executed the Data Preparation Pipeline for {emiten}")

    return data

def prepare_data_for_forecasting(emiten: str, start_date: str, end_date: str, rolling_window: int, download: bool = True) -> pd.DataFrame:
    """
    Orchestrates the full data preparation pipeline for making forecasts using the developed machine learning model

    This function serves as the helper, executing a sequence of steps:
    1. Downloads or loads historical stock data
    2. Generates a comprehensive set of technical indicators to be used as model features
    3. Gets the tail of the data for the forecasting data

    Args:
        emiten (str): The stock ticker symbol.
        start_date (str): The start date for the data ('YYYY-MM-DD')
        end_date (str): The end date for the data ('YYYY-MM-DD')
        rolling_window (int): An integers for the future median of price gain windows, correlates with the total of forecasting data
        download (bool): If True, downloads data from Yahoo Finance. If False, loads a local dummy file

    Returns:
        pd.DataFrame: A clean, feature-rich DataFrame ready for model training and evaluation
    """
    logging.info(f"Starting Data Preparation Pipeline for {emiten}")
    
    if download:
        logging.info(f"Downloading stock data for ticker {emiten}.JK")
        data = _download_stock_data(emiten, start_date, end_date)

    else:
        logging.info("Loading data from local 'dummy_data.csv' file")
        data = pd.read_csv('dataPreparation/dummy_data.csv')
        
    logging.info("Generating technical indicators as features")
    data = generate_all_technical_indicators(data)
    
    logging.info("Dropping all rows without any NaN values to create a clean forecasting dataset")
    forecasting_data = data.tail(rolling_window)
    
    logging.info("Added the emiten's code and date to the forecasting data")
    forecasting_data['Kode'] = emiten
    forecasting_data.reset_index(inplace=True)

    logging.info(f"Succesfully Prepare the Forecasting Data")

    return forecasting_data