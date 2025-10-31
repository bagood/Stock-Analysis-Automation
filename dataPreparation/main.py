import logging
import pandas as pd

from technicalIndicators.main import generate_all_technical_indicators
from dataPreparation.helper import _download_stock_data, _generate_labels_based_on_label_type

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def prepare_data_for_modelling_emiten(emiten: str, start_date: str, end_date: str, target_column: str, label_type: str, rolling_windows: list) -> pd.DataFrame:
    """
    Orchestrates the full data preparation pipeline for a machine learning model

    This function serves as the main controller, executing a sequence of steps:
    1. Downloads historical stock data
    2. Generates a comprehensive set of technical indicators to be used as model features
    3. Creates the target variable(s)
    4. Cleans the final dataset by removing any rows with missing values (NaNs) that result from the indicator and label generation

    Args:
        emiten (str): The stock emiten symbol
        start_date (str): The start date for the data ('YYYY-MM-DD')
        end_date (str): The end date for the data ('YYYY-MM-DD')
        target_column (str): The target column to use for label generation
        rolling_windows (list): A list of integers for the future statistic windows

    Returns:
        pd.DataFrame: A clean, feature-rich DataFrame ready for model training and evaluation
    """
    logging.info(f"----- Starting Data Preparation Pipeline for Emiten {emiten} -----")

    logging.info("Downloading the stock data")
    data = _download_stock_data(emiten, start_date, end_date)
 
    logging.info("Generating technical indicators as features")
    data = generate_all_technical_indicators(data)

    logging.info(f"Generating {'and '.join([f'{window}dd' for window in rolling_windows])} {' '.join(label_type.split('_'))} as target variables")
    data = _generate_labels_based_on_label_type(data, target_column, rolling_windows, label_type)

    logging.info(f"----- Succesfully Executed the Data Preparation Pipeline for Emiten {emiten} -----")

    return data

def prepare_data_for_modelling_industry(industry: str, start_date: str, end_date: str, target_column: str, label_type: str, rolling_windows: list) -> pd.DataFrame:
    """
    Orchestrates the full data preparation pipeline for a machine learning model on emitens' industry

    This function serves as the main controller, executing a sequence of steps:
    1. Downloads historical stock data belonged to a specific industry
    2. Generates a comprehensive set of technical indicators to be used as model features
    3. Creates the target variable(s)
    4. Cleans the final dataset by removing any rows with missing values (NaNs) that result from the indicator and label generation

    Args:
        industry (str): The stock emiten symbol
        start_date (str): The start date for the data ('YYYY-MM-DD')
        end_date (str): The end date for the data ('YYYY-MM-DD')
        target_column (str): The target column to use for label generation
        rolling_windows (list): A list of integers for the future statistic windows

    Returns:
        pd.DataFrame: A clean, feature-rich DataFrame ready for model training and evaluation
    """
    logging.info(f"----- Starting Data Preparation Pipeline for Industry {industry} -----")

    logging.info("filtering emitens belonged for the industry")
    stock_information_data = pd.read_csv('database/stocksInformation/stock_data_20251029.csv')
    filtered_stock_information_data = stock_information_data.loc[stock_information_data['Industry'] == industry, :]
    
    logging.info(f"Acquired {len(filtered_stock_information_data)} emitens")
    all_industry_data = pd.DataFrame()
    for i, emiten in enumerate(filtered_stock_information_data['Kode'].values):
        logging.info(f"Processing {i+1} out of {len(filtered_stock_information_data)} emitens")
        
        logging.info("Downloading stock data")
        data = _download_stock_data(emiten, start_date, end_date)
     
        logging.info("Generating technical indicators as features")
        data = generate_all_technical_indicators(data)

        logging.info("Appending the currenlty generated emiten data to all emitens data")
        all_industry_data = pd.concat((all_industry_data, data))

    logging.info(f"Generating {'and '.join([f'{window}dd' for window in rolling_windows])} {' '.join(label_type.split('_'))} as target variables")
    all_industry_data = _generate_labels_based_on_label_type(all_industry_data, target_column, rolling_windows, label_type)
    all_industry_data.sort_index(inplace=True)
    
    logging.info(f"----- Succesfully Executed the Data Preparation Pipeline for Industry {industry} -----")
    
    return all_industry_data

def prepare_data_for_forecasting(emiten: str, start_date: str, end_date: str, rolling_window: int) -> pd.DataFrame:
    """
    Orchestrates the full data preparation pipeline for making forecasts using the developed machine learning model

    This function serves as the main controller, executing a sequence of steps:
    1. Downloads historical stock data
    2. Generates a comprehensive set of technical indicators to be used as model features
    3. Gets the tail of the data for the forecasting data

    Args:
        emiten (str): The stock emiten symbol.
        start_date (str): The start date for the data ('YYYY-MM-DD')
        end_date (str): The end date for the data ('YYYY-MM-DD')
        rolling_window (int): An integers for the future median of price gain windows, correlates with the total of forecasting data
        download (bool): If True, downloads data from Yahoo Finance. If False, loads a local dummy file

    Returns:
        pd.DataFrame: A clean, feature-rich DataFrame ready for model training and evaluation
    """
    logging.info(f"----- Starting Data Preparation Pipeline for Emiten {emiten} -----")
    
    logging.info("Downloading stock data")
    data = _download_stock_data(emiten, start_date, end_date)
    
    logging.info("Generating technical indicators as features")
    data = generate_all_technical_indicators(data)
    
    logging.info("Finalizing the forecasting dataset")
    forecasting_data = data.tail(rolling_window)
    forecasting_data['Kode'] = emiten
    forecasting_data.reset_index(inplace=True)

    logging.info(f"----- Succesfully Prepare the Forecasting Data for Emiten {emiten} -----")

    return forecasting_data