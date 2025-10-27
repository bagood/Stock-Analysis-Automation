import os
import pickle
import logging
import numpy as np
import pandas as pd
from camel_converter import to_camel
from datetime import datetime, timedelta

from modelDevelopment.main import develop_model
from dataPreparation.helper import _download_stock_data
from dataPreparation.main import prepare_data_for_modelling, prepare_data_for_forecasting
from performStockAnalysis.helper import _initialize_repeatedly_used_variables, _combine_train_test_metrics_into_single_df, _save_developed_model, _save_csv_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def select_emiten_to_model(quantile_threshold: float = 0.6) -> np.array:
    """
    Selects the most actively traded stocks from a master list

    This function reads a list of stock tickers from an Excel file, downloads
    their trading volume over the last 45 days, and filters for the top 50%
    most liquid stocks based on average daily volume. This ensures that models
    are built only for stocks with sufficient trading activity

    Args:
        n_kode_saham_limit (int, optional): The number of stocks to process from the
                                          top of the Excel list. If 0, all stocks
                                          are considered. Defaults to 0

    Returns:
        np.array: An array of selected stock ticker strings
    """
    logging.info("Starting stock selection process based on recent trading volume")
    data_saham = pd.read_excel('performStockAnalysis/daftar_saham.xlsx')
    start_date = (datetime.now().date() - timedelta(days=45)).strftime('%Y-%m-%d')
    
    logging.info(f"Fetching volume data for {len(data_saham)} stocks from {start_date} to today")
    data_saham['Average Volume'] = data_saham['Kode'].apply(lambda val: np.mean(_download_stock_data(val, start_date, '')['Volume']))
    
    threshold = np.nanquantile(data_saham['Average Volume'].values, quantile_threshold)
    selected_emiten = data_saham.loc[data_saham['Average Volume'] >= threshold, 'Kode'].values
    logging.info(f"Stock selection complete. Selected ticker total of {len(selected_emiten)}")

    return selected_emiten

def develop_models_for_selected_emiten(selected_emiten: list, label_type: str, rolling_windows: list):
    """
    Orchestrates the model development pipeline for a list of selected stocks

    For each stock ticker, this function will:
    1. Prepare the data by generating features and target variables
    2. Develop n distinct models for each rolling window
    3. Save each trained model to a file
    4. Aggregate the performance metrics of all models into summary DataFrames

    Args:
        selected_emiten (list): A list of stock ticker symbols to process
    """
    logging.info(f"Starting Bulk Model Development for {len(selected_emiten)} Selected Stocks")
    
    target_columns, threshold_columns, positive_label, negative_label = _initialize_repeatedly_used_variables(label_type, rolling_windows)
    
    developed_date = datetime.now().date().strftime('%Y%m%d')
    failed_stocks = []

    for i, emiten in enumerate(selected_emiten):
        try:
            logging.info(f"Processing Emiten: {emiten} ({i+1}/{len(selected_emiten)})")
            
            logging.info(f"Preparing data for {emiten}")
            prepared_data = prepare_data_for_modelling(
                emiten=emiten, 
                start_date='2021-01-01', 
                end_date='', 
                target_column='Close',
                label_type=label_type,
                rolling_windows=rolling_windows, 
                download=True
            )

            for window, target_column, threshold_column in zip(rolling_windows, target_columns, threshold_columns):
                logging.info(f"Developing model for '{emiten}' - {window} Day Rolling Window")
                model, train_metrics, test_metrics = develop_model(prepared_data, target_column, positive_label, negative_label)

                logging.info(f"Saving models and collating performance metrics for {emiten}")
                _save_developed_model(model, label_type, emiten, f'{window}dd')

                logging.info(f"Measuring model performances on training and testing sets")
                train_test = _combine_train_test_metrics_into_single_df(emiten, train_metrics, test_metrics)
                train_test['Threshold'] = prepared_data[threshold_column].values[0]
        
                filename = f'database/modelPerformances/{to_camel(label_type)}/{window}dd-{developed_date}.csv'
                _ = _save_csv_file(train_test, filename)
                
            logging.info(f"Finished processing for Ticker: {emiten}")

        except:
            logging.warning(f"Failed processing for Ticker: {emiten}")

    failed_stock_path = f'database/modelPerformances/{to_camel(label_type)}/failedStocks-{developed_date}.txt'
    logging.info(f"Saving stocks that are failed being processed to '{failed_stock_path }'...")
    with open(failed_stock_path, "w") as file:
        for failed_stock in failed_stocks:
            file.write(failed_stock + "\n")
    logging.info("List of failed stocks saved successfully")

    logging.info("Bulk Model Development Complete")

    return

def forecast_using_the_developed_models(forecast_dd: int, label_type: str, development_date: str, min_test_gini: float):
    """
    Orchestrates the forecasting pipeline for stocks that exceeds the minimum test gini performance
    
    Args:
        forecast_dd (int): The desired upcoming days to be forcasted
        development_date (str): The date where the model is developed
        min_test_gini (float): The minimum gini performance for the model's testing performance
    """
    logging.info(f"Starting the Process of {forecast_dd} Days Forecasting")
    _, _, positive_label, negative_label = _initialize_repeatedly_used_variables(label_type)

    model_performance_dd_path = f'database/modelPerformances/{to_camel(label_type)}/{forecast_dd}dd-{development_date}.csv'
    logging.info(f"Loading the data from {model_performance_dd_path}")
    all_model_performances_days = pd.read_csv(model_performance_dd_path)
    
    logging.info(f'Select stock with a performance on testing data greater than {min_test_gini}')
    selected_model_performances_days = all_model_performances_days[all_model_performances_days['Test - Gini'] >= min_test_gini] \
                                            .sort_values('Test - Gini', ascending=False) \
                                            .reset_index(drop=True)

    selected_emiten = selected_model_performances_days['Kode'].unique()
    logging.info(f"Selected a total of {len(selected_emiten)} stocks that exceed the minimum model's performance")

    feature_file = 'modelDevelopment/technical_indicator_features.txt'
    logging.info(f'Loading feature names from {feature_file}')
    with open(feature_file, "r") as file:
        feature_columns = [line.strip() for line in file]
    logging.info(f'Loaded {len(feature_columns)} features')    

    for emiten in selected_emiten:
        try:
            logging.info('Prepare all stock data to be used for forecasting')
            forecasting_data = prepare_data_for_forecasting(
                emiten=emiten, 
                start_date='2021-01-01', 
                end_date='', 
                rolling_window=forecast_dd, 
                download=True
            )

            logging.info(f'Loading the developed {forecast_dd} days model for {emiten}')
            model_path = f'database/developedModels/{to_camel(label_type)}/{emiten}-{forecast_dd}dd-{development_date}.pkl'         
            with open(model_path, 'rb') as file:
                loaded_model = pickle.load(file)
            logging.info(f'Sucessfully loaded the developed {forecast_dd} days model for {emiten}')

            logging.info(f'Start forecasting using the loaded {forecast_dd} days model on the prepared forecasting data for {emiten}')
            forecast_column_name = f'Forecast {positive_label} {forecast_dd}dd'
            forecasting_data[forecast_column_name] = forecasting_data.apply(
                lambda row: loaded_model.predict_proba(row[feature_columns].values.reshape(1, -1))[0, list(loaded_model.classes_).index(positive_label)],
                axis=1
            )

            logging.info(f'Saving the {forecast_dd} days forecast result for {emiten}')
            selected_columns = ['Kode', 'Date', forecast_column_name]
            forecasting_data_to_save = forecasting_data.loc[forecasting_data['Date'] == forecasting_data['Date'].max(), selected_columns]
            forecast_path = f'database/forecastedStocks/{to_camel(label_type)}/forecast-{forecast_dd}dd-{development_date}.csv'  
            _ = _save_csv_file(forecasting_data_to_save, forecast_path)

            logging.info(f"Finished the process of {forecast_dd} Days forecasting for {emiten}")

        except:
            logging.warning(f"Failed in process of {forecast_dd} days forecasting for {emiten}")

    return