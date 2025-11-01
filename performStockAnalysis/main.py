import os
import pickle
import logging
import numpy as np
import pandas as pd
from camel_converter import to_camel
from datetime import datetime, timedelta

from modelDevelopment.main import develop_model
from dataPreparation.helper import _download_stock_data
from technicalIndicators.helper import get_all_technical_indicators
from dataPreparation.main import prepare_data_for_modelling_emiten, prepare_data_for_forecasting
from performStockAnalysis.helper import _initialize_repeatedly_used_variables, _combine_train_test_metrics_into_single_df, _save_developed_model, _save_csv_file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def select_emiten_to_model(quantile_threshold: float = 0.6) -> np.array:
    """
    Selects the most actively traded stocks from a master list

    This function reads a list of stock emitens from an Excel file, downloads
    their trading volume over the last 45 days, and filters for the top quantile_threshold
    most liquid stocks based on average daily volume. This ensures that models
    are built only for stocks with sufficient trading activity

    Args:
        quantile_threshold (float): The qunatile value for determining the selected emiten

    Returns:
        np.array: An array of selected stock emiten strings
    """
    logging.info("===== Starting stock selection process based on recent trading volume =====")
    data_saham = pd.read_csv('database/stocksInformation/stock_data_20251029.csv')
    start_date = (datetime.now().date() - timedelta(days=45)).strftime('%Y-%m-%d')
    
    logging.info(f"Fetching volume data for {len(data_saham)} stocks from {start_date} to today")
    data_saham['Average Volume'] = data_saham['Kode'].apply(lambda val: np.mean(_download_stock_data(val, start_date, '')['Volume']))
    
    threshold = np.nanquantile(data_saham['Average Volume'].values, quantile_threshold)
    selected_emiten = data_saham.loc[data_saham['Average Volume'] >= threshold, 'Kode'].values

    logging.info(f"===== Stock selection complete. Selected emiten total of {len(selected_emiten)} =====")

    return selected_emiten

def develop_models_for_selected_emiten(selected_emiten: list, label_types: list, rolling_windows: list):
    """
    Orchestrates the model development pipeline for a list of selected stocks

    For each stock emiten, this function will:
    1. Prepare the data by generating features and target variables
    2. Develop n distinct models for each rolling window for each m distinct label types
    3. Save each trained model to a file
    4. Aggregate the performance metrics of all models into summary DataFrames

    Args:
        selected_emiten (list): A list of stock emiten symbols to process
        label_types (list): A list of label types for model's target variables
        rolling_windows (list): A list of integers for the future statistic windows
    """
    logging.info(f"===== Starting Model Development for {len(selected_emiten)} Selected Emitens =====")
    
    list_of_variables = _initialize_repeatedly_used_variables(label_types, rolling_windows)
    developed_date = datetime.now().date().strftime('%Y%m%d')
    failed_stocks = []

    for i, emiten in enumerate(selected_emiten):
        try:
            logging.info(f"Processing Emiten: {emiten} ({i+1}/{len(selected_emiten)})")
            
            prepared_data = prepare_data_for_modelling_emiten(
                emiten=emiten, 
                start_date='2021-01-01', 
                end_date='', 
                target_column='Close',
                label_types=label_types,
                rolling_windows=rolling_windows
            )

            for label_type, (target_columns, threshold_columns, positive_label, negative_label) in zip(label_types, list_of_variables):
                for window, target_column, threshold_column in zip(rolling_windows, target_columns, threshold_columns):
                    logging.info(f"Developing the {label_type} {window} day rolling window model for {emiten}")
                    model, train_metrics, test_metrics = develop_model(prepared_data, target_column, positive_label, negative_label)

                    logging.info(f"Saving the developed {label_type} {window} day rolling window model for {emiten}")
                    _save_developed_model(model, label_type, emiten, f'{window}dd')

                    logging.info(f"Saving the developed {label_type} {window} day rolling window model's performance for {emiten}")
                    train_test = _combine_train_test_metrics_into_single_df(emiten, train_metrics, test_metrics)
                    train_test['Threshold'] = prepared_data[threshold_column].values[0]

                    filename = f'database/modelPerformances/{to_camel(label_type)}/{window}dd-{developed_date}.csv'
                    _ = _save_csv_file(train_test, filename)
                
            logging.info(f"Finished processing for Emiten: {emiten}")

        except:
            logging.warning(f"Failed processing for Emiten: {emiten}")

    failed_stock_path = f'database/modelPerformances/{to_camel(label_type)}/failedStocks-{developed_date}.txt'
    logging.info(f"Saving stocks that are failed being processed to '{failed_stock_path }'...")
    with open(failed_stock_path, "w") as file:
        for failed_stock in failed_stocks:
            file.write(failed_stock + "\n")
    logging.info("List of failed stocks saved successfully")

    logging.info(f"===== Finished Model Development for {len(selected_emiten)} Selected Emitens =====")

    return

def forecast_using_the_developed_models(all_forecast_dd: list, label_types: list, development_date: str, min_test_gini: float):
    """
    Orchestrates the forecasting pipeline for stocks that exceeds the minimum test gini performance
    
    Args:
        all_forecast_dd (list): A list of integers for the future statistic forecast
        label_types (list): A list of label types for model's target variables
        development_date (str): The date where the model is developed
        min_test_gini (float): The minimum gini performance for the model's testing performance
    """
    logging.info(f"===== Starting the Process of {'and '.join([f'{forecast_dd}dd' for forecast_dd in all_forecast_dd])} Days Forecasting on {' and '.join([' '.join(label_type.split('_')) for label_type in label_types])} Label Type =====")
    list_of_variables = _initialize_repeatedly_used_variables(label_types, all_forecast_dd)

    all_selected_emiten = None
    logging.info("Selecting which stocks to forecast")
    for label_type in label_types:
        for forecast_dd in all_forecast_dd:
            model_performance_dd_path = f'database/modelPerformances/{to_camel(label_type)}/{forecast_dd}dd-{development_date}.csv'
            logging.info(f"Loading the model's performance data from {model_performance_dd_path}")
            all_model_performances_days = pd.read_csv(model_performance_dd_path)

            if min_test_gini != None:
                logging.info(f'Select stocks with a Gini performance on testing data that is greater than {min_test_gini}')
                selected_model_performances_days = all_model_performances_days[all_model_performances_days['Test - Gini'] >= min_test_gini] \
                                                        .sort_values('Test - Gini', ascending=False) \
                                                        .reset_index(drop=True)
                selected_emiten = selected_model_performances_days['Kode'].unique()
                
            else:
                logging.info(f'Selecting all stocks available')
                selected_emiten = all_model_performances_days['Kode'].unique()
        
        if type(all_selected_emiten) == type(None):
            all_selected_emiten = selected_emiten
        else:
            all_selected_emiten = np.intersect1d(all_selected_emiten, selected_emiten)
    
    logging.info(f"Selected a total of {len(all_selected_emiten)} stocks that exceed the minimum model's performance")
    
    logging.info("Loading stock's technical indicators as features")
    feature_columns = get_all_technical_indicators()

    for emiten in all_selected_emiten:
        try:
            logging.info('Prepare all stock data to be used for forecasting')
            forecasting_data = prepare_data_for_forecasting(
                emiten=emiten, 
                start_date='2021-01-01', 
                end_date=''
            )

            for label_type, (target_columns, threshold_columns, positive_label, negative_label) in zip(label_types, list_of_variables):
                for forecast_dd, target_column, threshold_column in zip(all_forecast_dd, target_columns, threshold_columns):                    
                    logging.info(f"Starting the process of {label_type} {forecast_dd} Days forecasting for {emiten}")
                    
                    model_path = f'database/developedModels/{to_camel(label_type)}/{emiten}-{forecast_dd}dd-{development_date}.pkl'         
                    with open(model_path, 'rb') as file:
                        loaded_model = pickle.load(file)
            
                    forecast_column_name = f'Forecast {positive_label} {forecast_dd}dd'
                    forecasting_data[forecast_column_name] = forecasting_data.apply(
                        lambda row: loaded_model.predict_proba(row[feature_columns].values.reshape(1, -1))[0, list(loaded_model.classes_).index(positive_label)],
                        axis=1
                    )

                    selected_columns = ['Kode', 'Date', forecast_column_name]
                    forecasting_data_to_save = forecasting_data.loc[forecasting_data['Date'] == forecasting_data['Date'].max(), selected_columns]
                    forecast_path = f'database/forecastedStocks/{to_camel(label_type)}/forecast-{forecast_dd}dd-{development_date}.csv'  
                    _ = _save_csv_file(forecasting_data_to_save, forecast_path)

                    logging.info(f"Finished the process of {label_type} {forecast_dd} Days forecasting for {emiten}")

        except:
            logging.warning(f"Failed in the process of {label_type} {forecast_dd} days forecasting for {emiten}")
    
    logging.info(f"===== Finished the Process of {'and '.join([f'{forecast_dd}dd' for forecast_dd in all_forecast_dd])} Days Forecasting on {' and '.join([' '.join(label_type.split('_')) for label_type in label_types])} Label Type =====")

    return