import os
import pickle
import pandas as pd
from datetime import datetime
from camel_converter import to_camel

def _write_or_append_list_to_txt(list_value: list, file_path: str, write_mode: str):
    with open(file_path, write_mode) as file:
        for val in list_value:
            file.write(val + "\n")
    
    return

def _read_txt_as_list(file_path: str) -> list:
    with open(file_path, "r") as file:
        list_value = [line.strip() for line in file]
    
    return list_value

def _initialize_repeatedly_used_variables(label_types: list, rolling_windows: list = None) -> list:
    """
    (Internal Helper) Initialize repeatedyly used variables based on the given inputs
    
    Args:
        label_types (list): A list of label types for model's target variables
        rolling_windows (list): A list of integers for the future statistic windows
    
    Returns:
        list: An array containing several arrays with each contains target_columns, threshold_columns, positive_label, negative_label
    """
    list_of_variables = []
    for label_type in label_types:
        target_columns = None
        threshold_columns = None

        if label_type in ['linear_trend', 'linearTrend']:
            if rolling_windows != None:
                target_columns = [f'Linear Trend {window}dd' for window in rolling_windows]
                threshold_columns = [f'Threshold Linear Trend {window}dd' for window in rolling_windows]
            positive_label = 'Up Trend'
            negative_label = 'Down Trend'

        elif label_type in ['median_gain', 'medianGain']:
            if rolling_windows != None:
                target_columns = [f'Median Gain {window}dd' for window in rolling_windows]
                threshold_columns = [f'Threshold Median Gain {window}dd' for window in rolling_windows]
            positive_label = 'High Gain'
            negative_label = 'Low Gain'
        
        elif label_type in ['max_loss', 'maxLoss']:
            if rolling_windows != None:
                target_columns = [f'Max Loss {window}dd' for window in rolling_windows]
                threshold_columns = [f'Threshold Max Loss {window}dd' for window in rolling_windows]
            positive_label = 'Low Risk'
            negative_label = 'High Risk'
        
        variables = [target_columns, threshold_columns, positive_label, negative_label]
        list_of_variables.append(variables)
        
    return list_of_variables


def _combine_train_test_metrics_into_single_df(kode: str, train_metrics: dict, test_metrics: dict) -> pd.DataFrame:
    """
    (Internal Helper) Combines training and testing metrics into a single DataFrame row.

    Args:
        kode (str): The stock emiten symbol.
        train_metrics (dict): A dictionary of performance metrics for the training set.
        test_metrics (dict): A dictionary of performance metrics for the testing set.

    Returns:
        pd.DataFrame: A single-row DataFrame containing all metrics, prefixed with
                      'Train - ' or 'Test - ', and the stock 'Kode'.
    """
    train_df = pd.DataFrame(train_metrics)
    train_df.columns = [f'Train - {col}' for col in train_df.columns]

    test_df = pd.DataFrame(test_metrics)
    test_df.columns = [f'Test - {col}' for col in test_df.columns]

    train_test_df = pd.concat((train_df, test_df), axis=1)
    train_test_df.insert(0, 'Kode', kode)

    return train_test_df

def _save_developed_model(model, label_type: str, kode: str, model_type: str):
    """
    Saves a trained model object to a file using pickle.

    The filename is standardized to include the stock emiten, model type (e.g., '10dd'),
    and the date of creation.

    Args:
        model (object): The trained model object to be saved.
        kode (str): The stock emiten symbol.
        model_type (str): A descriptor for the model type (e.g., '10dd', '15dd').
    """ 
    developed_date = datetime.now().date().strftime('%Y%m%d')
    filename = f'database/developedModels/{to_camel(label_type)}/{kode}-{model_type}-{developed_date}.pkl'
    
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    return

def _save_csv_file(data: pd.DataFrame, filename: str):
    """
    (Internal Helper) Saves a pandas dataframe by either creating new file or appending to an existing file

    Args:
        data (pd.DataFrame): The pandas dataframe that is about to be saved
        filename (str): Filepath to save the data
    """
    if os.path.exists(filename):
        data.to_csv(filename, mode='a', index=False, header=False) 
    else:
        data.to_csv(filename, index=False)

    return