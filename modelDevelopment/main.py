import logging
import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from catboost import CatBoostClassifier
from sklearn.model_selection import PredefinedSplit

from modelDevelopment.helper import _split_data_to_train_val_test, _initializes_fit_tune_catboost_with_bayesian_optimization, _measure_model_performance
from technicalIndicators.helper import get_all_technical_indicators

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def develop_model(prepared_data: pd.DataFrame, target_column: str, positive_label: str, negative_label: str) -> (any, dict, dict):
    """
    Main orchestration function for the entire model development process

    This function loads the feature names, splits the data, tunes the model,
    and evaluates its final performance

    Args:
        prepared_data (pd.DataFrame): The fully prepared data from the previous step.
        target_column (str): The name of the target variable column
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label
        
    Returns:
        tuple: A tuple containing:
               - model (CatBoostClassifier): The final, trained model
               - train_metrics (dict): Performance metrics on the training set
               - test_metrics (dict): Performance metrics on the testing set
    """
    logging.info(f"----- Starting Model Development for Target: '{target_column}' -----")

    logging.info("Loading stock's technical indicators as features")
    feature_columns = get_all_technical_indicators()

    logging.info("Splitting data into training, validation, testing, and forecast sets")
    train_feature, train_target, test_feature, test_target, cv_split = _split_data_to_train_val_test(prepared_data, feature_columns, target_column)

    logging.info("Initializing CatBoost model and starting hyperparameter tuning with BayesSearchCV")
    model = _initializes_fit_tune_catboost_with_bayesian_optimization(train_feature, train_target, cv_split)

    logging.info("Measuring model performance for training data")
    train_metrics = _measure_model_performance(model, train_feature, train_target, positive_label, negative_label)
    logging.info(f"Training set performance summary - Gini: {train_metrics['Gini'][0]:.4f}")

    logging.info("Measuring model performance for testing data")
    test_metrics = _measure_model_performance(model, test_feature, test_target, positive_label, negative_label)
    logging.info(f"Testing set performance summary - Gini: {test_metrics['Gini'][0]:.4f}")

    logging.info(f"----- Model Development for '{target_column}' Finished Successfully -----")
    
    return model, train_metrics, test_metrics