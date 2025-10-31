import numpy as np
import pandas as pd
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from catboost import CatBoostClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def _split_data_to_train_val_test(data: pd.DataFrame, feature_columns: list, target_column: str) -> (np.array, np.array, np.array, np.array, PredefinedSplit):
    """
    (Internal Helper) Splits time-series data into training, validation, and testing sets

    This function implements a time-based split crucial for financial forecasting:
    - Training Set: All data preceding the test set
    - Validation Set (for Hyperparameter Tuning): The last 30 days of the training set
    - Test Set: The most recent days worth 10% of the overall training data

    Args:
        data (pd.DataFrame): The complete DataFrame containing features and the target
        feature_columns (list): A list of column names to be used as features
        target_column (str): The name of the column to be used as the target variable

    Returns:
        tuple: A tuple containing:
               - train_feature (np.array): Features for the training set
               - train_target (np.array): Target for the training set
               - test_feature (np.array): Features for the test set
               - test_target (np.array): Target for the test set
               - predefined_split_index (PredefinedSplit): An index for cross-validation
                 that designates the recent days worth 10% of the overall training data as the validation set
    """
    test_length = np.ceil(len(data) * 0.1).astype(int)
    test_data = data.tail(test_length)
    train_length = len(data) - test_length
    train_data = data.head(train_length)
    
    train_feature = train_data[feature_columns].values
    train_target = train_data[target_column].values
    test_feature = test_data[feature_columns].values
    test_target = test_data[target_column].values
    
    split_index = np.full(len(train_feature), -1, dtype=int)
    split_index[-test_length:] = 0
    predefined_split_index = PredefinedSplit(test_fold=split_index)
    
    return train_feature, train_target, test_feature, test_target, predefined_split_index

def _initializes_fit_tune_catboost_with_bayesian_optimization(train_feature: np.array, train_target: np.array, predefined_split_index: PredefinedSplit) -> any:
    """
    (Internal Helper) Initializes, fits, and tunes a CatBoost Classifier using Bayesian Optimization

    This function uses BayesSearchCV to efficiently search for the optimal
    hyperparameters for a CatBoost model. It validates performance using a
    predefined time-series split and fits the best-found model on the entire
    training dataset

    Args:
        train_feature (np.array): The feature set for training
        train_target (np.array): The target variable for training
        predefined_split_index (PredefinedSplit): The cross-validation strategy

    Returns:
        CatBoostClassifier: The best-performing model found by the search
    """
    model = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        logging_level='Silent'
    )

    search_spaces = {
        'depth': Integer(1, 5),
        'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
        'iterations': Integer(150, 300),
        'l2_leaf_reg': Real(0.5, 3.0)
    }
    
    scoring_method = 'roc_auc'
    if len(np.unique(train_target[-30:])) == 1:
        scoring_method = 'accuracy'
    
    hyper_tune_search = BayesSearchCV(
        estimator=model,
        search_spaces=search_spaces,
        n_iter=25,
        cv=predefined_split_index,
        scoring=scoring_method,
        n_jobs=-1,
        verbose=0
    )

    hyper_tune_search.fit(train_feature, train_target)
    best_model = hyper_tune_search.best_estimator_

    return best_model

def measure_model_performance(model, feature: np.array, target: np.array, positive_label: str, negative_label: str) -> dict:
    """
    (Internal Helper) Measures and reports the performance of the model on a given dataset

    Args:
        model: The trained classifier model
        feature (np.array): The feature set (e.g., train_feature or test_feature)
        target (np.array): The corresponding true target labels
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label

    Returns:
        dict: A dictionary containing all calculated performance metrics.
    """
    target_pred = model.predict(feature)
    target_pred_proba = model.predict_proba(feature)

    accuracy, prec_positive, prec_negative, rec_positive, rec_negative = _calculate_classification_metrics(target, target_pred, positive_label, negative_label)
    gini = _calculate_gini(model, target, target_pred_proba, positive_label)

    all_metrics = {
        'Accuracy': [accuracy],
        f'Precision {positive_label}': [prec_positive],
        f'Precision {negative_label}': [prec_negative],
        f'Recall {positive_label}': [rec_positive],
        f'Recall {negative_label}': [rec_negative],
        'Gini': [gini]
    }
    
    return all_metrics

def _measure_model_performance(model, feature: np.array, target: np.array, positive_label: str, negative_label: str) -> dict:
    """
    Measures and reports the performance of the model on a given dataset

    Args:
        model: The trained classifier model
        feature (np.array): The feature set (e.g., train_feature or test_feature)
        target (np.array): The corresponding true target labels
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label

    Returns:
        dict: A dictionary containing all calculated performance metrics.
    """
    target_pred = model.predict(feature)
    target_pred_proba = model.predict_proba(feature)

    accuracy, prec_positive, prec_negative, rec_positive, rec_negative = _calculate_classification_metrics(target, target_pred, positive_label, negative_label)
    gini = _calculate_gini(model, target, target_pred_proba, positive_label)

    all_metrics = {
        'Accuracy': [accuracy],
        f'Precision {positive_label}': [prec_positive],
        f'Precision {negative_label}': [prec_negative],
        f'Recall {positive_label}': [rec_positive],
        f'Recall {negative_label}': [rec_negative],
        'Gini': [gini]
    }
    
    return all_metrics

def _calculate_classification_metrics(target_true: np.array, target_pred: np.array, positive_label: str, negative_label: str) -> (np.array, np.array, np.array, np.array):
    """
    (Internal Helper) Calculates key classification metrics for a binary prediction task

    Args:
        target_true (np.array): The ground truth labels
        target_pred (np.array): The predicted labels from the model
        positive_label (str): The positive class of the predicted label
        negative_label (str): The negative class of the predicted label

    Returns:
        tuple: A tuple containing accuracy, precision for both classes, and recall for both classes
    """
    accuracy = accuracy_score(target_true, target_pred)
    precision_positive = precision_score(target_true, target_pred, pos_label=positive_label, zero_division=0)
    precision_negative = precision_score(target_true, target_pred, pos_label=negative_label, zero_division=0)
    recall_positive = recall_score(target_true, target_pred, pos_label=positive_label, zero_division=0)
    recall_negative = recall_score(target_true, target_pred, pos_label=negative_label, zero_division=0)

    return accuracy, precision_negative, precision_negative, recall_positive, recall_negative

def _calculate_gini(model: any, target_true: np.array, target_pred_proba: np.array, positive_label: str) -> float:
    """
    (Internal Helper) Calculates the Gini coefficient from the model's prediction probabilities

    The Gini coefficient is a common metric for evaluating binary classification
    models and is derived from the Area Under the ROC Curve (AUC)
    Formula: Gini = 2 * AUC - 1

    Args:
        model (any): The trianed catboost model for binary classifications
        target_true (np.array): The true labels of the target variable
        target_pred_proba (np.array): The predicted probabilities for each class
        positive_label (str): The positive class of the predicted label

    Returns:
        float: The calculated Gini coefficient, or 0.0 if AUC cannot be calculated
    """
    try:
        positive_class_true = (target_true == positive_label).astype(int)
        
        positive_class_index = np.where(model.classes_ == positive_label)[0][0]
        positive_class_prob = target_pred_proba[:, positive_class_index]
        auc = roc_auc_score(positive_class_true, positive_class_prob)
        gini = 2 * auc - 1
    except (ValueError, IndexError):
        gini = 0.0

    return gini