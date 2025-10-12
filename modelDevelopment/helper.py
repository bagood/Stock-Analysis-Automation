import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

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
        positive_class_true = (target_true.T[0] == positive_label).astype(int)
        
        positive_class_index = np.where(model.classes_ == positive_label)[0][0]
        positive_class_prob = target_pred_proba[:, positive_class_index]
        
        auc = roc_auc_score(positive_class_true, positive_class_prob)
        gini = 2 * auc - 1
    except (ValueError, IndexError):
        logging.warning("Could not calculate Gini coefficient (likely only one class in target). Returning 0.0")
        gini = 0.0

    return gini