import numpy as np


def MSE(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error

    Args:
        y_pred (np.ndarray): predicted labels
        y (np.ndarray): true labels

    Returns:
        float: mse
    """
    return np.mean((y - y_pred) ** 2)


def d_MSE(y_pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (y_pred - y) * 2 / np.size(y)
