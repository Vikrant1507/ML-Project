import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
from typing import Dict, Any

# -------------------------
# Evaluation Metrics
# -------------------------


def calculate_metric(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics.

    input:
    y_true: True target values
    y_pred: Predicted target values

    Returns:
    Dict: Dictionary of evaluation metrics"""

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


# -------------------------
# Visual Evaluation
# -------------------------


def plot_actual_vs_predicted(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "ACtual vs Predicted Values"
) -> None:
    """Scatter plot comparing actual and predicted values"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("predicted Values")
    plt.title(title)
    plt.grid(True)
    plt.show()
