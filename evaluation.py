import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import shap
from typing import Dict, Any, List
from tensorflow.keras.models import Sequential

# Set up logging
logger = logging.getLogger("EvaluationLogger")
logger.setLevel(logging.INFO)


def calculate_metric(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""

    y_true = np.abs(y_true)
    y_pred = np.abs(y_pred)
    logger.info("Calculating evaluation metrics")
    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }
    logger.info(f"Calculated Metrics: {metrics}")
    return metrics


def plot_actual_vs_predicted(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "Actual vs Predicted Values"
) -> None:
    """Scatter plot comparing actual and predicted values"""
    logger.info("Creating Actual vs Predicted scatter plot")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residual Analysis"
) -> None:
    """Residual plots for error analysis."""
    logger.info("Creating Residual Analysis plots")
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Residual distribution
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title("Residual Distribution")
    ax1.set_xlabel("Residuals")

    # Residuals vs Predicted
    sns.scatterplot(x=y_pred, y=residuals, ax=ax2, alpha=0.6)
    ax2.axhline(y=0, color="r", linestyle="--")
    ax2.set_title("Residuals vs Predicted")
    ax2.set_xlabel("Predicted Values")
    ax2.set_ylabel("Residuals")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def explain_with_shap(
    model: Any, X: np.ndarray, feature_names: List[str], n_samples: int = 100
) -> None:
    """Generate SHAP explanations for model predictions."""
    logger.info(f"Generating SHAP explanations with {n_samples} samples")
    # Sample data for faster computation
    X_sample = shap.utils.sample(X, n_samples)

    # Create appropriate explainer based on model type
    try:
        if hasattr(model, "feature_importances_"):
            logger.info("Using TreeExplainer for tree-based models")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names)

        elif isinstance(model, Sequential):
            logger.info("Using DeepExplainer for neural network models")
            explainer = shap.DeepExplainer(model, X_sample)
            shap_values = explainer.shap_values(X_sample)
            shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names)
        else:
            logger.warning("SHAP explanation not supported for this model type")
            print("SHAP explanation not supported for this model type.")
    except Exception as e:
        logger.error(f"Error in SHAP explanation: {e}")
        print(f"Error in SHAP explanation: {e}")


def get_feature_importance(
    model: Any, X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]
) -> Dict[str, float]:
    """Extract feature importance scores from model."""
    logger.info("Calculating feature importance")

    if hasattr(model, "feature_importances_"):
        logger.info("Using built-in feature importances for tree-based models")
        return dict(zip(feature_names, model.feature_importances_))

    elif isinstance(model, Sequential):
        logger.info("Using permutation importance for neural network models")
        result = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        return dict(zip(feature_names, result.importances_mean))

    else:
        logger.error("Feature importance not available for this model type")
        raise NotImplementedError(
            "Feature importance not available for this model type"
        )


def generate_evaluation_report(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_shap_samples: int = 100,
) -> Dict[str, Any]:
    """Generate comprehensive evaluation report."""
    logger.info("Starting comprehensive model evaluation")

    # Calculate predictions
    logger.info("Making predictions on test data")
    y_pred = model.predict(X_test).flatten()

    # Basic metrics calculation
    logger.info("Calculating evaluation metrics")
    metrics = calculate_metric(y_test, y_pred)

    # Feature importance extraction
    try:
        logger.info("Extracting feature importance")
        importance = get_feature_importance(model, X_test, y_test, feature_names)
    except Exception as e:
        logger.error(f"Feature importance calculation error: {e}")
        print(f"Feature importance calculation error: {e}")
        importance = None

    # SHAP explanations generation
    try:
        logger.info("Generating SHAP explanations")
        explain_with_shap(model, X_test, feature_names, n_shap_samples)
    except Exception as e:
        logger.error(f"SHAP explanation error: {e}")
        print(f"SHAP explanation error: {e}")

    # Create visualizations for actual vs predicted and residuals
    logger.info("Creating evaluation visualizations")
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residuals(y_test, y_pred)

    logger.info("Evaluation report generation completed")
    return {
        "metrics": metrics,
        "feature_importance": importance,
        "model_type": type(model).__name__,
        "limitations": [
            "Assumes linear relationships between features and target",
            "Performance dependent on data quality",
            "May not capture complex nonlinear interactions",
        ],
    }
