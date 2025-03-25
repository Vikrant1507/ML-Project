from typing import Tuple, List, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
)  # Added BatchNormalization import
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.regularizers import l2  # Added for regularization
from xgboost import XGBRegressor
import optuna
import random
import logging

# Set up logging
logger = logging.getLogger("ModelLogger")
logger.setLevel(logging.INFO)


# Uncomment and modify set_seeds if needed
def set_seeds(seed: int = 42):
    """
    Set seeds for reproducibility across multiple libraries.

    Args:
        seed (int): Seed value for random number generators
    """
    logger.info(f"Setting random seeds with seed value: {seed}")
    np.random.seed(seed)
    tf.random.set_seed(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_simple_nn(
    input_shape: int,
    n_units: int = 256,  # Increased neurons
    learning_rate: float = 0.0005,  # Reduced learning rate
    dropout_rate: float = 0.2,  # Less dropout
) -> Sequential:
    """Build a Simple Regression neural Network"""
    try:
        # Log model configuration
        logger.info("Building Neural Network")
        logger.info(
            f"Model Configuration: input_shape={input_shape}, n_units={n_units}, "
            f"learning_rate={learning_rate}, dropout_rate={dropout_rate}"
        )

        model = Sequential(
            [
                Dense(n_units, activation="relu", input_shape=(input_shape,)),
                Dropout(dropout_rate),
                Dense(128, activation="relu"),  # Added more neurons
                Dropout(0.2),
                Dense(64, activation="relu"),
                Dense(1),
            ]
        )

        # Compile the model with the specified learning rate
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])

        logger.info("Neural Network model built and compiled successfully")
        return model

    except Exception as e:
        logger.error(f"Error building neural network: {e}", exc_info=True)
        raise


def build_xgboost(n_estimators: int = 100, learning_rate: float = 0.1) -> XGBRegressor:
    """
    Build an XGBoost regression model.

    Input:
    n_estimators (int): Number of boosting rounds
    learning_rate (float): Boosting learning rate

    Returns:
    XGBRegressor: Configured XGBoost model
    """
    try:
        logger.info("Building XGBoost Regression Model")
        logger.info(
            f"Model Configuration: n_estimators={n_estimators}, learning_rate={learning_rate}"
        )

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective="reg:squarederror",
        )

        logger.info("XGBoost model configured successfully")
        return model

    except Exception as e:
        logger.error(f"Error building XGBoost model: {e}", exc_info=True)
        raise


def train_model(
    model: any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
) -> Any:
    """
    Generic model training function.

    Input:
    model: Model object with scikit-learn or Keras interface
    X_train (np.ndarray): Training features
    y_train (np.ndarray): Training labels
    epochs (int): Training epochs (for neural networks)
    batch_size (int): Batch size (for neural networks)

    Returns: Trained model
    """
    try:
        logger.info("Starting model training")
        logger.info(
            f"Training Configuration: "
            f"Model Type={type(model).__name__}, "
            f"Epochs={epochs}, Batch Size={batch_size}"
        )

        if isinstance(model, Sequential):
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
        else:
            model.fit(X_train, y_train)

        logger.info("Model training completed successfully")
        return model

    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        raise


def cross_validate(
    model: any, X: np.ndarray, y: np.ndarray, n_splits: int = 5
) -> Dict[str, float]:
    """Perform k-fold cross-validation."""
    try:
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_absolute_error

        logger.info(f"Performing {n_splits}-fold Cross-Validation")

        kf = KFold(n_splits=n_splits)
        mae_scores = []

        # Convert y to NumPy array to avoid KeyError
        y = np.array(y)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            logger.info(f"Cross-validation fold {fold}/{n_splits}")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model_copy = type(model)() if not isinstance(model, Sequential) else model
            trained_model = train_model(model_copy, X_train, y_train)
            y_pred = trained_model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            mae_scores.append(mae)
            logger.info(f"Fold {fold} MAE: {mae}")

        results = {"mean_mae": np.mean(mae_scores), "std_mae": np.std(mae_scores)}
        logger.info(f"Cross-Validation Results: {results}")
        return results

    except Exception as e:
        logger.error(f"Error during cross-validation: {e}", exc_info=True)
        raise


import logging
import numpy as np
import tensorflow as tf
import optuna
from typing import Dict, Any

# Ensure logger is configured
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def optimize_nn_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """
    Optimize neural network hyperparameters using Optuna.
    Args:
    X_train (np.ndarray): Training features
    y_train (np.ndarray): Training labels
    X_val (np.ndarray): Validation features
    y_val (np.ndarray): Validation labels
    n_trials (int): Number of optimization trials
    Returns:
    Dict: Best hyperparameters
    """
    # Log the start of hyperparameter optimization
    logger.info(f"Starting Hyperparameter Optimization")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Validation data shape: {X_val.shape}")

    def objective(trial):
        try:
            # Define hyperparameter search space
            params = {
                "n_units": trial.suggest_int("n_units", 32, 256),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2),
                "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            }

            # Log current trial parameters
            logger.info(f"Trial {trial.number} parameters: {params}")

            # Build and train the model
            model = build_simple_nn(X_train.shape[1], **params)

            # Early stopping to prevent overfitting
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )

            model.fit(
                X_train,
                y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0,
            )

            # Evaluate using Mean Absolute Error
            _, mae = model.evaluate(X_val, y_val, verbose=0)

            logger.info(f"Trial {trial.number} MAE: {mae}")
            return mae

        except Exception as e:
            logger.error(f"Error in trial {trial.number}: {e}", exc_info=True)
            return float("inf")

    try:
        # Create and run the Optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        # Log the best trial results
        logger.info("Best trial completed")
        logger.info(f"Best MAE value: {study.best_trial.value}")
        logger.info("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"  {key}: {value}")

        return study.best_params

    except Exception as e:
        logger.error(f"Error during hyperparameter optimization: {e}", exc_info=True)
        raise
