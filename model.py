from typing import Tuple, List, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam  # type: ignore
from xgboost import XGBRegressor
import optuna

# -------------------------
# Base Model Architectures
# -------------------------


def build_simple_nn(
    input_shape: int,
    n_units: int = 256,  # Increased neurons
    learning_rate: float = 0.0005,  # Reduced learning rate
    dropout_rate: float = 0.2,  # Less dropout
) -> Sequential:
    """Build a Simple Regression neural Network"""
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

    return model


def build_xgboost(n_estimators: int = 100, learning_rate: float = 0.1) -> XGBRegressor:
    """
    Build an XGBoost regression model.

    Input:
    n_estimators (int): Number of boosting rounds
    learning_rate (float): Boosting learning rate

    Returns:
    XGBRegressor: Configured XGBoost model
    """

    return XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        objective="reg:squarederror",
    )


# -------------------------
# Training & Evaluation
# -------------------------


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
    if isinstance(model, Sequential):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    else:
        model.fit(X_train, y_train)

    return model


def cross_validate(
    model: any, X: np.ndarray, y: np.ndarray, n_splits: int = 5
) -> Dict[str, float]:
    """Perform k-fold cross-validation."""
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_absolute_error

    kf = KFold(n_splits=n_splits)
    mae_scores = []

    # Convert y to NumPy array to avoid KeyError
    y = np.array(y)

    for train_idx, val_idx in kf.split(X):  # Add X as parameter here
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = train_model(model, X_train, y_train)
        y_pred = model.predict(X_val)
        mae_scores.append(mean_absolute_error(y_val, y_pred))

    return {"mean_mae": np.mean(mae_scores), "std_mae": np.std(mae_scores)}


# -------------------------
# Hyperparameter Optimization
# -------------------------


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

    def objective(trial):
        params = {
            "n_units": trial.suggest_int("n_units", 32, 256),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        }
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
        _, mae = model.evaluate(X_val, y_val)  # , verbose=0)
        return mae

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    return study.best_params
