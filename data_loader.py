import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Tuple
import logging
import sys


# Configure logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (default: logging.INFO)
        log_file: Optional file path to save logs
    """
    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a logger
    logger = logging.getLogger("DataProcessingLogger")
    logger.setLevel(log_level)

    # Clear any existing handlers to prevent duplicate logs
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Initialize logger
logger = setup_logging()


def load_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the dataset from a CSV file."""
    try:
        logger.info(f"Loading data from: {filepath}")
        data = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")

        # Separate numeric and non-numeric columns
        numeric_cols = data.select_dtypes(include=np.number).columns

        # Fill missing values only in numeric columns
        for col in numeric_cols:
            data[col].fillna(data[col].mean(), inplace=True)
            logger.debug(f"Filled missing values in {col} with mean")

        # For non-numeric columns it needed to filled with mode or a default value
        non_numeric_cols = data.select_dtypes(exclude=np.number).columns
        for col in non_numeric_cols:
            if col != data.columns[0]:  # Skip the ID column (assuming it's the first)
                data[col].fillna(
                    data[col].mode()[0] if not data[col].mode().empty else "",
                    inplace=True,
                )
                logger.debug(f"Filled missing values in {col} with mode")

        # Define features and target variable
        X = data.iloc[:, 1:-1]  # Assuming first column is ID, last is target
        y = data["vomitoxin_ppb"]

        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y

    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {e}")
        raise


# -------------------------
# Exploratory Data Analysis (EDA)
# -------------------------


def generate_summary_statistics(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Generate Summary Statistice for features and target"""

    summary = X.describe().T  # T is for transpose swaps rows and columns.
    summary["target_corr"] = X.apply(lambda col: col.corr(y))
    return summary


def plot_boxplots(data: pd.DataFrame, title: str = "Boxplot of Features") -> None:
    """Generate boxplots for numerical features to identify outliers."""
    plt.figure(figsize=(15, 5))
    data.iloc[:, 1:-1].boxplot()
    plt.title(title)
    plt.show()


def plot_spectral_reflectance(X: pd.DataFrame) -> None:
    """Line plot of average reflectance across wavelengths."""
    avg_reflectance = X.mean(axis=0)

    plt.figure(figsize=(15, 6))
    plt.plot(avg_reflectance, marker="o", linestyle="-", color="b")
    plt.title("Average Reflectance Over Wavelengths")
    plt.xlabel("Wavelength Bands")
    plt.ylabel("Reflectance")

    # Adjust x-axis ticks to show only some of the labels
    plt.xticks(range(0, len(avg_reflectance), 10), rotation=45)  # Show every 10th tick

    plt.grid(True)  # Adding grid for better readability
    plt.show()


def plot_sample_heatmap(X: pd.DataFrame) -> None:
    # Plot heatmaps for sample comparisons.

    # Converting array to DataFrame because it is in np array form
    X_df = pd.DataFrame(X)

    plt.figure(figsize=(10, 5))
    corr_matrix = X_df.corr()  # Compute correlation matrix
    sns.heatmap(corr_matrix, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()


# -------------------------
# Preprocessing
# -------------------------


def handle_missing_data(
    X: pd.DataFrame, strategy: "mean", threshold: float = 0.5
) -> pd.DataFrame:
    # decide whether to drop a feature (column) based on the percentage of missing values.
    """Handle missing data using imputation or removal."""
    if strategy == "remove":
        # Remove rows with >threshold% missing values
        X_clean = X.dropna(thresh=int(X.shape[1] * threshold))
    else:
        # Impute missing values
        if strategy == "mean":
            X_clean = X.fillna(X.mean())
        elif strategy == "median":
            X_clean = X.fillna(X.median())
        elif strategy == "mode":
            X_clean = X.fillna(X.mode().iloc[0])

    return X_clean


def detect_outlier_zscore(X: pd.DataFrame, threshold: float = 3):
    """Detect Outlier using zscore"""
    # z_score calculation
    z_score = np.abs((X - X.mean()) / X.std())

    # flags rows where feature has zscore greater than 3std dev
    outliers = (z_score > threshold).any(axis=1)

    # count and get indices of outliers
    outliers_count = outliers.sum()
    outliers_indices = outliers[outliers].index.tolist()

    return outliers_count, outliers_indices


import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def detect_outliers_multiple_methods(X: pd.DataFrame, method="all", threshold=3):
    """
    Detect outliers using three methods:
    1. Z-Score Method
    2. Interquartile Range (IQR) Method
    3. Isolation Forest

    Parameters:
    -----------
    X : pd.DataFrame
        Input dataframe
    method : str, optional (default='all')
        Specific method to use or 'all' to return results from all methods
    threshold : float, optional (default=3)
        Threshold for outlier detection

    Returns:
    --------
    dict or pd.Index
        Indices of detected outliers or dictionary of results
    """
    # Ensure input is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    results = {}

    # 1. Z-Score Method
    def z_score_outliers(data, threshold=3):
        z_scores = np.abs(stats.zscore(data))
        return np.where(np.any(z_scores > threshold, axis=1))[0]

    z_score_indices = z_score_outliers(X_scaled, threshold)
    results["z_score"] = X.index[z_score_indices]

    # 2. Interquartile Range (IQR) Method
    def iqr_outliers(data, threshold=1.5):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers_mask = ((data < lower_bound) | (data > upper_bound)).any(axis=1)
        return np.where(outliers_mask)[0]

    iqr_indices = iqr_outliers(X_scaled)
    results["iqr"] = X.index[iqr_indices]

    # 3. Isolation Forest
    def isolation_forest_outliers(data, contamination=0.1):
        clf = IsolationForest(contamination=contamination, random_state=42)
        y_pred = clf.fit_predict(data)
        return np.where(y_pred == -1)[0]

    iso_forest_indices = isolation_forest_outliers(X_scaled)
    results["isolation_forest"] = X.index[iso_forest_indices]

    # Print summary
    print("Outlier Detection Summary:")
    for method_name, indices in results.items():
        print(f"{method_name.replace('_', ' ').title()}: {len(indices)} outliers")

    # Return based on method parameter
    if method == "all":
        return results
    elif method in results:
        return results[method]
    else:
        raise ValueError(
            f"Invalid method: {method}. Choose 'all' or one of: {list(results.keys())}"
        )


def remove_outliers(X, method="consensus", threshold=3):
    """Remove outliers from dataset using specified method"""
    try:
        logger.info(f"Removing outliers using {method} method")
        # If X is a tuple of (X, y), separate them
        if isinstance(X, tuple) and len(X) == 2:
            X, y = X
            outlier_indices = detect_outliers_multiple_methods(X, method, threshold)
            logger.info(f"Number of outliers detected: {len(outlier_indices)}")
            X_cleaned = X.drop(index=outlier_indices)
            y_cleaned = y.drop(index=outlier_indices)
            return X_cleaned, y_cleaned

        # If X is a single dataframe
        outlier_indices = detect_outliers_multiple_methods(X, method, threshold)
        logger.info(f"Number of outliers detected: {len(outlier_indices)}")
        return X.drop(index=outlier_indices)
    except Exception as e:
        logger.error(f"Error removing outliers: {e}")
        raise


# Add logging to other key functions, for example:
def preprocess_data(X: pd.DataFrame) -> np.ndarray:
    """Preprocess the features by standardizing them."""
    try:
        logger.info("Starting data preprocessing")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.info("Data preprocessing completed")
        return X_scaled
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise


# -------------------------
# Advanced Data Quality Checks
# -------------------------


def detect_sensor_drift(X: pd.DataFrame) -> pd.DataFrame:
    """Checking sensor drift by statistics computation"""
    return X.describe().loc[["mean", "std"]]


def detect_sensor_drift_PCA(X: pd.DataFrame, y: pd.Series, n_components=2):
    """Perform PCA to detect sensor drift and visualize it"""
    # first we have to scale the feature to bring in similiar scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Applying PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)

    # PCA plot
    plt.figure(figsize=(10, 5))
    plt.scatter(pca_result[:, 0], pca_result[0:, 1], c=y, cmap="plasma")
    plt.xlabel("principal component 1")
    plt.ylabel("principal component 2")
    plt.title("PCA for sensor Drift Detection")
    plt.show()


def create_spectral_indices(X: pd.DataFrame) -> pd.DataFrame:
    """additional spectral indices(Ex: ratios of bands)"""
    # Example: Normalized Difference Index between two arbitrary bands
    band1, band2 = X.columns[10], X.columns[20]
    X["ndi"] = (X[band1] - X[band2]) / (X[band1] + X[band2])

    return X


# -------------------------
# Data Splitting
# -------------------------


def split_data(
    X: np.ndarray, y: pd.Series, test_size: float = 0.2, **kwargs
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, **kwargs)
