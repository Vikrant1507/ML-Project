import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Tuple


def load_data(filepath: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the dataset from a CSV file."""
    data = pd.read_csv(filepath)

    # Separate numeric and non-numeric columns
    numeric_cols = data.select_dtypes(include=np.number).columns

    # Fill missing values only in numeric columns
    for col in numeric_cols:
        data[col].fillna(data[col].mean(), inplace=True)

    # For non-numeric columns it needed to filled with mode or a default value
    # filling with mode (most common value)
    non_numeric_cols = data.select_dtypes(
        exclude=np.number
    ).columns  # np.number includes integers and floats
    for col in non_numeric_cols:
        if col != data.columns[0]:  # Skip the ID column (assuming it's the first)
            data[col].fillna(
                data[col].mode()[0] if not data[col].mode().empty else "", inplace=True
            )

    # Define features and target variable
    X = data.iloc[:, 1:-1]  # Assuming first column is ID, last is target
    y = data["vomitoxin_ppb"]

    return X, y


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


def remove_outliers(X: pd.DataFrame, y: pd.Series, threshold: float = 3):
    """removeing outliers using detect_outlier_zscore function."""
    outliers_count, outliers_indices = detect_outlier_zscore(X, threshold)

    # Remove outliers from X and y
    X_cleaned = X.drop(index=outliers_indices)
    y_cleaned = y.drop(index=outliers_indices)

    print(f"Removed {outliers_count} outliers.")

    return X_cleaned, y_cleaned


def preprocess_data(X: pd.DataFrame) -> np.ndarray:
    """Preprocess the features by standardizing them."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


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
    X: np.ndarray, y: pd.Series, test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Split the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=42)
