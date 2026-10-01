from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def dataset_meta_features(X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Extract basic dataset meta-features."""
    n_samples, n_features = X.shape
    num_numeric = X.select_dtypes(include=["number"]).shape[1]
    num_categorical = X.select_dtypes(exclude=["number"]).shape[1]

    class_counts = y.value_counts(normalize=True)
    imbalance = float(class_counts.max() - class_counts.min()) if len(class_counts) > 1 else 0.0

    numeric = X.select_dtypes(include=["number"]).to_numpy(dtype=float)
    feature_means = np.nanmean(numeric, axis=0) if numeric.size else np.array([0.0])
    feature_std = np.nanstd(numeric, axis=0) if numeric.size else np.array([0.0])
    skewness = np.nanmean(skew(numeric, axis=0, nan_policy="omit")) if numeric.size else 0.0
    kurt = np.nanmean(kurtosis(numeric, axis=0, nan_policy="omit")) if numeric.size else 0.0

    corr_matrix = pd.DataFrame(numeric).corr().abs().values if numeric.size and numeric.shape[1] > 1 else np.array([[0.0]])
    mean_corr = float(np.nanmean(corr_matrix))
    max_corr = float(np.nanmax(corr_matrix))

    return {
        "n_samples": float(n_samples),
        "n_features": float(n_features),
        "num_numeric": float(num_numeric),
        "num_categorical": float(num_categorical),
        "class_imbalance": float(imbalance),
        "mean_feature_std": float(np.nanmean(feature_std)),
        "mean_feature_skewness": float(skewness),
        "mean_feature_kurtosis": float(kurt),
        "mean_corr": mean_corr,
        "max_corr": max_corr,
    }


def meta_feature_names() -> List[str]:
    return [
        "n_samples",
        "n_features",
        "num_numeric",
        "num_categorical",
        "class_imbalance",
        "mean_feature_std",
        "mean_feature_skewness",
        "mean_feature_kurtosis",
        "mean_corr",
        "max_corr",
    ]
