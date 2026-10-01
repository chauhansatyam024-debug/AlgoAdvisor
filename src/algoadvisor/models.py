from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


BASE_ALGORITHMS: Dict[str, ClassifierMixin] = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "knn": KNeighborsClassifier(),
    "svm": SVC(probability=True),
}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = X.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []
    if numeric_columns:
        transformers.append(("num", StandardScaler(), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers, remainder="drop")


def evaluate_base_algorithms(
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 3,
) -> Tuple[str, Dict[str, float]]:
    """Evaluate candidate base algorithms and return the best one by mean accuracy."""
    results: Dict[str, float] = {}
    preprocessor = build_preprocessor(X)

    for name, estimator in BASE_ALGORITHMS.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        results[name] = float(np.mean(scores))

    best_algorithm = max(results, key=results.get)
    return best_algorithm, results


def summarize_base_results(results: Dict[str, float]) -> str:
    return ", ".join([f"{name}: {score:.3f}" for name, score in results.items()])
