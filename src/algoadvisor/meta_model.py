from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .features import meta_feature_names


@dataclass
class MetaModelResult:
    model: Pipeline
    feature_names: List[str]
    label_mapping: Dict[str, int]
    reverse_mapping: Dict[int, str]


class MetaModel:
    def __init__(self) -> None:
        self.feature_names = meta_feature_names()
        self.label_mapping: Dict[str, int] = {}
        self.reverse_mapping: Dict[int, str] = {}
        self.model: Optional[Pipeline] = None

    def fit(self, X_meta: pd.DataFrame, y_labels: pd.Series) -> None:
        labels = sorted(y_labels.unique())
        self.label_mapping = {label: idx for idx, label in enumerate(labels)}
        self.reverse_mapping = {idx: label for label, idx in self.label_mapping.items()}

        y_encoded = y_labels.map(self.label_mapping)
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )
        self.model.fit(X_meta[self.feature_names], y_encoded)

    def predict(self, X_meta: pd.DataFrame) -> List[str]:
        if self.model is None:
            raise ValueError("MetaModel is not trained yet")
        prediction = self.model.predict(X_meta[self.feature_names])
        return [self.reverse_mapping[int(value)] for value in prediction]

    def score(self, X_meta: pd.DataFrame, y_labels: pd.Series) -> float:
        if self.model is None:
            raise ValueError("MetaModel is not trained yet")
        y_encoded = y_labels.map(self.label_mapping)
        return float(self.model.score(X_meta[self.feature_names], y_encoded))
