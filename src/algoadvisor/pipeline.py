from typing import Dict, List, Tuple

import pandas as pd

from .data import load_csv_dataset
from .features import dataset_meta_features
from .meta_model import MetaModel
from .models import evaluate_base_algorithms


class AlgoAdvisorPipeline:
    def __init__(self) -> None:
        self.meta_model = MetaModel()

    def build_training_example(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], str]:
        meta = dataset_meta_features(X, y)
        best_algorithm, _ = evaluate_base_algorithms(X, y)
        return meta, best_algorithm

    def fit_meta_model(self, training_data: List[Dict[str, float]], labels: List[str]) -> None:
        X_meta = pd.DataFrame(training_data)
        y_labels = pd.Series(labels)
        self.meta_model.fit(X_meta, y_labels)

    def recommend_algorithm(self, X: pd.DataFrame, y: pd.Series) -> str:
        meta = dataset_meta_features(X, y)
        prediction = self.meta_model.predict(pd.DataFrame([meta]))
        return prediction[0]

    def score_meta_model(self, test_data: List[Dict[str, float]], test_labels: List[str]) -> float:
        X_meta = pd.DataFrame(test_data)
        y_labels = pd.Series(test_labels)
        return self.meta_model.score(X_meta, y_labels)

    def run_pipeline_on_dataset(self, path: str, target_column: str = None) -> Dict[str, object]:
        X, y, df = load_csv_dataset(path, target_column)
        best_algorithm, base_scores = evaluate_base_algorithms(X, y)
        return {
            "dataset_path": path,
            "best_algorithm": best_algorithm,
            "base_scores": base_scores,
            "meta_features": dataset_meta_features(X, y),
        }
