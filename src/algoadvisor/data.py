import os
from typing import Dict, Optional, Tuple

import pandas as pd


def load_csv_dataset(path: str, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load a dataset from a CSV file.

    Returns the feature frame, target series, and original dataframe.
    """
    df = pd.read_csv(path)

    if target_column is None:
        target_column = df.columns[-1]

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    y = df[target_column]
    X = df.drop(columns=[target_column])
    return X, y, df


def list_dataset_files(directory: str, extensions=None) -> Dict[str, str]:
    """Return a map of dataset file names to absolute paths."""
    if extensions is None:
        extensions = {".csv"}

    result = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                result[file] = os.path.join(root, file)
    return result
