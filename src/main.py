import argparse
import json
from pathlib import Path
from typing import Dict, List

from algoadvisor.pipeline import AlgoAdvisorPipeline
from algoadvisor.data import list_dataset_files, load_csv_dataset
from algoadvisor.features import dataset_meta_features
from algoadvisor.models import summarize_base_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlgoAdvisor CLI")
    parser.add_argument("--datasets", type=str, help="Path to a dataset folder or CSV file")
    parser.add_argument("--target", type=str, default=None, help="Target column name")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON file")
    return parser.parse_args()


def collect_dataset_paths(path: str) -> List[str]:
    input_path = Path(path)
    if input_path.is_dir():
        return list(list_dataset_files(str(input_path)).values())
    if input_path.is_file():
        return [str(input_path)]
    raise FileNotFoundError(f"Dataset path not found: {path}")


def main() -> None:
    args = parse_args()
    pipeline = AlgoAdvisorPipeline()
    dataset_paths = collect_dataset_paths(args.datasets)

    results: List[Dict[str, object]] = []
    for dataset_path in dataset_paths:
        X, y, _ = load_csv_dataset(dataset_path, args.target)
        base_result = pipeline.run_pipeline_on_dataset(dataset_path, args.target)
        results.append(
            {
                "dataset_path": dataset_path,
                "best_algorithm": base_result["best_algorithm"],
                "base_scores": base_result["base_scores"],
                "meta_features": base_result["meta_features"],
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} dataset(s). Results saved to {args.output}")


if __name__ == "__main__":
    main()
