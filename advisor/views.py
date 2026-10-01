import json
from pathlib import Path

from django.conf import settings
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render

from algoadvisor.data import load_csv_dataset
from algoadvisor.features import dataset_meta_features
from algoadvisor.models import evaluate_base_algorithms, summarize_base_results


def index(request: HttpRequest) -> HttpResponse:
    return render(request, "advisor/index.html", {})


def upload_dataset(request: HttpRequest) -> HttpResponse:
    result = None
    error = None
    if request.method == "POST" and request.FILES.get("dataset"):
        dataset_file = request.FILES["dataset"]
        upload_path = Path(settings.BASE_DIR) / "uploaded_dataset.csv"
        with open(upload_path, "wb") as f:
            for chunk in dataset_file.chunks():
                f.write(chunk)

        try:
            X, y, _ = load_csv_dataset(str(upload_path), request.POST.get("target_name") or None)
            meta_features = dataset_meta_features(X, y)
            best_algorithm, base_scores = evaluate_base_algorithms(X, y)
            result = {
                "meta_features": meta_features,
                "best_algorithm": best_algorithm,
                "base_scores": base_scores,
            }
        except Exception as exc:
            error = str(exc)

    return render(request, "advisor/results.html", {"result": result, "error": error})
