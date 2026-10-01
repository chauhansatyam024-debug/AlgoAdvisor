# AlgoAdvisor

AlgoAdvisor is a meta-learning framework for automated ML algorithm selection.

## Goal

Predict the best ML algorithm for a dataset using dataset meta-features before training all candidate models.

## Project structure

- `src/algoadvisor/`
  - `data.py` — dataset loading utilities
  - `features.py` — meta-feature extraction
  - `models.py` — base algorithm evaluation and selection
  - `meta_model.py` — meta-model training and prediction
  - `pipeline.py` — end-to-end workflow helpers
- `src/main.py` — CLI entrypoint
- `requirements.txt` — Python dependencies

## Setup

1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

This scaffold includes a CLI entrypoint and helper modules for:

- loading datasets
- extracting dataset meta-features
- training base algorithms and selecting the best one
- training a meta-model that predicts the recommended algorithm
- recommending the best algorithm for a new dataset

You can extend the pipeline by adding dataset downloads and model evaluation logic.
