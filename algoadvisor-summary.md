# AlgoAdvisor — Project Context Summary

## Background
B.Tech AIML student, 3rd semester, working toward becoming an AI/VR game developer. Project suggestions were explored across several rounds — starting broad (DS/algo-based, stats-based, beginner ML), narrowing into pure ML fundamentals (regression, classification, clustering from scratch), then advanced/unique ML topics.

## Final Decision
**Building "AlgoAdvisor" — a meta-learning framework for automated ML algorithm selection.**

## Core Idea
Instead of manually trying multiple ML algorithms (KNN, logistic regression, decision tree, SVM, etc.) on a new dataset to see which works best, AlgoAdvisor trains a **meta-model** that predicts the best-performing algorithm for a given dataset *before* running them all.

## How It Works
1. Collect ~30-50 varied public datasets (Kaggle/UCI)
2. Extract meta-features from each (number of samples, number of features, class imbalance ratio, skewness/kurtosis, correlation stats, etc.)
3. Actually train 4-5 base algorithms on each dataset and record which performs best (this becomes the label)
4. Train a meta-model that maps meta-features → best algorithm
5. On a new/unseen dataset: extract its meta-features, predict the best algorithm, **then automatically train that recommended algorithm and return the trained model + performance metrics** (this is the full end-to-end flow, not just a recommendation)

## Evaluation Plan
Compare AlgoAdvisor's top pick's accuracy vs. the average accuracy across all algorithms vs. the actual best algorithm's accuracy — measure how often AlgoAdvisor's recommendation matches or lands in the top 2.

## Why This Idea Over Alternatives
More novel/"systems-thinking" than other advanced ML ideas considered (e.g., feature importance stability analysis, conformal prediction, federated learning simulation). Mirrors real-world AutoML systems (like auto-sklearn) rather than being a typical "train one model, report accuracy" project.

## Naming
- **Project name:** AlgoAdvisor
- **Subtitle:** "AlgoAdvisor: A Meta-Learning Framework for Automated ML Algorithm Selection"

## Next Steps Needed
- Exact meta-features to extract
- Which base algorithms to include
- Week-by-week build plan
