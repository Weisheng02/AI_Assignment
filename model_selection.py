"""Reproduce the Member 2 feature/model selection evidence.

Three TF-IDF representations are compared with the same balanced Logistic
Regression family under three-fold stratified cross-validation. The deployed
0.20 confidence gate is applied to out-of-fold probabilities, so coverage and
fallback effects are measured without training-fold leakage.
"""

import json
import os
import tempfile

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline

from evaluate import _prepare_evaluation_dataset
from src.preprocessing import load_intents


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "intents.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "model_selection_results.json")
CONFIDENCE_THRESHOLD = 0.20


def _classifier(vectorizer, c_value):
    return Pipeline(
        [
            ("tfidf", vectorizer),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def _metrics(y_true, predictions, coverage):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "weighted_precision": round(float(precision), 4),
        "weighted_recall": round(float(recall), 4),
        "weighted_f1": round(float(f1), 4),
        "coverage": round(float(coverage), 4),
        "fallback_rate": round(float(1.0 - coverage), 4),
    }


def run_model_selection(output_path=OUTPUT_PATH):
    patterns, tags, _, metadata = _prepare_evaluation_dataset(
        load_intents(DATASET_PATH)
    )
    x_values = np.asarray(patterns, dtype=object)
    y_values = np.asarray(tags, dtype=object)
    class_labels = np.asarray(sorted(set(tags)), dtype=object)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    candidates = {
        "word_tfidf_1_2_lr_c10": _classifier(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
            10.0,
        ),
        "char_wb_tfidf_3_5_lr_c30": _classifier(
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
            ),
            30.0,
        ),
        "word_char_union_lr_c30": _classifier(
            FeatureUnion(
                [
                    (
                        "word",
                        TfidfVectorizer(
                            ngram_range=(1, 2),
                            sublinear_tf=True,
                        ),
                    ),
                    (
                        "char",
                        TfidfVectorizer(
                            analyzer="char_wb",
                            ngram_range=(3, 5),
                            sublinear_tf=True,
                        ),
                    ),
                ]
            ),
            30.0,
        ),
    }

    results = []
    for name, candidate in candidates.items():
        probabilities = cross_val_predict(
            candidate,
            x_values,
            y_values,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )
        max_indices = probabilities.argmax(axis=1)
        max_probabilities = probabilities.max(axis=1)
        raw_predictions = class_labels[max_indices]
        gated_predictions = np.where(
            max_probabilities < CONFIDENCE_THRESHOLD,
            "fallback",
            raw_predictions,
        )
        coverage = float(np.mean(max_probabilities >= CONFIDENCE_THRESHOLD))
        raw_metrics = _metrics(y_values, raw_predictions, 1.0)
        gated_metrics = _metrics(y_values, gated_predictions, coverage)
        results.append(
            {
                "candidate": name,
                "raw_metrics": raw_metrics,
                "deployment_threshold_metrics": gated_metrics,
            }
        )

    winner = max(
        results,
        key=lambda item: item["deployment_threshold_metrics"]["weighted_f1"],
    )["candidate"]
    payload = {
        "protocol": "3-fold stratified out-of-fold predictions",
        "random_state": 42,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "dataset": metadata,
        "candidate_count": len(results),
        "selection_metric": "deployment-threshold weighted F1",
        "selected_candidate": winner,
        "results": results,
    }

    output_directory = os.path.dirname(output_path)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_directory,
        prefix=".model-selection-",
        suffix=".json",
        delete=False,
    ) as temporary:
        json.dump(payload, temporary, indent=2)
        temporary_path = temporary.name
    os.replace(temporary_path, output_path)
    return payload


if __name__ == "__main__":
    result = run_model_selection()
    print(f"Selected: {result['selected_candidate']}")
    for row in result["results"]:
        metrics = row["deployment_threshold_metrics"]
        print(
            f"{row['candidate']}: accuracy={metrics['accuracy']:.4f}, "
            f"F1={metrics['weighted_f1']:.4f}, coverage={metrics['coverage']:.4f}"
        )
    print(f"Saved: {OUTPUT_PATH}")
