"""Leakage-safe and reproducible evaluation for the university chatbot.

The local Dialogflow-style implementation is explicitly reported as a local
baseline.  It is not a measurement of the live Google Dialogflow agent.
Response quality is evaluated only when an independent, human-authored response
set is available; training responses are never reused as test truth.
"""

import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

try:
    from rouge_score import rouge_scorer
except ImportError:  # Tested standards-based fallback; dependency is in requirements.txt.
    rouge_scorer = None

from src.dialogflow_client import DialogflowSimulatorClient
from src.ml_model import UniversityIntentClassifier
from src.preprocessing import clean_text, extract_dataset, load_intents


DEPLOYMENT_CONFIDENCE_THRESHOLD = 0.20
RESPONSE_QUALITY_NA = (
    "N/A - no independent human-authored response reference set was provided"
)
FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), "data", "user_feedback.json")
EVALUATION_RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "data", "evaluation_results.json"
)
RESPONSE_QUALITY_TEST_PATH = os.path.join(
    os.path.dirname(__file__), "data", "response_quality_test.json"
)

SURVEY_METRICS = (
    (
        "intent_accuracy",
        "Intent Recognition Accuracy & Precision",
        "Chatbot correctly understood student questions and intent",
    ),
    (
        "response_quality",
        "Response Relevancy & Quality",
        "Chatbot answers were informative, clear, and accurate",
    ),
    (
        "ui_navigability",
        "User Interface & Navigability",
        "Streamlit GUI was easy to navigate and chat with",
    ),
    (
        "response_speed",
        "Response Speed / Low Latency",
        "Chatbot answered promptly without noticeable delay",
    ),
    (
        "overall_satisfaction",
        "Overall System Satisfaction",
        "Overall student satisfaction with the chatbot system",
    ),
)


def calculate_response_quality(
    responses_dict,
    test_intents,
    test_predictions,
    model_instance=None,
    independent_references=None,
):
    """Return N/A unless an independent response reference set exists.

    The previous implementation selected a response from ``responses_dict`` and
    then compared it with that same dictionary.  That circular calculation made
    a correct intent appear to have a high BLEU/ROUGE score even though no
    independently labelled response had been assessed.

    ``independent_references`` is reserved for a future human-authored test set.
    Until the project contains such a set, both legacy numeric return values are
    NaN.  The unused positional parameters are retained for API compatibility.
    """
    del responses_dict, test_intents, test_predictions, model_instance
    if independent_references is not None:
        raise NotImplementedError(
            "Independent response references must be evaluated with a documented "
            "human/exact-match protocol before reporting a response-quality score."
        )
    return float("nan"), float("nan")


def _metric_tokenize(text):
    """Deterministic tokenizer shared by BLEU and the ROUGE fallback."""
    return re.findall(r"\b\w+\b", str(text).lower(), flags=re.UNICODE)


def _counter_rouge1_f1(reference, candidate):
    """Standard multiset unigram precision/recall F1 fallback for ROUGE-1."""
    reference_counts = Counter(_metric_tokenize(reference))
    candidate_counts = Counter(_metric_tokenize(candidate))
    if not reference_counts or not candidate_counts:
        return 0.0
    overlap = sum((reference_counts & candidate_counts).values())
    precision = overlap / sum(candidate_counts.values())
    recall = overlap / sum(reference_counts.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_independent_response_cases(response_quality_path, intents_data):
    """Load and validate response cases against the training dataset.

    Validation fails closed if a query duplicates a cleaned training pattern, an
    expected intent is unknown, or a reference answer duplicates a stored
    training response.  This makes circular response evaluation detectable.
    """
    with open(response_quality_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("response-quality file must contain a 'cases' list")
    if not payload["cases"]:
        raise ValueError("response-quality test set must contain at least one case")

    training_patterns, training_tags, responses_dict = extract_dataset(intents_data)
    known_intents = set(training_tags)
    cleaned_training_patterns = set(training_patterns)
    cleaned_training_responses = {
        clean_text(response)
        for responses in responses_dict.values()
        for response in responses
        if clean_text(response)
    }

    validated_cases = []
    seen_ids = set()
    seen_queries = set()
    for index, raw_case in enumerate(payload["cases"], 1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"response case {index} must be a JSON object")

        case_id = str(raw_case.get("id", "")).strip()
        query = str(raw_case.get("query", "")).strip()
        expected_intent = str(raw_case.get("expected_intent", "")).strip()
        reference = str(raw_case.get("reference_answer", "")).strip()
        source_urls = raw_case.get("source_urls")

        if not case_id or case_id in seen_ids:
            raise ValueError(f"response case {index} has a missing or duplicate id")
        if not query or not reference:
            raise ValueError(f"response case {case_id} needs a query and reference")
        if expected_intent not in known_intents:
            raise ValueError(
                f"response case {case_id} uses unknown intent: {expected_intent}"
            )
        if not isinstance(source_urls, list) or not source_urls:
            raise ValueError(f"response case {case_id} needs at least one source URL")
        if not all(
            isinstance(url, str) and url.startswith(("https://", "http://"))
            for url in source_urls
        ):
            raise ValueError(f"response case {case_id} has an invalid source URL")

        cleaned_query = clean_text(query)
        cleaned_reference = clean_text(reference)
        if not cleaned_query or cleaned_query in seen_queries:
            raise ValueError(f"response case {case_id} has an empty or duplicate query")
        if cleaned_query in cleaned_training_patterns:
            raise ValueError(
                f"response case {case_id} query appears in the training patterns"
            )
        if cleaned_reference in cleaned_training_responses:
            raise ValueError(
                f"response case {case_id} reuses a training response as its reference"
            )

        seen_ids.add(case_id)
        seen_queries.add(cleaned_query)
        validated_cases.append(
            {
                "id": case_id,
                "query": query,
                "cleaned_query": cleaned_query,
                "expected_intent": expected_intent,
                "reference_answer": reference,
                "source_urls": list(source_urls),
            }
        )

    return {
        "protocol": str(payload.get("protocol", "")).strip(),
        "version": str(payload.get("version", "")).strip(),
        "cases": validated_cases,
        "validation": {
            "case_count": len(validated_cases),
            "query_training_pattern_overlap_count": 0,
            "reference_training_response_match_count": 0,
            "all_expected_intents_exist": True,
        },
    }


def _deterministic_response(predicted_intent, responses_dict):
    """Select the first stored response, never a random response."""
    responses = responses_dict.get(predicted_intent, [])
    if responses:
        return responses[0], "first response for predicted intent"
    return (
        "I'm sorry, I didn't quite understand your question. Could you please "
        "rephrase or ask about university courses, fees, admissions, or campus "
        "facilities?",
        "fixed fallback response",
    )


def _score_independent_response_records(records):
    """Compute corpus BLEU and mean ROUGE-1 F1 over independent references."""
    reference_corpus = [
        [_metric_tokenize(record["reference_answer"])] for record in records
    ]
    candidate_corpus = [_metric_tokenize(record["candidate_response"]) for record in records]
    bleu = corpus_bleu(
        reference_corpus,
        candidate_corpus,
        smoothing_function=SmoothingFunction().method1,
    )

    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        rouge_values = [
            scorer.score(
                record["reference_answer"],
                record["candidate_response"],
            )["rouge1"].fmeasure
            for record in records
        ]
        rouge_backend = "rouge-score RougeScorer rouge1 F1, use_stemmer=True"
    else:
        rouge_values = [
            _counter_rouge1_f1(
                record["reference_answer"],
                record["candidate_response"],
            )
            for record in records
        ]
        rouge_backend = (
            "Counter multiset unigram precision/recall F1 fallback "
            "(rouge-score unavailable)"
        )

    case_count = len(records)
    intent_accuracy = sum(record["intent_correct"] for record in records) / case_count
    coverage = sum(record["predicted_intent"] != "fallback" for record in records) / case_count
    return {
        "BLEU Score (g.ii)": round(float(bleu), 4),
        "ROUGE-1 Score (g.ii)": round(float(np.mean(rouge_values)), 4),
        "Intent Accuracy": round(float(intent_accuracy), 4),
        "Coverage": round(float(coverage), 4),
        "Case Count": case_count,
        "ROUGE Backend": rouge_backend,
        "Cases": records,
    }


def evaluate_independent_response_quality(
    dataset_path,
    response_quality_path=RESPONSE_QUALITY_TEST_PATH,
):
    """Evaluate deployed Member 1/2 engines on independent response cases."""
    intents_data = load_intents(dataset_path)
    validated = load_independent_response_cases(response_quality_path, intents_data)
    _, _, responses_dict = extract_dataset(intents_data)

    deployed_ml = UniversityIntentClassifier(
        confidence_threshold=DEPLOYMENT_CONFIDENCE_THRESHOLD
    )
    deployed_ml.train(dataset_path)
    deployed_dialogflow_local = DialogflowSimulatorClient(dataset_path)

    records_by_model = {
        "Member 1 (Dialogflow ES)": [],
        "Member 2 (TF-IDF + Logistic Reg)": [],
    }
    for case in validated["cases"]:
        ml_intent, ml_confidence = deployed_ml.predict_intent(
            case["query"],
            log_if_fallback=False,
        )
        ml_response, ml_policy = _deterministic_response(ml_intent, responses_dict)
        records_by_model["Member 2 (TF-IDF + Logistic Reg)"].append(
            {
                **case,
                "predicted_intent": ml_intent,
                "confidence": round(float(ml_confidence), 4),
                "intent_correct": ml_intent == case["expected_intent"],
                "candidate_response": ml_response,
                "candidate_selection": ml_policy,
            }
        )

        dialogflow_result = deployed_dialogflow_local.detect_intent(case["query"])
        dialogflow_intent = deployed_dialogflow_local.normalize_intent_name(
            dialogflow_result.get("intent_name", "")
        )
        dialogflow_response, dialogflow_policy = _deterministic_response(
            dialogflow_intent,
            responses_dict,
        )
        records_by_model["Member 1 (Dialogflow ES)"].append(
            {
                **case,
                "predicted_intent": dialogflow_intent,
                "confidence": round(float(dialogflow_result.get("confidence", 0.0)), 4),
                "intent_correct": dialogflow_intent == case["expected_intent"],
                "candidate_response": dialogflow_response,
                "candidate_selection": dialogflow_policy,
            }
        )

    metrics = {
        model: _score_independent_response_records(records)
        for model, records in records_by_model.items()
    }
    return {
        "protocol": validated["protocol"],
        "version": validated["version"],
        "test_file": os.path.relpath(response_quality_path, os.path.dirname(__file__)),
        "validation": validated["validation"],
        "candidate_policy": (
            "Models predict an intent deterministically; the candidate is the first "
            "stored response for that predicted intent, or a fixed fallback. "
            "Independent references are never selected from training responses."
        ),
        "bleu_protocol": (
            "NLTK corpus_bleu over lowercased regex word tokens with "
            "SmoothingFunction.method1"
        ),
        "models": metrics,
    }


def _prepare_evaluation_dataset(intents_data):
    """Deduplicate cleaned phrases and exclude ambiguous cross-label phrases."""
    patterns, tags, responses_dict = extract_dataset(intents_data)

    labels_by_text = defaultdict(set)
    for text, tag in zip(patterns, tags):
        labels_by_text[text].add(tag)

    ambiguous_texts = {
        text for text, labels in labels_by_text.items() if len(labels) > 1
    }
    unique_patterns = []
    unique_tags = []
    seen = set()
    for text, tag in zip(patterns, tags):
        if text in ambiguous_texts or text in seen:
            continue
        seen.add(text)
        unique_patterns.append(text)
        unique_tags.append(tag)

    same_label_duplicate_rows = sum(
        len([tag for candidate, tag in zip(patterns, tags) if candidate == text]) - 1
        for text, labels in labels_by_text.items()
        if len(labels) == 1
    )
    metadata = {
        "raw_pattern_count": len(patterns),
        "evaluation_pattern_count": len(unique_patterns),
        "same_label_duplicate_rows_removed": same_label_duplicate_rows,
        "ambiguous_cleaned_phrases_excluded": sorted(ambiguous_texts),
    }
    return unique_patterns, unique_tags, responses_dict, metadata


def _classification_metrics(y_true, predictions):
    """Compute standard sklearn metrics and a labelled confusion matrix."""
    predictions = list(predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    labels = sorted(set(y_true) | set(predictions))
    matrix = confusion_matrix(y_true, predictions, labels=labels)
    fallback_count = sum(prediction == "fallback" for prediction in predictions)
    sample_count = len(predictions)

    return {
        "Accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1-Score": round(float(f1), 4),
        "Coverage": round(float((sample_count - fallback_count) / sample_count), 4),
        "Fallback Rate": round(float(fallback_count / sample_count), 4),
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix.tolist(),
        },
    }


def _result_row(member, engine_type, metrics, response_metrics=None):
    if response_metrics is None:
        bleu = float("nan")
        rouge = float("nan")
        response_intent_accuracy = float("nan")
        response_coverage = float("nan")
        response_case_count = 0
        response_status = RESPONSE_QUALITY_NA
    else:
        bleu = response_metrics["BLEU Score (g.ii)"]
        rouge = response_metrics["ROUGE-1 Score (g.ii)"]
        response_intent_accuracy = response_metrics["Intent Accuracy"]
        response_coverage = response_metrics["Coverage"]
        response_case_count = response_metrics["Case Count"]
        response_status = (
            f"Independent reference test ({response_metrics['Case Count']} cases); "
            f"intent accuracy={response_metrics['Intent Accuracy']:.4f}; "
            f"coverage={response_metrics['Coverage']:.4f}"
        )
    return {
        "Member": member,
        "Engine Type": engine_type,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1-Score": metrics["F1-Score"],
        "Coverage": metrics["Coverage"],
        "Fallback Rate": metrics["Fallback Rate"],
        # Retain legacy columns so app.py remains compatible.  Baselines remain
        # N/A because they are not deployed response engines.
        "BLEU Score (g.ii)": bleu,
        "ROUGE-1 Score (g.ii)": rouge,
        "Response Test Intent Accuracy": response_intent_accuracy,
        "Response Test Coverage": response_coverage,
        "Response Test Cases": response_case_count,
        "Response Quality Status": response_status,
    }


def evaluate_models(
    dataset_path: str,
    response_quality_path=RESPONSE_QUALITY_TEST_PATH,
):
    """Train and evaluate all local models on one leakage-safe 80/20 split.

    Compatibility: the function still returns ``(results_df, X_test, y_test,
    ml_predictions)`` for app.py.  Detailed confusion matrices, predictions and
    split metadata are stored in ``results_df.attrs['evaluation_details']``.
    """
    intents_data = load_intents(dataset_path)
    patterns, tags, responses_dict, dataset_metadata = _prepare_evaluation_dataset(
        intents_data
    )

    X_train, X_test, y_train, y_test = train_test_split(
        patterns,
        tags,
        test_size=0.20,
        random_state=42,
        stratify=tags,
    )

    overlap = sorted(set(X_train).intersection(X_test))
    if overlap:
        raise AssertionError(f"train/test text leakage detected: {overlap}")

    print(
        f"Dataset: {len(patterns)} unique, unambiguous patterns across "
        f"{len(set(tags))} intent classes"
    )
    print(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

    # Member 2: use the exact deployed confidence threshold and fallback logic.
    ml_clf = UniversityIntentClassifier(
        confidence_threshold=DEPLOYMENT_CONFIDENCE_THRESHOLD
    )
    ml_clf.train_from_examples(X_train, y_train, responses_dict)
    ml_predictions = [
        ml_clf.predict_intent(text, log_if_fallback=False)[0] for text in X_test
    ]
    ml_metrics = _classification_metrics(y_test, ml_predictions)

    # Baseline 1: Multinomial Naive Bayes.
    nb_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1))),
            ("clf", MultinomialNB(alpha=1.0)),
        ]
    )
    nb_pipeline.fit(X_train, y_train)
    nb_predictions = list(nb_pipeline.predict(X_test))
    nb_metrics = _classification_metrics(y_test, nb_predictions)

    # Baseline 2: Linear SVM.
    svm_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1))),
            ("clf", LinearSVC(C=0.1, max_iter=1000, random_state=42)),
        ]
    )
    svm_pipeline.fit(X_train, y_train)
    svm_predictions = list(svm_pipeline.predict(X_test))
    svm_metrics = _classification_metrics(y_test, svm_predictions)

    # Member 1 local baseline: only training phrases are available to both its
    # exact-match lookup and Jaccard search.  This is not the cloud agent.
    dialogflow_local = DialogflowSimulatorClient(
        dataset_path,
        training_patterns=X_train,
        training_tags=y_train,
    )
    dialogflow_results = [dialogflow_local.detect_intent(text) for text in X_test]
    dialogflow_predictions = [
        dialogflow_local.normalize_intent_name(result.get("intent_name", ""))
        for result in dialogflow_results
    ]
    dialogflow_metrics = _classification_metrics(y_test, dialogflow_predictions)

    response_evaluation = None
    if response_quality_path and os.path.exists(response_quality_path):
        response_evaluation = evaluate_independent_response_quality(
            dataset_path,
            response_quality_path,
        )
    response_models = response_evaluation["models"] if response_evaluation else {}

    results_df = pd.DataFrame(
        [
            _result_row(
                "Member 1 (Dialogflow ES)",
                "Local train-only Dialogflow-style simulator (not cloud)",
                dialogflow_metrics,
                response_models.get("Member 1 (Dialogflow ES)"),
            ),
            _result_row(
                "Member 2 (TF-IDF + Logistic Reg)",
                f"Deployed local ML; confidence threshold={DEPLOYMENT_CONFIDENCE_THRESHOLD:.2f}",
                ml_metrics,
                response_models.get("Member 2 (TF-IDF + Logistic Reg)"),
            ),
            _result_row(
                "Baseline 1 (Multinomial Naïve Bayes)",
                "Local probabilistic baseline",
                nb_metrics,
            ),
            _result_row(
                "Baseline 2 (Linear SVM)",
                "Local support-vector baseline",
                svm_metrics,
            ),
        ]
    )

    model_metrics = {
        "Member 1 (Dialogflow ES)": dialogflow_metrics,
        "Member 2 (TF-IDF + Logistic Reg)": ml_metrics,
        "Baseline 1 (Multinomial Naïve Bayes)": nb_metrics,
        "Baseline 2 (Linear SVM)": svm_metrics,
    }
    predictions_by_model = {
        "Member 1 (Dialogflow ES)": dialogflow_predictions,
        "Member 2 (TF-IDF + Logistic Reg)": ml_predictions,
        "Baseline 1 (Multinomial Naïve Bayes)": nb_predictions,
        "Baseline 2 (Linear SVM)": svm_predictions,
    }
    results_df.attrs["evaluation_details"] = {
        "methodology": {
            **dataset_metadata,
            "split": "80/20 stratified",
            "random_state": 42,
            "training_count": len(X_train),
            "test_count": len(X_test),
            "train_test_text_overlap_count": len(overlap),
            "deployment_confidence_threshold": DEPLOYMENT_CONFIDENCE_THRESHOLD,
            "dialogflow_training_example_count": (
                dialogflow_local.training_example_count
            ),
            "dialogflow_scope": "local simulator; training phrases only; not cloud",
            "response_quality": (
                "independent reference test"
                if response_evaluation
                else RESPONSE_QUALITY_NA
            ),
        },
        "response_quality": response_evaluation or {
            "status": RESPONSE_QUALITY_NA,
            "models": {},
        },
        "confusion_matrices": {
            name: values["confusion_matrix"] for name, values in model_metrics.items()
        },
        "test_cases": [
            {
                "text": text,
                "expected_intent": expected,
                "predictions": {
                    model: predictions[index]
                    for model, predictions in predictions_by_model.items()
                },
            }
            for index, (text, expected) in enumerate(zip(X_test, y_test))
        ],
    }

    for _, row in results_df.iterrows():
        print(
            f"{row['Member']}: accuracy={row['Accuracy']:.4f}, "
            f"F1={row['F1-Score']:.4f}, coverage={row['Coverage']:.4f}, "
            f"fallback={row['Fallback Rate']:.4f}"
        )

    return results_df, X_test, y_test, ml_predictions


def _validate_rating(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer from 1 to 5")
    if not 1 <= int(value) <= 5:
        raise ValueError(f"{name} must be between 1 and 5")
    return int(value)


def save_user_feedback(
    user,
    accuracy,
    quality,
    ui,
    speed,
    satisfaction,
    comments="",
):
    """Validate and atomically append live feedback to the survey data file."""
    ratings = {
        "intent_accuracy": _validate_rating("accuracy", accuracy),
        "response_quality": _validate_rating("quality", quality),
        "ui_navigability": _validate_rating("ui", ui),
        "response_speed": _validate_rating("speed", speed),
        "overall_satisfaction": _validate_rating("satisfaction", satisfaction),
    }
    safe_user = str(user or "Anonymous Student").strip()[:100]
    safe_comments = str(comments or "").strip()[:2000]

    logs = []
    if os.path.exists(FEEDBACK_PATH):
        try:
            with open(FEEDBACK_PATH, "r", encoding="utf-8") as file:
                logs = json.load(file)
            if not isinstance(logs, list):
                raise ValueError("feedback file must contain a JSON list")
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing feedback file is unreadable; refusing to overwrite") from error

    logs.append(
        {
            "user": safe_user,
            **ratings,
            "comments": safe_comments,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    feedback_dir = os.path.dirname(FEEDBACK_PATH)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=feedback_dir,
        prefix=".feedback-",
        suffix=".json",
        delete=False,
    ) as temporary_file:
        json.dump(logs, temporary_file, indent=2)
        temporary_path = temporary_file.name
    os.replace(temporary_path, FEEDBACK_PATH)


def _survey_na_frame(reason):
    return pd.DataFrame(
        [
            {
                "Usability Metric": label,
                "Description": description,
                "Mean Rating (1-5)": float("nan"),
                "Satisfaction Rate": "N/A",
                "Respondents": 0,
                "Data Status": reason,
            }
            for _, label, description in SURVEY_METRICS
        ]
    )


def get_user_satisfaction_metrics():
    """Summarize only validated survey records; never substitute fake scores."""
    if not os.path.exists(FEEDBACK_PATH):
        return _survey_na_frame("N/A - no survey data file")

    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as file:
            logs = json.load(file)
    except (OSError, json.JSONDecodeError):
        return _survey_na_frame("N/A - survey data file is unreadable")

    if not isinstance(logs, list) or not logs:
        return _survey_na_frame("N/A - no survey responses")

    validated_logs = []
    try:
        for item in logs:
            validated_logs.append(
                {
                    key: _validate_rating(key, item[key])
                    for key, _, _ in SURVEY_METRICS
                }
            )
    except (KeyError, TypeError, ValueError):
        return _survey_na_frame("N/A - survey records failed validation")

    respondent_count = len(validated_logs)
    rows = []
    for key, label, description in SURVEY_METRICS:
        mean = round(
            sum(item[key] for item in validated_logs) / respondent_count,
            2,
        )
        rows.append(
            {
                "Usability Metric": label,
                "Description": description,
                "Mean Rating (1-5)": mean,
                "Satisfaction Rate": f"{round(mean / 5 * 100, 1)}%",
                "Respondents": respondent_count,
                "Data Status": "Observed local survey records",
            }
        )
    return pd.DataFrame(rows)


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def save_evaluation_results(results_df, output_path=EVALUATION_RESULTS_PATH):
    """Persist a standards-compliant JSON artifact with metrics and matrices."""
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "results": [_json_safe(row) for row in results_df.to_dict(orient="records")],
        "details": _json_safe(results_df.attrs.get("evaluation_details", {})),
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
    return output_path


if __name__ == "__main__":
    dataset_file = os.path.join(os.path.dirname(__file__), "data", "intents.json")
    results, _, _, _ = evaluate_models(dataset_file)
    print("\nLeakage-safe classification results")
    print(results.to_string(index=False))
    output_file = save_evaluation_results(results)
    print(f"\nDetailed confusion matrices and test predictions saved to: {output_file}")
    response_details = results.attrs["evaluation_details"]["response_quality"]
    if response_details.get("models"):
        print("Independent response-quality results")
        for model, metrics in response_details["models"].items():
            print(
                f"{model}: BLEU={metrics['BLEU Score (g.ii)']:.4f}, "
                f"ROUGE-1 F1={metrics['ROUGE-1 Score (g.ii)']:.4f}, "
                f"intent accuracy={metrics['Intent Accuracy']:.4f}, "
                f"coverage={metrics['Coverage']:.4f}, n={metrics['Case Count']}"
            )
    else:
        print(f"Response quality: {RESPONSE_QUALITY_NA}")
    print("\nObserved survey summary")
    print(get_user_satisfaction_metrics().to_string(index=False))
