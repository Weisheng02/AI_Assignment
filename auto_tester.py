"""Human-labelled functional probes for the two deployed local engines."""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime

import pandas as pd

# Ensure project root is in sys.path.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dialogflow_client import DialogflowSimulatorClient
from src.ml_model import UniversityIntentClassifier
from src.preprocessing import load_intents


DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "intents.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "data", "latest_test_results.json")

# Expected intents were manually assigned before running the models.  A
# ``fallback`` label means the supplied dataset does not contain an answer for
# that request; returning an unrelated supported intent is counted as a failure.
TEST_CASES = [
    {
        "query": "What time does the university library close?",
        "expected_intent": "library",
    },
    {
        "query": "How much is the tuition fee for Computer Science?",
        "expected_intent": "fees",
    },
    {
        "query": "Where is the student hostel located?",
        "expected_intent": "accommodation",
    },
    {
        "query": "How do I apply for admission?",
        "expected_intent": "admission",
    },
    {
        "query": "Are there any sports or gym facilities on campus?",
        "expected_intent": "sports",
    },
    {
        "query": "What is the passing mark for exams?",
        "expected_intent": "assessment_rules",
    },
    {
        "query": "Can I pay my fees using online banking?",
        "expected_intent": "fees",
    },
    {
        "query": "What are the entry requirements for Diploma courses?",
        "expected_intent": "admission",
    },
    {
        "query": "Is there a shuttle bus service available?",
        "expected_intent": "transport",
    },
    {
        "query": "How to join student clubs and societies?",
        "expected_intent": "clubs_societies",
    },
]

# Backward-compatible alias used by earlier scripts/notebooks.
TEST_QUERIES = [case["query"] for case in TEST_CASES]


def is_streamlit_running(url="http://localhost:8501"):
    """Check whether the Streamlit web application is online."""
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "HealthCheckProbe"},
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _assert_prediction_contract(result, intent_key, confidence_key, response_key):
    """Fail immediately if an engine violates its documented result schema."""
    assert isinstance(result, dict), "prediction must be a dictionary"
    assert isinstance(result.get(intent_key), str) and result[intent_key], (
        f"missing non-empty {intent_key}"
    )
    confidence = result.get(confidence_key)
    assert isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0, (
        f"{confidence_key} must be between 0 and 1"
    )
    assert isinstance(result.get(response_key), str) and result[response_key].strip(), (
        f"missing non-empty {response_key}"
    )


def run_auto_test(save_results=True, strict=False, check_web=True):
    """Run labelled probes and return detailed pass-rate statistics.

    ``strict=True`` asserts that both engines pass every labelled case.  The
    default is diagnostic: failures are recorded and reported without hiding
    them.  ``save_results=False`` keeps unit tests read-only.
    """
    web_online = is_streamlit_running() if check_web else False
    classifier = UniversityIntentClassifier(confidence_threshold=0.20)
    classifier.train(DATASET_PATH)
    dialogflow_local = DialogflowSimulatorClient(DATASET_PATH)

    valid_intents = {
        intent["tag"] for intent in load_intents(DATASET_PATH)["intents"]
    } | {"fallback"}
    results = []

    for index, case in enumerate(TEST_CASES, 1):
        query = case["query"]
        expected_intent = case["expected_intent"]
        assert expected_intent in valid_intents, (
            f"unknown expected intent in test case: {expected_intent}"
        )

        start = time.perf_counter()
        ml_result = classifier.get_response(query, log_if_fallback=False)
        ml_latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _assert_prediction_contract(
            ml_result,
            "predicted_tag",
            "confidence",
            "response",
        )

        start = time.perf_counter()
        dialogflow_result = dialogflow_local.detect_intent(query)
        dialogflow_latency_ms = round((time.perf_counter() - start) * 1000, 2)
        _assert_prediction_contract(
            dialogflow_result,
            "intent_name",
            "confidence",
            "response",
        )
        dialogflow_intent = dialogflow_local.normalize_intent_name(
            dialogflow_result["intent_name"]
        )

        ml_intent = ml_result["predicted_tag"]
        record = {
            "id": index,
            "query": query,
            "expected_intent": expected_intent,
            "ml_predicted_intent": ml_intent,
            "ml_pass": ml_intent == expected_intent,
            "ml_confidence": round(float(ml_result["confidence"]), 4),
            "ml_latency_ms": ml_latency_ms,
            "ml_response": ml_result["response"],
            "df_predicted_intent": dialogflow_intent,
            "df_raw_intent_name": dialogflow_result["intent_name"],
            "df_pass": dialogflow_intent == expected_intent,
            "df_confidence": round(float(dialogflow_result["confidence"]), 4),
            "df_latency_ms": dialogflow_latency_ms,
            "df_response": dialogflow_result["response"],
        }
        results.append(record)

    total = len(results)
    ml_pass_count = sum(record["ml_pass"] for record in results)
    df_pass_count = sum(record["df_pass"] for record in results)
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "streamlit_online": web_online,
        "evaluation_protocol": "manually labelled expected intents",
        "total_queries_tested": total,
        "ml_pass_count": ml_pass_count,
        "ml_pass_rate": round(ml_pass_count / total, 4),
        "df_pass_count": df_pass_count,
        "df_pass_rate": round(df_pass_count / total, 4),
        "avg_ml_confidence": round(
            sum(record["ml_confidence"] for record in results) / total,
            4,
        ),
        "avg_ml_latency_ms": round(
            sum(record["ml_latency_ms"] for record in results) / total,
            2,
        ),
        "results": results,
    }

    # Structural assertions ensure that missing cases cannot silently inflate a
    # pass rate.  Strict correctness is opt-in because this diagnostic is also
    # intended to reveal current model failures.
    assert len(results) == len(TEST_CASES)
    assert ml_pass_count + sum(not record["ml_pass"] for record in results) == total
    assert df_pass_count + sum(not record["df_pass"] for record in results) == total
    if strict:
        assert ml_pass_count == total, (
            f"ML engine passed {ml_pass_count}/{total} labelled cases"
        )
        assert df_pass_count == total, (
            f"Dialogflow local baseline passed {df_pass_count}/{total} labelled cases"
        )

    if save_results:
        with open(RESULTS_PATH, "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)

    display = pd.DataFrame(
        [
            {
                "Query": record["query"],
                "Expected": record["expected_intent"],
                "ML Intent": record["ml_predicted_intent"],
                "ML Pass": record["ml_pass"],
                "Dialogflow Local Intent": record["df_predicted_intent"],
                "DF Pass": record["df_pass"],
            }
            for record in results
        ]
    )
    print(display.to_string(index=False))
    print(
        f"\nML labelled pass rate: {ml_pass_count}/{total} "
        f"({summary['ml_pass_rate'] * 100:.1f}%)"
    )
    print(
        f"Dialogflow local labelled pass rate: {df_pass_count}/{total} "
        f"({summary['df_pass_rate'] * 100:.1f}%)"
    )
    if save_results:
        print(f"Detailed labelled results saved to: {RESULTS_PATH}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with an assertion failure unless both engines pass every case",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="print results without replacing data/latest_test_results.json",
    )
    arguments = parser.parse_args()
    run_auto_test(save_results=not arguments.no_save, strict=arguments.strict)
