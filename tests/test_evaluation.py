import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluate import (
    RESPONSE_QUALITY_NA,
    RESPONSE_QUALITY_TEST_PATH,
    _counter_rouge1_f1,
    calculate_response_quality,
    evaluate_independent_response_quality,
    evaluate_models,
    get_user_satisfaction_metrics,
    load_independent_response_cases,
    save_evaluation_results,
)
from src.preprocessing import clean_text, extract_dataset, load_intents


DATASET_PATH = str(Path(__file__).resolve().parents[1] / "data" / "intents.json")


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.x_test, cls.y_test, cls.ml_predictions = evaluate_models(
            DATASET_PATH
        )
        cls.details = cls.results.attrs["evaluation_details"]
        cls.response_case_count = cls.details["response_quality"]["validation"][
            "case_count"
        ]

    def test_split_has_no_text_overlap(self):
        methodology = self.details["methodology"]
        self.assertEqual(methodology["train_test_text_overlap_count"], 0)
        self.assertEqual(
            methodology["dialogflow_training_example_count"],
            methodology["training_count"],
        )
        self.assertEqual(len(self.x_test), methodology["test_count"])

    def test_dialogflow_result_is_not_the_old_leaked_score(self):
        row = self.results[self.results["Member"].str.contains("Member 1")].iloc[0]
        self.assertIn("not cloud", row["Engine Type"])
        self.assertLess(row["Accuracy"], 0.95)

    def test_deployment_threshold_reports_coverage_and_fallback(self):
        row = self.results[self.results["Member"].str.contains("Member 2")].iloc[0]
        self.assertGreater(row["Fallback Rate"], 0.0)
        self.assertAlmostEqual(row["Coverage"] + row["Fallback Rate"], 1.0, places=4)
        self.assertIn("threshold=0.20", row["Engine Type"])

    def test_response_quality_uses_only_independent_references(self):
        member_rows = self.results[self.results["Member"].str.contains("Member")]
        baseline_rows = self.results[self.results["Member"].str.contains("Baseline")]
        self.assertTrue(member_rows["BLEU Score (g.ii)"].notna().all())
        self.assertTrue(member_rows["ROUGE-1 Score (g.ii)"].notna().all())
        self.assertTrue(baseline_rows["BLEU Score (g.ii)"].isna().all())
        self.assertTrue(baseline_rows["ROUGE-1 Score (g.ii)"].isna().all())
        self.assertEqual(
            set(member_rows["Response Test Cases"]),
            {self.response_case_count},
        )
        self.assertEqual(
            set(baseline_rows["Response Quality Status"]),
            {RESPONSE_QUALITY_NA},
        )
        bleu, rouge = calculate_response_quality({}, None, [])
        self.assertTrue(math.isnan(bleu))
        self.assertTrue(math.isnan(rouge))

    def test_response_case_validation_proves_no_circular_truth(self):
        response_details = self.details["response_quality"]
        self.assertEqual(
            response_details["validation"]["query_training_pattern_overlap_count"],
            0,
        )
        self.assertEqual(
            response_details["validation"]["reference_training_response_match_count"],
            0,
        )
        intents_data = load_intents(DATASET_PATH)
        training_patterns, _, responses = extract_dataset(intents_data)
        training_response_texts = {
            clean_text(response)
            for values in responses.values()
            for response in values
        }
        for model_metrics in response_details["models"].values():
            self.assertEqual(model_metrics["Case Count"], self.response_case_count)
            for case in model_metrics["Cases"]:
                self.assertNotIn(case["cleaned_query"], set(training_patterns))
                self.assertNotIn(
                    clean_text(case["reference_answer"]),
                    training_response_texts,
                )
                self.assertNotEqual(
                    clean_text(case["candidate_response"]),
                    clean_text(case["reference_answer"]),
                )
                self.assertTrue(case["source_urls"])

    def test_validator_rejects_a_training_response_as_reference(self):
        intents_data = load_intents(DATASET_PATH)
        first_intent = intents_data["intents"][0]
        circular_payload = {
            "protocol": "negative test",
            "version": "test",
            "cases": [
                {
                    "id": "CIRCULAR",
                    "query": "A deliberately unique circular-reference probe 987654",
                    "expected_intent": first_intent["tag"],
                    "reference_answer": first_intent["responses"][0],
                    "source_urls": ["https://example.edu/source"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "circular.json"
            path.write_text(json.dumps(circular_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reuses a training response"):
                load_independent_response_cases(str(path), intents_data)

    def test_validator_rejects_a_training_query(self):
        intents_data = load_intents(DATASET_PATH)
        first_intent = intents_data["intents"][0]
        overlap_payload = {
            "protocol": "negative test",
            "version": "test",
            "cases": [
                {
                    "id": "OVERLAP",
                    "query": first_intent["patterns"][0],
                    "expected_intent": first_intent["tag"],
                    "reference_answer": "A deliberately independent reference answer.",
                    "source_urls": ["https://example.edu/source"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.json"
            path.write_text(json.dumps(overlap_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training patterns"):
                load_independent_response_cases(str(path), intents_data)

    def test_response_evaluation_is_reproducible(self):
        first = evaluate_independent_response_quality(
            DATASET_PATH,
            RESPONSE_QUALITY_TEST_PATH,
        )
        second = evaluate_independent_response_quality(
            DATASET_PATH,
            RESPONSE_QUALITY_TEST_PATH,
        )
        self.assertEqual(first, second)

    def test_counter_rouge_fallback_is_standard_unigram_f1(self):
        self.assertEqual(_counter_rouge1_f1("a b", "a b"), 1.0)
        self.assertAlmostEqual(_counter_rouge1_f1("a b", "a c"), 0.5)
        self.assertEqual(_counter_rouge1_f1("a", "z"), 0.0)

    def test_confusion_matrices_are_labelled_and_square(self):
        matrices = self.details["confusion_matrices"]
        self.assertEqual(len(matrices), 4)
        for value in matrices.values():
            labels = value["labels"]
            matrix = value["matrix"]
            self.assertEqual(len(matrix), len(labels))
            self.assertTrue(all(len(row) == len(labels) for row in matrix))
            self.assertEqual(sum(sum(row) for row in matrix), len(self.y_test))

    def test_missing_survey_returns_na_instead_of_hardcoded_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = str(Path(directory) / "missing-feedback.json")
            with patch("evaluate.FEEDBACK_PATH", missing_path):
                survey = get_user_satisfaction_metrics()
        self.assertTrue(survey["Mean Rating (1-5)"].isna().all())
        self.assertEqual(set(survey["Satisfaction Rate"]), {"N/A"})
        self.assertEqual(set(survey["Respondents"]), {0})

    def test_saved_json_uses_null_not_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = str(Path(directory) / "evaluation.json")
            save_evaluation_results(self.results, output_path)
            payload = json.loads(Path(output_path).read_text(encoding="utf-8"))
        self.assertIsInstance(payload["results"][0]["BLEU Score (g.ii)"], float)
        self.assertIsNone(payload["results"][2]["BLEU Score (g.ii)"])
        self.assertIn("confusion_matrices", payload["details"])
        self.assertEqual(
            payload["details"]["response_quality"]["validation"]["case_count"],
            self.response_case_count,
        )


if __name__ == "__main__":
    unittest.main()
