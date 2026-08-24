import json
import tempfile
import unittest
from pathlib import Path

from src.dialogflow_client import DialogflowSimulatorClient
from src.ml_model import UniversityIntentClassifier


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.dataset_path = str(Path(self.temporary_directory.name) / "intents.json")
        Path(self.dataset_path).write_text(
            json.dumps(
                {
                    "intents": [
                        {
                            "tag": "greeting",
                            "patterns": ["hello there", "good morning"],
                            "responses": ["Hello"],
                        },
                        {
                            "tag": "goodbye",
                            "patterns": ["goodbye now", "see you"],
                            "responses": ["Bye"],
                        },
                        {
                            "tag": "fees",
                            "patterns": ["tuition fees", "course cost"],
                            "responses": ["Fee information"],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_ml_requires_training(self):
        classifier = UniversityIntentClassifier()
        with self.assertRaises(RuntimeError):
            classifier.predict_intent("hello", log_if_fallback=False)

    def test_ml_can_train_from_explicit_examples(self):
        classifier = UniversityIntentClassifier()
        classifier.train_from_examples(
            ["hello there", "goodbye now"],
            ["greeting", "goodbye"],
            {"greeting": ["Hello"], "goodbye": ["Bye"]},
        )
        tag, confidence = classifier.predict_intent(
            "hello there", log_if_fallback=False
        )
        self.assertEqual(tag, "greeting")
        self.assertEqual(confidence, 1.0)

    def test_dialogflow_local_baseline_uses_only_explicit_training_phrases(self):
        client = DialogflowSimulatorClient(
            self.dataset_path,
            training_patterns=["hello there", "goodbye now"],
            training_tags=["greeting", "goodbye"],
        )
        self.assertEqual(client.training_example_count, 2)
        result = client.detect_intent("tuition fees")
        self.assertEqual(
            client.normalize_intent_name(result["intent_name"]),
            "fallback",
        )

    def test_dialogflow_training_arguments_must_be_paired(self):
        with self.assertRaises(ValueError):
            DialogflowSimulatorClient(
                self.dataset_path,
                training_patterns=["hello"],
            )


if __name__ == "__main__":
    unittest.main()
