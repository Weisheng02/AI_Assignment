import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DialogflowParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.intent_data = json.loads(
            (PROJECT_ROOT / "data" / "intents.json").read_text(encoding="utf-8")
        )
        cls.entity_data = json.loads(
            (PROJECT_ROOT / "data" / "entities.json").read_text(encoding="utf-8")
        )

    def test_library_parameter_resolves_to_official_opening_hours(self):
        directory = next(
            entity
            for entity in self.entity_data["entities"]
            if entity["name"] == "campus_service_directory"
        )
        library = next(
            entry for entry in directory["entries"] if "library" in entry["synonyms"]
        )
        self.assertEqual(
            library["value"],
            "https://library.tarc.edu.my/about-us/opening-hours",
        )

    def test_required_service_uses_parameter_aware_static_response(self):
        intent = next(
            item
            for item in self.intent_data["intents"]
            if item["tag"] == "campus_service_lookup"
        )
        service = next(
            parameter
            for parameter in intent["parameters"]
            if parameter["name"] == "service"
        )
        self.assertTrue(service["required"])
        self.assertTrue(service["prompts"])
        self.assertEqual(service["entity"], "@campus_service_directory")
        self.assertIn("$service.original", intent["responses"][0])
        self.assertIn("$service", intent["responses"][0])


if __name__ == "__main__":
    unittest.main()
