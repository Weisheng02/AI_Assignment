import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ActiveLearningAccessTests(unittest.TestCase):
    def test_review_actions_do_not_require_an_environment_pin(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")

        self.assertNotIn("CHATBOT_ADMIN_PIN", app_source)
        self.assertNotIn("Administrator PIN", app_source)
        self.assertNotIn("Admin Review Actions", app_source)
        self.assertIn("Merge Query & Retrain Model", app_source)
        self.assertIn("I confirm that all logged queries may be cleared.", app_source)


if __name__ == "__main__":
    unittest.main()
