import unittest

from src.preprocessing import analyze_sentiment, clean_text


class PreprocessingTests(unittest.TestCase):
    def test_digits_and_course_codes_are_preserved(self):
        cleaned = clean_text("Intake 2026 for course A123 costs RM5000")
        self.assertIn("2026", cleaned.split())
        self.assertIn("a123", cleaned.split())
        self.assertIn("rm5000", cleaned.split())

    def test_help_is_not_negative(self):
        result = analyze_sentiment("Can you help me with admission?")
        self.assertEqual(result["sentiment"], "Neutral Inquiry")

    def test_explicit_urgent_language_remains_negative(self):
        result = analyze_sentiment("This is an urgent problem")
        self.assertEqual(result["sentiment"], "Negative / Urgent")


if __name__ == "__main__":
    unittest.main()
