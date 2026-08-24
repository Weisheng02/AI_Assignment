import unittest

from src.dialogflow_webhook import (
    CONTACT_URL,
    INTAKE_URL,
    PROGRAMME_URL,
    build_fulfillment_response,
)


class DialogflowWebhookTests(unittest.TestCase):
    def test_programme_lookup_uses_parameters_and_official_sources(self):
        result = build_fulfillment_response({
            "queryResult": {
                "action": "programme.lookup",
                "parameters": {"programme": "computer science", "intake": "June"},
            }
        })
        self.assertIn("computer science", result["fulfillmentText"])
        self.assertIn("June", result["fulfillmentText"])
        self.assertIn(PROGRAMME_URL, result["fulfillmentText"])
        self.assertIn(INTAKE_URL, result["fulfillmentText"])

    def test_campus_lookup_uses_official_contact_directory(self):
        result = build_fulfillment_response({
            "queryResult": {
                "action": "campus.service.lookup",
                "parameters": {"service": "library", "contact_channel": "email"},
            }
        })
        self.assertIn("library", result["fulfillmentText"])
        self.assertIn("email", result["fulfillmentText"])
        self.assertIn(CONTACT_URL, result["fulfillmentText"])

    def test_unknown_action_is_refused(self):
        result = build_fulfillment_response({"queryResult": {"action": "unsafe.action"}})
        self.assertIn("cannot perform", result["fulfillmentText"])

    def test_missing_query_result_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "queryResult"):
            build_fulfillment_response({})


if __name__ == "__main__":
    unittest.main()
