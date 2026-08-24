import unittest

from auto_tester import TEST_CASES, run_auto_test


class AutoTesterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = run_auto_test(
            save_results=False,
            strict=False,
            check_web=False,
        )

    def test_every_case_has_an_independent_expected_intent(self):
        self.assertEqual(self.summary["total_queries_tested"], len(TEST_CASES))
        self.assertTrue(
            all(case.get("expected_intent") for case in TEST_CASES)
        )

    def test_pass_rates_match_recorded_outcomes(self):
        results = self.summary["results"]
        expected_ml_rate = sum(item["ml_pass"] for item in results) / len(results)
        expected_df_rate = sum(item["df_pass"] for item in results) / len(results)
        self.assertAlmostEqual(self.summary["ml_pass_rate"], expected_ml_rate, places=4)
        self.assertAlmostEqual(self.summary["df_pass_rate"], expected_df_rate, places=4)

    def test_diagnostic_reports_observed_outcomes_without_hardcoded_rate(self):
        self.assertGreaterEqual(self.summary["ml_pass_rate"], 0.0)
        self.assertLessEqual(self.summary["ml_pass_rate"], 1.0)
        self.assertGreaterEqual(self.summary["df_pass_rate"], 0.0)
        self.assertLessEqual(self.summary["df_pass_rate"], 1.0)
        for item in self.summary["results"]:
            self.assertEqual(
                item["ml_pass"],
                item["ml_predicted_intent"] == item["expected_intent"],
            )
            self.assertEqual(
                item["df_pass"],
                item["df_predicted_intent"] == item["expected_intent"],
            )


if __name__ == "__main__":
    unittest.main()
