import unittest

from src.inference import messages_for, validate_response


class InferenceContractTests(unittest.TestCase):
    def test_prompt_has_only_frozen_system_and_user_messages(self):
        messages = messages_for("No water since Monday at Ward 4.")
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Return no reasoning, markdown, or commentary.", messages[0]["content"])

    def test_response_validation_does_not_repair_wrapped_json(self):
        valid = (
            '{"service_domain":"water_supply","issue_type":"service_outage_or_non_delivery",'
            '"location":"Ward 4","event_date_or_time":"Monday","amount_inr":null,'
            '"service_identifier":null,"urgency":"routine","missing_information":["none"],'
            '"formal_summary":"Water service has been unavailable in Ward 4 since Monday."}'
        )
        prediction, error = validate_response(valid)
        self.assertIsNotNone(prediction)
        self.assertIsNone(error)
        prediction, error = validate_response(f"```json\n{valid}\n```")
        self.assertIsNone(prediction)
        self.assertEqual(error["type"], "invalid_json")


if __name__ == "__main__":
    unittest.main()
