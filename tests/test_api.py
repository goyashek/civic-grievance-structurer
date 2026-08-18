import json
import unittest

from pydantic import ValidationError

from src import api
from src.inference import InferenceResult


class FakeInference:
    def __init__(self, result):
        self.result = result

    def infer(self, complaint):
        return self.result


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.previous_inference = api._inference

    def tearDown(self):
        api._inference = self.previous_inference

    def test_health_and_model_info_are_metadata_only(self):
        self.assertEqual(api.health(), {"status": "ok"})
        info = api.model_info()
        self.assertEqual(info["model_version"], "civicstruct-qlora:1")
        self.assertEqual(info["schema_version"], "1.0")
        self.assertEqual(info["evaluator_version"], "2.0")

    def test_structure_uses_shared_result(self):
        api._inference = FakeInference(
            InferenceResult(True, {"formal_summary": "ok"}, "raw", None, 0.1)
        )
        result = api.structure(api.StructureRequest(complaint="A complaint"))
        self.assertEqual(result.ok, True)
        self.assertEqual(result.raw_response, "raw")

    def test_invalid_model_output_is_visible_as_422(self):
        api._inference = FakeInference(
            InferenceResult(
                False,
                None,
                "not json",
                {"type": "invalid_json", "message": "bad JSON"},
                0.1,
            )
        )
        result = api.structure(api.StructureRequest(complaint="A complaint"))
        self.assertEqual(result.status_code, 422)
        self.assertEqual(json.loads(result.body)["error"]["type"], "invalid_json")

    def test_request_rejects_blank_and_oversized_text(self):
        with self.assertRaises(ValidationError):
            api.StructureRequest(complaint="   ")
        with self.assertRaises(ValidationError):
            api.StructureRequest(complaint="x" * (api.MAX_COMPLAINT_CHARS + 1))


if __name__ == "__main__":
    unittest.main()
