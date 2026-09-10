import unittest

from subtitle_pipeline.llm_response import (
    finish_reason,
    structured_request_body,
    structured_response_content,
)


class StructuredLLMResponseTests(unittest.TestCase):
    def test_builds_prompt_request_with_shared_model_options(self) -> None:
        body = structured_request_body(
            model="model",
            prompt_name="segment-translate-cues.md",
            prompt="user prompt",
            max_tokens=123,
            temperature=0.1,

            thinking="enabled",
        )

        self.assertEqual(body["model"], "model")
        self.assertEqual(body["max_tokens"], 123)
        self.assertEqual(body["messages"][1]["content"], "user prompt")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual(body["thinking"], {"type": "enabled"})

    def test_extracts_only_complete_structured_responses(self) -> None:
        response = {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"ok":true}'}}
            ]
        }

        self.assertEqual(
            structured_response_content(response, finish_reason=finish_reason),
            '{"ok":true}',
        )

        response["choices"][0]["finish_reason"] = "length"
        with self.assertRaisesRegex(RuntimeError, "finish_reason=length"):
            structured_response_content(response, finish_reason=finish_reason)


if __name__ == "__main__":
    unittest.main()
