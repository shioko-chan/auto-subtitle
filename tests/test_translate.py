import unittest
from unittest.mock import patch

from subtitle_pipeline.config import LLMConfig
from subtitle_pipeline.subtitles import Cue
from subtitle_pipeline.translate import OpenAICompatibleTranslator, _parse_json_object


class TranslationTests(unittest.TestCase):
    def test_ssl_context_combines_platform_and_certifi_ca(self):
        with patch("subtitle_pipeline.translate.ssl.create_default_context") as create, patch(
            "subtitle_pipeline.translate.certifi.where", return_value="/ca/certifi.pem"
        ):
            translator = OpenAICompatibleTranslator(LLMConfig(), "secret")
        self.assertIs(translator.ssl_context, create.return_value)
        create.return_value.load_verify_locations.assert_called_once_with(
            cafile="/ca/certifi.pem"
        )

    def test_translates_in_batches_and_preserves_timing(self):
        translator = OpenAICompatibleTranslator(LLMConfig(batch_size=2), "secret")
        responses = [
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"translations":[{"id":0,"text":"甲"},{"id":1,"text":"乙"}]}'
                        }
                    }
                ]
            },
            {
                "choices": [
                    {"message": {"content": '{"translations":[{"id":0,"text":"丙"}]}'}}
                ]
            },
        ]
        cues = [Cue(0, 1, "a"), Cue(1, 2, "b"), Cue(2, 3, "c")]
        with patch.object(translator, "_request", side_effect=responses) as request:
            result = translator.translate(cues)
        self.assertEqual([cue.text for cue in result], ["甲", "乙", "丙"])
        self.assertEqual([(cue.start, cue.end) for cue in result], [(0, 1), (1, 2), (2, 3)])
        self.assertEqual(request.call_count, 2)

    def test_parses_fenced_json_from_less_strict_provider(self):
        parsed = _parse_json_object('```json\n{"translations": []}\n```')
        self.assertEqual(parsed, {"translations": []})

    def test_translates_title_and_description_together(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(metadata_description_max_chars=13), "secret"
        )
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"title":"中文标题","description":"中文简介\\nhttps://example.com"}'
                    }
                }
            ]
        }
        with patch.object(translator, "_request", return_value=response) as request:
            result = translator.translate_metadata("Original", "A very long description")
        self.assertEqual(result, ("中文标题", "中文简介\nhttps://example.com"))
        body = request.call_args.args[0]
        prompt = body["messages"][1]["content"]
        self.assertIn('"description": "A very long d"', prompt)


if __name__ == "__main__":
    unittest.main()
