import unittest
from unittest.mock import patch

from subtitle_pipeline.config import LLMConfig
from subtitle_pipeline.subtitles import Cue
from subtitle_pipeline.translate import OpenAICompatibleTranslator, _parse_json_object


class TranslationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
