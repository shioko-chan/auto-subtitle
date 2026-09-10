import unittest
from pathlib import Path

from subtitle_pipeline.response_schemas import response_format
from subtitle_pipeline.translate import _prepare_api_request
from subtitle_pipeline.config import LLMConfig


class ResponseSchemaTests(unittest.TestCase):
    def test_every_prompt_has_a_closed_strict_schema(self):
        prompts = Path(__file__).resolve().parents[1] / 'src/subtitle_pipeline/prompts'
        names = list(prompts.glob('*.md'))
        self.assertEqual(len(names), 8)

        def check(schema):
            if schema.get('type') == 'object':
                self.assertFalse(schema['additionalProperties'])
                self.assertEqual(set(schema['required']), set(schema['properties']))
                for child in schema['properties'].values():
                    check(child)
            if schema.get('type') == 'array':
                check(schema['items'])

        for name in names:
            with self.subTest(prompt=name.name):
                fmt = response_format(name.name)
                self.assertTrue(fmt['json_schema']['strict'])
                check(fmt['json_schema']['schema'])

    def test_api_formats_preserve_the_same_contract(self):
        fmt = response_format('select-ocr-song-titles.md')
        body = {'messages': [], 'max_tokens': 128, 'response_format': fmt}
        _, chat = _prepare_api_request(LLMConfig(), body)
        _, responses = _prepare_api_request(LLMConfig(api_style='responses'), body)
        self.assertEqual(chat['response_format'], fmt)
        self.assertEqual(responses['text']['format'], {'type': 'json_schema', **fmt['json_schema']})
        self.assertNotIn('json_schema', responses['text']['format'])

    def test_unknown_prompt_fails_and_schema_mutations_are_isolated(self):
        with self.assertRaises(KeyError):
            response_format('unknown.md')
        fmt = response_format('asr-correct.md')
        fmt['json_schema']['schema']['properties'].clear()
        self.assertIn('segments', response_format('asr-correct.md')['json_schema']['schema']['properties'])
