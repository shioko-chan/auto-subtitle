import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from subtitle_pipeline.cache import CacheStore
from subtitle_pipeline.config import LLMConfig, TranslationConfig
from subtitle_pipeline.prompt_budget import PromptBudgetExceeded, chunk_text
from subtitle_pipeline.translate import OpenAICompatibleTranslator


def response(value):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value, ensure_ascii=False)}}]}


class BudgetedTranslationTests(unittest.TestCase):
    def test_text_chunking_preserves_every_character_and_checks_fixed_overhead(self):
        text = '一二三\n四五六\n七八九十'
        def render(value):
            return {'text': '固定' + value}
        def validate(body):
            if len(body['text']) > 7:
                raise PromptBudgetExceeded('limit')
        chunks = chunk_text(text, render_request=render, validate_request=validate)
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(len('固定' + value) <= 7 for value in chunks))
        with self.assertRaises(PromptBudgetExceeded):
            chunk_text('正文', render_request=lambda text: {'text': '固定规则过长无法容纳' + text}, validate_request=validate)

    def test_lyrics_are_planned_before_generation_and_keep_global_line_ids(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), TranslationConfig(), '')
        def ids(body):
            return [int(value) for value in re.findall(r'<(\d+)>', body['messages'][-1]['content'])]
        events = []
        def validate(body):
            events.append('validate')
            if len(ids(body)) > 2:
                raise PromptBudgetExceeded('limit')
        def request(body):
            events.append('generate')
            return response({'lines': [{'line_id': i, 'text': f'译文{i}'} for i in ids(body)]})
        with patch.object(translator, 'validate_request', side_effect=validate), patch.object(translator, '_request', side_effect=request) as send:
            result, _ = translator.translate_lyrics('song', 'artist', ['歌詞'] * 5)
        self.assertEqual(result, {i: f'译文{i}' for i in range(5)})
        self.assertEqual(send.call_count, 3)
        self.assertNotIn('validate', events[events.index('generate'):])

    def test_metadata_chunks_preserve_description_and_cache_completed_pieces(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), TranslationConfig(), '')
        sources = []
        def source(body):
            return json.loads(body['messages'][-1]['content'].split('INPUT:\n', 1)[1])
        def validate(body):
            if len(json.dumps(source(body), ensure_ascii=False)) > 360:
                raise PromptBudgetExceeded('limit')
        def request(body):
            validate(body)
            value = source(body)
            sources.append(value)
            return response({'title': '标题', 'description': value.get('description', ''),
                             'content_summary': '摘要', 'tags': ['标签']})
        description = ''.join(f'正文{index}\n' for index in range(200))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.sqlite3'
            CacheStore(path).stage('metadata', lambda: {})
            with patch.object(translator, 'validate_request', side_effect=validate), patch.object(translator, '_request', side_effect=request):
                first = translator.translate_metadata('title', description, subtitle_evidence=''.join(f'证据{i}\n' for i in range(100)), cache_path=path)
                count = len(sources)
                second = translator.translate_metadata('title', description, subtitle_evidence=''.join(f'证据{i}\n' for i in range(100)), cache_path=path)
        self.assertEqual(first, second)
        self.assertEqual(count, len(sources))
        self.assertGreater(count, 2)
        self.assertEqual(''.join(value.get('description', '') for value in sources if value.get('partial_input')), description)
        self.assertEqual(''.join(value.get('subtitle_evidence', '') for value in sources if value.get('partial_input')), ''.join(f'证据{i}\n' for i in range(100)))
        self.assertEqual(first[0], '标题')
        self.assertEqual(first[1].replace('\n', ''), description.replace('\n', ''))

    def test_oversized_single_lyric_fails_before_any_generation(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), TranslationConfig(), '')
        with patch.object(translator, 'validate_request', side_effect=PromptBudgetExceeded('fixed content')), patch.object(translator, '_request') as send:
            with self.assertRaises(PromptBudgetExceeded):
                translator.translate_lyrics('song', 'artist', ['原文'])
            send.assert_not_called()
