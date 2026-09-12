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

    def test_metadata_uses_one_request_with_matching_terms_and_characters(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), TranslationConfig(), '')
        title = '【千石ユノ / バンドリ】3回目'
        context = {'terms': {'バンドリ': 'BanG Dream!', '無関係': '无关'},
                   'characters': [{'source_name': '千石ユノ', 'canonical': '千石由乃'}],
                   'video': {'description': '宣传' * 10000}}
        with patch.object(translator, '_request', return_value=response(
                {'title': '千石由乃第3回', 'tags': ['千石由乃']})) as send:
            result = translator.translate_metadata(title, translation_context=context)
        send.assert_called_once()
        body = send.call_args.args[0]
        value = json.loads(body['messages'][1]['content'].split('INPUT:\n', 1)[1])
        self.assertEqual(value, {'title': title, 'terms': {
            '千石ユノ': '千石由乃', 'バンドリ': 'BanG Dream!'}})
        self.assertEqual(result, '千石由乃第3回')

    def test_oversized_single_lyric_fails_before_any_generation(self):
        translator = OpenAICompatibleTranslator(LLMConfig(), TranslationConfig(), '')
        with patch.object(translator, 'validate_request', side_effect=PromptBudgetExceeded('fixed content')), patch.object(translator, '_request') as send:
            with self.assertRaises(PromptBudgetExceeded):
                translator.translate_lyrics('song', 'artist', ['原文'])
            send.assert_not_called()
