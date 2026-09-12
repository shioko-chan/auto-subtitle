import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from subtitle_pipeline.cache import CachedProviderMismatchError
from subtitle_pipeline.config import LLMConfig, SegmentationConfig, TranslationConfig
from subtitle_pipeline.local_segmentation import LocalUnit, SpeakerTrack
from subtitle_pipeline.song_identification import SongIdentificationResult, split_aligned_song_cues
from subtitle_pipeline.staged_translation import run_joint_translation, _validate_joint_response
from subtitle_pipeline.subtitles import Cue, TimedTextUnit, text_display_width


def response(*values):
    return {"choices": [{"message": {"content": json.dumps({"cues": list(values)})}}]}


def translated(start, end, text="译文"):
    return {"start_id": start, "end_id": end, "text": text}


class StagedTranslationTests(unittest.TestCase):
    def run_joint(self, request, cues=None, **kwargs):
        cues = cues if cues is not None else [Cue(0, 1, "こんにちは", "A"), Cue(1, 2, "世界", "A")]
        by_speaker = {}
        for index, cue in enumerate(cues):
            track = cue.speaker or "unknown"
            values = by_speaker.setdefault(track, [])
            values.append(LocalUnit(track, len(values), (index,), cue.start, cue.end,
                                    cue.text, cue.speaker, cue.kind,
                                    preferred_translation=cue.preferred_translation))
        args = dict(tracks=[SpeakerTrack(key, key, tuple(values)) for key, values in by_speaker.items()],
                    source_cues=cues, segmentation=SegmentationConfig(),
                    llm=LLMConfig(max_concurrency=1, max_retries=2), translation=TranslationConfig(),
                    request=request, parse_content=lambda value: json.loads(value)["cues"],
                    finish_reason=lambda _: "stop", retry_delay=lambda *_: None,
                    is_nontransient=lambda _: False, log_invalid_response=lambda *_: None,
                    local_translate=Mock(return_value="机器翻译"), translation_context={},
                    honorific_rules="", maximum_units=20, cache_path=None)
        args.update(kwargs)
        return run_joint_translation(**args)

    def test_one_request_selects_boundaries_and_translation(self):
        request = Mock(return_value=response(translated(0, 1, "你好世界")))
        source, output = self.run_joint(request)
        request.assert_called_once()
        self.assertEqual([(c.start, c.end, c.text) for c in output], [(0, 2, "你好世界")])
        self.assertEqual(source[0].text, "こんにちは世界")
        self.assertEqual(output[0].source_text, source[0].text)
        self.assertEqual(len(source[0].source_units), 2)
        self.assertEqual(request.call_args.args[0]["response_format"]["json_schema"]["name"], "segment_translate_cues")

    def test_coverage_and_width_are_business_constraints(self):
        invalid = [[translated(1, 1)], [translated(0, 0)],
                   [translated(0, 1), translated(1, 1)],
                   [translated(False, 1)], [translated(0, 2)],
                   [translated(0, 1, "")], [translated(0, 1, "字" * 41)]]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                _validate_joint_response(values, 2, 20)

    def test_width_failure_retries_with_more_boundaries(self):
        request = Mock(side_effect=[response(translated(0, 1, "字" * 41)),
                                    response(translated(0, 0), translated(1, 1))])
        source, output = self.run_joint(request)
        self.assertEqual(len(output), 2)
        self.assertEqual(request.call_count, 2)
        self.assertIn("PREVIOUS_RESPONSE_ERROR", request.call_args.args[0]["messages"][-1]["content"])

    def test_invalid_window_splits_after_retries(self):
        request = Mock(side_effect=[response(), response(), response(translated(0, 0, "一")), response(translated(0, 0, "二"))])
        source, output = self.run_joint(request)
        self.assertEqual([c.text for c in output], ["一", "二"])
        self.assertEqual([(c.start, c.end) for c in source], [(0, 1), (1, 2)])

    def test_atomic_translation_that_cannot_fit_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "two-line"):
            self.run_joint(Mock(return_value=response(translated(0, 0, "字" * 41))),
                           cues=[Cue(0, 1, "原文")], local_translate=lambda _: "字" * 41)

    def test_preferred_lyrics_are_preserved_and_separate(self):
        cues = [Cue(0, 1, "前", "A"), Cue(1, 2, "歌", "A", "singing", preferred_translation="歌词"), Cue(2, 3, "後", "A")]
        request = Mock(return_value=response(translated(0, 0)))
        source, output = self.run_joint(request, cues)
        self.assertEqual([c.text for c in output], ["译文", "歌词", "译文"])
        self.assertEqual(request.call_count, 2)

    def test_speakers_are_not_merged_and_pairs_are_sorted(self):
        cues = [Cue(0, 1, "一", "A"), Cue(1, 2, "二", "B"), Cue(2, 3, "三", "A")]
        source, output = self.run_joint(Mock(return_value=response(translated(0, 0))), cues,
                                       translation=TranslationConfig(batch_cues=1))
        self.assertEqual([c.speaker for c in source], ["A", "B", "A"])
        self.assertEqual([(c.start, c.end) for c in source], [(c.start, c.end) for c in output])

    def test_all_retrieval_precedes_generation(self):
        events = []
        def retrieve(cues, chat):
            events.append("rag")
            return []
        def request(body):
            events.append("llm")
            return response(translated(0, 0))
        self.run_joint(request, translation=TranslationConfig(batch_cues=1), retrieve_knowledge=retrieve)
        self.assertEqual(events, ["rag", "rag", "llm", "llm"])

    def test_completed_cache_restores_both_outputs_without_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            first = self.run_joint(Mock(return_value=response(translated(0, 1))), cache_path=path)
            second = self.run_joint(Mock(side_effect=AssertionError("request")), cues=[], cache_path=path)
            self.assertEqual(first, second)

    def test_resumed_joint_translation_rejects_a_different_provider_before_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            with self.assertRaises(KeyboardInterrupt):
                self.run_joint(Mock(side_effect=KeyboardInterrupt()), cache_path=path)
            request = Mock()
            retrieve = Mock()
            with self.assertRaises(CachedProviderMismatchError):
                self.run_joint(request, cache_path=path, retrieve_knowledge=retrieve,
                               llm=LLMConfig(base_url="https://another-provider.example/v1"))
            request.assert_not_called()
            retrieve.assert_not_called()

    def test_split_resume_reuses_completed_child(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            request = Mock(side_effect=[response(), response(), response(translated(0, 0, "左")), KeyboardInterrupt()])
            with self.assertRaises(KeyboardInterrupt):
                self.run_joint(request, cache_path=path)
            request = Mock(return_value=response(translated(0, 0, "右")))
            _, output = self.run_joint(request, cache_path=path,
                                       retrieve_knowledge=Mock(side_effect=AssertionError("retrieval")))
            request.assert_called_once()
            self.assertEqual([c.text for c in output], ["左", "右"])
    def test_actual_request_budget_failure_splits_without_truncating_source(self):
        from subtitle_pipeline.prompt_budget import PromptBudgetExceeded
        request = Mock(side_effect=[PromptBudgetExceeded("actual tokenizer budget"),
                                    response(translated(0, 0, "一")), response(translated(0, 0, "二"))])
        _, output = self.run_joint(request)
        self.assertEqual([c.text for c in output], ["一", "二"])
        self.assertEqual(request.call_count, 3)
        self.assertIn("こんにちは", request.call_args_list[1].args[0]["messages"][-1]["content"])
        self.assertIn("世界", request.call_args_list[2].args[0]["messages"][-1]["content"])

    def test_single_unit_budget_failure_never_uses_machine_translation(self):
        from subtitle_pipeline.prompt_budget import PromptBudgetExceeded
        fallback = Mock(side_effect=AssertionError("fallback"))
        with self.assertRaises(PromptBudgetExceeded):
            self.run_joint(Mock(side_effect=PromptBudgetExceeded("budget")),
                           cues=[Cue(0, 1, "原文")], local_translate=fallback)
        fallback.assert_not_called()

    def test_term_references_chat_and_dialogue_survive_joint_prompt(self):
        from subtitle_pipeline.fan_knowledge import KnowledgeHit, KnowledgeScore
        score = KnowledgeScore(0, 0, 0, 0, 0, 0, 0, 0)
        term = KnowledgeHit("term", "term", "名称", "固定译名", None, score, (), {"term_reference": 1})
        fact = KnowledgeHit("fact", "note", "背景", "背景资料", None, score, ())
        request = Mock(return_value=response(translated(0, 0)))
        self.run_joint(request, translation=TranslationConfig(batch_cues=1),
                       retrieve_knowledge=lambda *_: [term, fact], retrieve_chat=lambda _: "聊天证据")
        prompt = request.call_args_list[0].args[0]["messages"][-1]["content"]
        self.assertIn("固定译名", prompt)
        self.assertIn("背景资料", prompt)
        self.assertIn("聊天证据", prompt)
        self.assertIn("<A>世界", prompt)

    def test_song_line_splits_on_pyshiro_units_at_japanese_limit(self):
        units = tuple(
            TimedTextUnit(character, index * 0.5, (index + 1) * 0.5)
            for index, character in enumerate("一二三四五六")
        )
        result = split_aligned_song_cues(
            SongIdentificationResult(
                [
                    Cue(
                        0,
                        3,
                        "一二三四五六",
                        "A",
                        "singing",
                        preferred_translation="甲乙丙丁戊己",
                        source_units=units,
                    )
                ],
                [],
            ),
            3.0,
        )

        self.assertEqual(
            [cue.text for cue in result.corrected_cues], ["一二三", "四五六"]
        )
        self.assertEqual(
            [cue.preferred_translation for cue in result.corrected_cues],
            ["甲乙丙", "丁戊己"],
        )
        self.assertEqual(
            [(cue.start, cue.end) for cue in result.corrected_cues],
            [(0.0, 1.5), (1.5, 3.0)],
        )
        self.assertTrue(
            all(text_display_width(cue.text) <= 3 for cue in result.corrected_cues)
        )
