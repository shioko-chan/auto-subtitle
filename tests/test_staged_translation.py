from subtitle_pipeline.repetition import RepetitionLoopError, RepetitionMatch
import json
import tempfile
import unittest
from pathlib import Path

from subtitle_pipeline.config import LLMConfig, SegmentationConfig, TranslationConfig
from subtitle_pipeline.fan_knowledge import KnowledgeHit, KnowledgeScore
from subtitle_pipeline.local_segmentation import LocalUnit, SpeakerTrack
from subtitle_pipeline.song_identification import (
    SongIdentificationResult,
    split_aligned_song_cues,
)
from subtitle_pipeline.staged_translation import (
    _topic_request_groups,
    run_fixed_translation,
    run_segmentation,
)
from subtitle_pipeline.subtitles import Cue, TimedTextUnit, text_display_width


def parse_cues(content):
    return json.loads(content)["cues"]


def finish_reason(_response):
    return "stop"


def no_delay(_error, _attempt):
    return None


def not_nontransient(_error):
    return False


def ignore_audit(*_args):
    return None


class StagedTranslationTests(unittest.TestCase):
    def test_single_cue_format_failure_does_not_use_machine_translation(self):
        source = [Cue(0, 1, "原文", "A")]

        with self.assertRaises(RuntimeError):
            run_fixed_translation(
                source_cues=source,
                llm=LLMConfig(),
                translation=TranslationConfig(),
                request=lambda _body: {
                    "choices": [{"message": {"content": '{"cues": []}'}}]
                },
                parse_content=parse_cues,
                finish_reason=finish_reason,
                retry_delay=no_delay,
                is_nontransient=not_nontransient,
                log_invalid_response=ignore_audit,
                local_translate=lambda _text: self.fail(
                    "format errors must not use machine translation"
                ),
                translation_context={},
                honorific_rules="",
                cache_path=None,
            )

    def test_translation_does_not_reject_residual_japanese(self):
        source = [Cue(0, 1, "原文", "A")]

        result = run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            translation=TranslationConfig(),
            request=lambda _body: {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"cues": [{"cue_id": 0, "text": "这是すやバラ"}]},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda _text: self.fail(
                "non-empty LLM output must not use machine translation"
            ),
            translation_context={},
            honorific_rules="",
            cache_path=None,
        )

        self.assertEqual(result[0].text, "这是すやバラ")

    def test_fixed_translation_separates_terms_from_background_knowledge(self):
        source = [Cue(0, 1, "すやバラです", "A")]
        prompts: list[str] = []
        term = KnowledgeHit(
            "term:suyabara",
            "term",
            "すやバラ",
            "すやバラ的固定中文译法为助眠抒情歌回。",
            None,
            KnowledgeScore(1, 0, 0, 0, 0, 0, 1, 2),
            ("すやバラ",),
            {"term_reference": 1},
        )
        background = KnowledgeHit(
            "chunk:history",
            "document_chunk",
            "历史直播",
            "这是相关的历史直播背景。",
            None,
            KnowledgeScore(1, 0, 0, 0, 0, 0, 1, 2),
            ("すやバラ",),
        )

        def request(body):
            prompts.append(body["messages"][1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"cues": [{"cue_id": 0, "text": "助眠抒情歌回"}]},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            translation=TranslationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: text,
            translation_context={},
            honorific_rules="",
            cache_path=None,
            retrieve_knowledge=lambda _cues, _chat: [term, background],
        )

        prompt = prompts[0]
        term_section = prompt.split("TERM_REFERENCE:\n", 1)[1].split(
            "\n\nDIALOGUE_CONTEXT:", 1
        )[0]
        topic_block = prompt.split("<TOPIC_BLOCK>", 1)[1]
        self.assertIn("固定中文译法为助眠抒情歌回", term_section)
        self.assertNotIn("历史直播背景", term_section)
        self.assertIn("历史直播背景", topic_block)
        self.assertNotIn("固定中文译法为助眠抒情歌回", topic_block)

    def test_translation_repetition_minimizer_identifies_target_cue(self):
        source = [Cue(0, 1, "しいドタタタンドタタタン", "A")]
        audits: list[tuple[object, ...]] = []
        calls = 0

        def request(_body):
            nonlocal calls
            calls += 1
            raise RepetitionLoopError(RepetitionMatch("哒", 160, 0, 160))

        result = run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            translation=TranslationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=lambda *args: audits.append(args),
            local_translate=lambda text: f"MT:{text}",
            translation_context={},
            honorific_rules="",
            cache_path=None,
        )

        diagnosis = next(value[2] for value in audits if value[0] == "repetition minimizer")
        self.assertEqual(calls, 2)
        self.assertEqual(result[0].text, "MT:しいドタタタンドタタタン")
        self.assertEqual(diagnosis["trigger"], "target")
        self.assertTrue(diagnosis["minimal_reproducer"])

    def test_translation_repetition_minimizer_identifies_chat(self):
        source = [Cue(0, 1, "原文", "A")]
        audits: list[tuple[object, ...]] = []
        calls = 0

        def request(body):
            nonlocal calls
            calls += 1
            prompt = body["messages"][1]["content"]
            if "会触发循环的聊天" in prompt:
                raise RepetitionLoopError(RepetitionMatch("哒", 160, 0, 160))
            content = (
                json.dumps(
                    {"cues": [{"cue_id": 0, "text": "无聊天译文"}]},
                    ensure_ascii=False,
                )
            )
            return {"choices": [{"message": {"content": content}}]}

        result = run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            translation=TranslationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=lambda *args: audits.append(args),
            local_translate=lambda text: f"MT:{text}",
            translation_context={},
            honorific_rules="",
            cache_path=None,
            retrieve_chat=lambda _cues: "[chat] 会触发循环的聊天",
        )

        diagnosis = next(value[2] for value in audits if value[0] == "repetition minimizer")
        self.assertEqual(calls, 2)
        self.assertEqual(result[0].text, "无聊天译文")
        self.assertEqual(diagnosis["trigger"], "chat")
        self.assertFalse(diagnosis["minimal_reproducer"])

    def test_translation_repetition_minimizer_identifies_knowledge(self):
        source = [Cue(0, 1, "原文", "A")]
        audits: list[tuple[object, ...]] = []
        calls = 0
        hit = KnowledgeHit(
            "knowledge:loop",
            "document_chunk",
            "循环知识",
            "会触发循环的知识正文",
            None,
            KnowledgeScore(1, 0, 0, 0, 0, 0, 1, 2),
            ("循环",),
        )

        def request(body):
            nonlocal calls
            calls += 1
            prompt = body["messages"][1]["content"]
            if "会触发循环的知识正文" in prompt:
                raise RepetitionLoopError(RepetitionMatch("哒", 160, 0, 160))
            content = (
                json.dumps(
                    {"cues": [{"cue_id": 0, "text": "无知识译文"}]},
                    ensure_ascii=False,
                )
            )
            return {"choices": [{"message": {"content": content}}]}

        result = run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            translation=TranslationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=lambda *args: audits.append(args),
            local_translate=lambda text: f"MT:{text}",
            translation_context={},
            honorific_rules="",
            cache_path=None,
            retrieve_knowledge=lambda _cues, _chat: [hit],
        )

        diagnosis = next(value[2] for value in audits if value[0] == "repetition minimizer")
        self.assertEqual(calls, 3)
        self.assertEqual(result[0].text, "无知识译文")
        self.assertEqual(diagnosis["trigger"], "knowledge")
        self.assertEqual(diagnosis["knowledge_ids"], ["knowledge:loop"])

    def test_short_cues_are_bounded_by_topic_size(self):
        source = [Cue(i, i + 1, f"短句{i}", "A") for i in range(20)]
        prompts: list[str] = []

        def request(body):
            prompt = body["messages"][1]["content"]
            prompts.append(prompt)
            cue_count = prompt.count('<CUE id="')
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": i, "text": f"译文{i}"}
                                        for i in range(cue_count)
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        result = run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            translation=TranslationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: text,
            translation_context={},
            honorific_rules="",
            cache_path=None,
        )

        self.assertEqual([prompt.count('<CUE id="') for prompt in prompts], [16, 4])
        self.assertEqual(len(result), 20)

    def test_segmentation_accepts_an_overwide_source_group_without_retry(self):
        cues = [Cue(0, 1, "あいう", "A"), Cue(1, 2, "えおか", "A")]
        track = SpeakerTrack(
            "A",
            "A",
            (
                LocalUnit("A", 0, (0,), 0, 1, "あいう", "A", "speech"),
                LocalUnit("A", 1, (1,), 1, 2, "えおか", "A", "speech"),
            ),
        )
        requests = 0

        def request(_body):
            nonlocal requests
            requests += 1
            response = {"cues": [{"start_id": 0, "end_id": 1}]}
            return {"choices": [{"message": {"content": json.dumps(response)}}]}

        result = run_segmentation(
            tracks=[track],
            source_cues=cues,
            segmentation=SegmentationConfig(),
            llm=LLMConfig(max_retries=2),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            source_maximum_units=4.0,
            cache_path=None,
            sudachi_versions={},
        )

        self.assertEqual([cue.text for cue in result], ["あいうえおか"])
        self.assertEqual(requests, 1)

    def test_fixed_translation_cannot_change_source_boundaries(self):
        source = [Cue(0, 1, "原文一", "A"), Cue(1, 2, "原文二", "A")]

        def request(_body):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": 0, "text": "译文一"},
                                        {"cue_id": 1, "text": "译文二"},
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        translated = run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            translation=TranslationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: f"MT:{text}",
            translation_context={},
            honorific_rules="",
            cache_path=None,
        )

        self.assertEqual([(cue.start, cue.end) for cue in translated], [(0, 1), (1, 2)])
        self.assertEqual([cue.text for cue in translated], ["译文一", "译文二"])

    def test_fixed_translation_shares_topic_knowledge_across_speakers(self):
        source = [
            Cue(0, 1, "本日分の貢ぎ物", "A"),
            Cue(1, 2, "等身大フィギュア", "B"),
        ]
        retrieved: list[list[Cue]] = []
        prompts: list[str] = []

        def retrieve(cues, _chat_text):
            retrieved.append(cues)
            return [
                KnowledgeHit(
                    f"knowledge:{text}",
                    "catchphrase",
                    text,
                    f"{text}的专属知识。",
                    None,
                    KnowledgeScore(1, 0, 0, 1, 1, 0, 1, 5),
                    (text,),
                )
                for text in (cue.text for cue in cues)
            ]

        def request(body):
            prompts.append(body["messages"][1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": 0, "text": "今日的礼物"},
                                        {"cue_id": 1, "text": "等身大手办"},
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            translation=TranslationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: f"MT:{text}",
            translation_context={},
            honorific_rules="",
            cache_path=None,
            retrieve_knowledge=retrieve,
        )

        self.assertEqual(
            [[cue.text for cue in group] for group in retrieved],
            [["本日分の貢ぎ物", "等身大フィギュア"]],
        )
        topic = prompts[0].index("<TOPIC_BLOCK>")
        first = prompts[0].index('<CUE id="0"')
        second = prompts[0].index('<CUE id="1"')
        self.assertLess(topic, first)
        self.assertLess(first, second)
        self.assertEqual(prompts[0].count("本日分の貢ぎ物的专属知识"), 1)
        self.assertEqual(prompts[0].count("等身大フィギュア的专属知识"), 1)

    def test_topic_groups_cross_speakers_but_not_verified_lyrics(self):
        cues = [
            Cue(0, 1, "質問", "A"),
            Cue(1, 2, "返事", "B"),
            Cue(2, 3, "歌詞", "A", "singing", preferred_translation="歌词"),
            Cue(3, 4, "歌の後", "B"),
        ]

        groups = _topic_request_groups(cues, [0, 1, 3], TranslationConfig())

        self.assertEqual(groups, [[0, 1], [3]])

    def test_local_prompt_budget_splits_an_oversized_topic(self):
        source = [Cue(i, i + 1, "長い字幕" * 180, "A") for i in range(4)]
        request_sizes: list[int] = []
        retrievals = 0

        def retrieve(_cues, _chat):
            nonlocal retrievals
            retrievals += 1
            return []

        def request(body):
            prompt = body["messages"][1]["content"]
            cue_count = prompt.count('<CUE id="')
            request_sizes.append(cue_count)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": i, "text": f"译文{i}"}
                                        for i in range(cue_count)
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        result = run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(
                local_server_enabled=True,
                local_server_context_size=7200,
            ),
            translation=TranslationConfig(max_tokens=4096),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: f"MT:{text}",
            translation_context={},
            honorific_rules="",
            cache_path=None,
            retrieve_knowledge=retrieve,
        )

        self.assertGreater(len(request_sizes), 1)
        self.assertEqual(retrievals, 1)
        self.assertEqual(len(result), 4)

    def test_local_prompt_budget_reduces_evidence_before_splitting_topic(self):
        source = [Cue(i, i + 1, f"短句{i}", "A") for i in range(4)]
        prompts: list[str] = []

        def request(body):
            prompt = body["messages"][1]["content"]
            prompts.append(prompt)
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": i, "text": f"译文{i}"}
                                        for i in range(4)
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        chat = "\n".join(f"[chat] 普通聊天消息{i}" for i in range(500))
        result = run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(
                local_server_enabled=True,
                local_server_context_size=7200,
            ),
            translation=TranslationConfig(max_tokens=4096),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: f"MT:{text}",
            translation_context={},
            honorific_rules="",
            cache_path=None,
            retrieve_chat=lambda _cues: chat,
        )

        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0].count('<CUE id="'), 4)
        self.assertLess(prompts[0].count("普通聊天消息"), 500)
        self.assertEqual(len(result), 4)

    def test_fixed_translation_retrieves_chat_once_for_request_group(self):
        source = [Cue(10, 11, "これ", "A"), Cue(11, 12, "それ", "A")]
        prompts: list[str] = []

        def request(body):
            prompts.append(body["messages"][1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": 0, "text": "这个"},
                                        {"cue_id": 1, "text": "那个"},
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            translation=TranslationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: text,
            translation_context={},
            honorific_rules="",
            cache_path=None,
            retrieve_chat=lambda cues: (
                f"chat@{min(cue.start for cue in cues):.0f}-"
                f"{max(cue.end for cue in cues):.0f}"
            ),
        )

        self.assertIn("chat@10-12", prompts[0])
        self.assertEqual(prompts[0].count("chat@10-12"), 1)

    def test_fixed_translation_prepares_all_topic_evidence_before_llm(self):
        source = [Cue(0, 1, "一", "A"), Cue(10, 11, "二", "A")]
        events: list[str] = []

        def retrieve(cues, _chat):
            events.append(f"retrieve:{cues[0].text}")
            return []

        def request(body):
            events.append("request")
            cue_count = body["messages"][1]["content"].count('<CUE id="')
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": index, "text": f"译文{index}"}
                                        for index in range(cue_count)
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            translation=TranslationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: text,
            translation_context={},
            honorific_rules="",
            cache_path=None,
            retrieve_knowledge=retrieve,
        )

        self.assertEqual(events[:2], ["retrieve:一", "retrieve:二"])
        self.assertEqual(events.count("request"), 2)

    def test_fixed_translation_deduplicates_chat_shared_by_adjacent_cues(self):
        source = [Cue(10, 11, "一", "A"), Cue(11, 12, "二", "A")]
        prompts: list[str] = []

        def request(body):
            prompts.append(body["messages"][1]["content"])
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": 0, "text": "一"},
                                        {"cue_id": 1, "text": "二"},
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        run_fixed_translation(
            source_cues=source,
            llm=LLMConfig(),
            translation=TranslationConfig(),
            request=request,
            parse_content=parse_cues,
            finish_reason=finish_reason,
            retry_delay=no_delay,
            is_nontransient=not_nontransient,
            log_invalid_response=ignore_audit,
            local_translate=lambda text: text,
            translation_context={},
            honorific_rules="",
            cache_path=None,
            retrieve_chat=lambda _cue: "[+1.0s chat] 共同证据",
        )

        self.assertEqual(prompts[0].count("共同证据"), 1)
        self.assertIn("[+1.0s chat] 共同证据", prompts[0])

    def test_fixed_translation_cache_does_not_depend_on_knowledge(self):
        source = [Cue(0, 1, "原文", "A")]

        def request(_body):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"cues": [{"cue_id": 0, "text": "译文"}]},
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            }

        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary) / "translation.json"
            arguments = {
                "source_cues": source,
                "llm": LLMConfig(),
                "translation": TranslationConfig(),
                "parse_content": parse_cues,
                "finish_reason": finish_reason,
                "retry_delay": no_delay,
                "is_nontransient": not_nontransient,
                "log_invalid_response": ignore_audit,
                "local_translate": lambda text: f"MT:{text}",
                "translation_context": {},
                "honorific_rules": "",
                "cache_path": cache,
            }
            first = run_fixed_translation(
                **arguments,
                request=request,
                retrieve_knowledge=lambda _cues, _chat: [],
            )
            second = run_fixed_translation(
                **arguments,
                request=lambda _body: self.fail("LLM should not run on cache hit"),
                retrieve_knowledge=lambda _cues, _chat: self.fail(
                    "knowledge retrieval should not run on cache hit"
                ),
                retrieve_chat=lambda _cue: self.fail(
                    "chat retrieval should not run on cache hit"
                ),
            )

        self.assertEqual(first, second)

    def test_fixed_translation_audits_llm_downgrade_and_cache_sources(self):
        source = [Cue(0, 1, "原文一", "A"), Cue(1, 2, "原文二", "A")]

        def request(_body):
            return {
                "_audit_request_id": "request-1",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "cues": [
                                        {"cue_id": 0, "text": "译文一"},
                                        {"cue_id": 1, "text": ""},
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ],
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "translation.json"
            audit = root / "translation-audit.jsonl"
            arguments = {
                "source_cues": source,
                "llm": LLMConfig(),
                "translation": TranslationConfig(),
                "parse_content": parse_cues,
                "finish_reason": finish_reason,
                "retry_delay": no_delay,
                "is_nontransient": not_nontransient,
                "log_invalid_response": ignore_audit,
                "local_translate": lambda text: f"MT:{text}",
                "translation_context": {},
                "honorific_rules": "",
                "cache_path": cache,
                "audit_path": audit,
            }
            run_fixed_translation(**arguments, request=request)
            run_fixed_translation(
                **arguments,
                request=lambda _body: self.fail("cache hit should skip LLM"),
            )
            events = [json.loads(line) for line in audit.read_text().splitlines()]

        self.assertEqual(
            [event["translation_source"] for event in events],
            ["llm", "local_mt", "llm", "local_mt"],
        )
        self.assertTrue(events[2]["cache_hit"])
        self.assertEqual(events[3]["downgrade_reason"], "empty_translation")
        self.assertEqual(events[0]["request_id"], "request-1")
        self.assertEqual(events[1]["downgrade_reason"], "empty_translation")
        self.assertEqual(events[1]["source_text"], "原文二")
        self.assertEqual(events[1]["final_text"], "MT:原文二")

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


if __name__ == "__main__":
    unittest.main()
