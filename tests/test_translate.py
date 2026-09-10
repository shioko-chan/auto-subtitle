import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from subtitle_pipeline.config import LLMConfig, TranslationConfig
from subtitle_pipeline.translate import (
    LLMHTTPError,
    LocalLLMError,
    OpenAICompatibleTranslator,
    _is_nontransient_http_error,
    _normalize_api_response,
    _parse_json_object,
    _prompt_section_sizes,
    _transient_retry_delay,
    is_llm_quota_exhausted,
)


class PromptBudgetTests(unittest.TestCase):
    def test_auxiliary_and_stage_requests_cannot_bypass_local_budget(self):
        from subtitle_pipeline.prompt_budget import PromptBudgetExceeded
        translator = OpenAICompatibleTranslator(LLMConfig(local_server_enabled=True, local_server_context_size=100), TranslationConfig(), '')
        body = {'messages': [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'context'}], 'max_tokens': 40}
        for send in (translator.request, translator.stage_request(None, 'asr_correction')):
            with patch.object(translator, '_budget_post', side_effect=[{'prompt': 'formatted'}, {'tokens': list(range(61))}]), patch('subtitle_pipeline.translate.urllib.request.urlopen') as network:
                with self.assertRaises(PromptBudgetExceeded):
                    send(body)
                network.assert_not_called()

    def test_only_http_402_is_balance_exhaustion(self) -> None:
        quota = LLMHTTPError(402, "insufficient balance")
        wrapped = RuntimeError("term extraction failed")
        wrapped.__cause__ = quota

        self.assertTrue(is_llm_quota_exhausted(wrapped))
        self.assertFalse(is_llm_quota_exhausted(LLMHTTPError(429, "rate limit")))

    @patch("subtitle_pipeline.translate.urllib.request.urlopen")
    def test_local_request_has_no_network_timeout(self, urlopen) -> None:
        response = urlopen.return_value.__enter__.return_value
        response.__iter__.return_value = iter(_stream_lines("{}"))
        translator = OpenAICompatibleTranslator(
            LLMConfig(local_server_enabled=True, local_server_context_size=16384), TranslationConfig(), ""
        )

        with patch.object(translator, "_budget_post", side_effect=[{"prompt": "test"}, {"tokens": [1]}]):
            translator.request({"messages": []})

        self.assertNotIn("timeout", urlopen.call_args.kwargs)

    def test_prompt_section_sizes_separate_dynamic_evidence(self):
        sections = _prompt_section_sizes(
            "rules\nREFERENCE:\nref\nCURRENT_VIDEO_CHAT:\nchat\n"
            "DIALOGUE_CONTEXT:\ncontext\nSOURCE:\nsource"
        )

        self.assertEqual(sections["reference"]["characters"], 3)
        self.assertEqual(sections["current_video_chat"]["characters"], 4)
        self.assertEqual(sections["dialogue_context"]["characters"], 7)
        self.assertEqual(sections["source"]["characters"], 6)
        self.assertEqual(sections["fixed_prompt"]["characters"], 5)

    def test_request_context_audit_records_actual_usage_and_section_sizes(self):
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / "llm-audit.jsonl"
            translator = OpenAICompatibleTranslator(
                LLMConfig(local_server_enabled=True, local_server_context_size=16384),
                TranslationConfig(),
                "secret",
                audit_path=audit,
            )
            translator._log_request_context(
                {
                    "model": "test",
                    "max_tokens": 8192,
                    "messages": [
                        {
                            "role": "user",
                            "content": "SOURCE:\n原文\nCURRENT_VIDEO_CHAT:\n聊天",
                        }
                    ],
                },
                {
                    "usage": {
                        "prompt_tokens": 123,
                        "completion_tokens": 45,
                    }
                },
            )
            event = json.loads(audit.read_text())

        self.assertEqual(event["prompt_tokens"], 123)
        self.assertEqual(event["completion_tokens"], 45)
        self.assertEqual(event["context_capacity"], 16384)
        self.assertIn("source", event["sections"])
        self.assertIn("current_video_chat", event["sections"])

    @patch("subtitle_pipeline.translate.urllib.request.urlopen")
    def test_successful_request_audit_records_request_and_response(
        self, urlopen
    ) -> None:
        response = urlopen.return_value.__enter__.return_value
        response.__iter__.return_value = iter(_stream_lines('{"cues":[]}'))
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / "llm-audit.jsonl"
            translator = OpenAICompatibleTranslator(
                LLMConfig(local_server_enabled=True, local_server_context_size=16384),
                TranslationConfig(),
                "secret",
                audit_path=audit,
            )
            with patch.object(translator, "_budget_post", side_effect=[{"prompt": "test"}, {"tokens": [1]}]):
                translator.request({"messages": [{"role": "user", "content": "test"}]})
            events = [json.loads(line) for line in audit.read_text().splitlines()]

        success = next(event for event in events if event["event"] == "llm_response")
        self.assertTrue(success["request_id"])
        self.assertEqual(success["request"]["messages"][0]["content"], "test")
        self.assertEqual(
            success["response"]["choices"][0]["message"]["content"],
            '{"cues":[]}',
        )

    def test_local_llm_error_is_not_retried_as_remote_http_failure(self) -> None:
        error = LocalLLMError("local inference failed")

        self.assertIsNone(_transient_retry_delay(error, 1))
        self.assertTrue(_is_nontransient_http_error(error))


def _response(content):
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}


class ApiCompatibilityTests(unittest.TestCase):
    def test_lyrics_prompt_excludes_runtime_audit_payloads(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(), TranslationConfig(), "secret"
        )
        response = _response(
            json.dumps(
                {"lines": [{"line_id": 0, "text": "准备好去寻找答案"}]},
                ensure_ascii=False,
            )
        )
        with patch.object(translator, "_request", return_value=response) as request:
            translator.translate_lyrics(
                "Song",
                "Artist",
                ["Ready set and find out"],
                translation_context={
                    "video": {"description": "large video context"},
                    "franchises": [
                        {"name": "夢限大みゅーたいぷ", "background": "large background"}
                    ],
                    "terms": {"ミヤコ": "都子"},
                    "asr_evidence": [{"text": "large ASR audit"}],
                    "identified_songs": [{"pyshiro": "large alignment audit"}],
                },
            )

        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertIn("夢限大みゅーたいぷ", prompt)
        self.assertIn("ミヤコ", prompt)
        self.assertNotIn("large background", prompt)
        self.assertNotIn("large video context", prompt)
        self.assertNotIn("large ASR audit", prompt)
        self.assertNotIn("large alignment audit", prompt)

    def test_lyrics_failure_does_not_fall_back_to_machine_translation(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(max_retries=1), TranslationConfig(), "secret"
        )
        with (
            patch.object(translator, "_request", side_effect=ValueError("invalid")),
            patch.object(
                translator.local_translator,
                "translate",
                side_effect=AssertionError("machine translation must not run"),
            ),
            self.assertRaisesRegex(RuntimeError, "lyrics translation exhausted"),
        ):
            translator.translate_lyrics("Song", "Artist", ["歌詞"])

    def test_metadata_prompt_excludes_large_audit_payloads(self):
        translator = OpenAICompatibleTranslator(
            LLMConfig(), TranslationConfig(), "secret"
        )
        response = _response(
            json.dumps(
                {
                    "title": "标题",
                    "description": "简介",
                    "content_summary": "摘要",
                    "tags": ["标签"],
                },
                ensure_ascii=False,
            )
        )
        with patch.object(translator, "_request", return_value=response) as request:
            translator.translate_metadata(
                "title",
                "description",
                translation_context={
                    "terms": {"ミヤコ": "都子"},
                    "identified_songs": [{"alignment": "large"}],
                    "asr_evidence": [{"text": "large"}],
                },
            )
        prompt = request.call_args.args[0]["messages"][1]["content"]
        self.assertIn("ミヤコ", prompt)
        self.assertNotIn("identified_songs", prompt)
        self.assertNotIn("asr_evidence", prompt)

    def test_only_transient_http_failures_receive_backoff(self):
        self.assertIsNotNone(_transient_retry_delay(LLMHTTPError(429, "rate"), 1))
        self.assertIsNotNone(_transient_retry_delay(LLMHTTPError(503, "busy"), 1))
        self.assertIsNone(_transient_retry_delay(LLMHTTPError(401, "auth"), 1))
        self.assertTrue(_is_nontransient_http_error(LLMHTTPError(401, "auth")))

    def test_parses_fenced_json(self):
        self.assertEqual(_parse_json_object('```json\n{"value":1}\n```'), {"value": 1})

    def test_normalizes_openai_responses_output(self):
        normalized = _normalize_api_response(
            LLMConfig(api_style="responses"),
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )
        self.assertEqual(normalized["choices"][0]["message"]["content"], "ok")


def _stream_lines(content):
    chunk = {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}]}
    return [("data: " + json.dumps(chunk) + "\n").encode(), b"\n", b"data: [DONE]\n", b"\n"]


class RemoteStreamingTests(unittest.TestCase):
    @patch("subtitle_pipeline.translate.urllib.request.urlopen")
    def test_remote_interfaces_stream_and_preserve_response(self, urlopen):
        import io
        for style in ("chat_completions", "responses"):
            translator = OpenAICompatibleTranslator(LLMConfig(api_style=style), TranslationConfig(), "secret")
            if style == "chat_completions":
                raw = b"".join(_stream_lines('{"ok":true}'))
            else:
                payload = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": '{"ok":true}'}]}]}
                raw = ("data: " + json.dumps({"type": "response.completed", "response": payload}) + "\n\n").encode()
            response = io.BytesIO(raw)
            urlopen.return_value = response
            result = translator.request({"messages": []})
            self.assertTrue(response.closed)
            sent = json.loads(urlopen.call_args.args[0].data)
            self.assertTrue(sent["stream"])
            self.assertEqual("stream_options" in sent, style == "chat_completions")
            self.assertIn("timeout", urlopen.call_args.kwargs)
            self.assertEqual(result["choices"][0]["message"]["content"], '{"ok":true}')

    @patch("subtitle_pipeline.translate.urllib.request.urlopen")
    def test_remote_loop_closes_connection_without_success_audit(self, urlopen):
        import io
        from subtitle_pipeline.repetition import RepetitionLoopError
        for style in ("chat_completions", "responses"):
            if style == "chat_completions":
                raw = b"".join(_stream_lines("前" * 160))
            else:
                raw = ("data: " + json.dumps({"type": "response.output_text.delta", "item_id": "a", "delta": "前" * 160}) + "\n\n").encode()
            response = io.BytesIO(raw)
            urlopen.return_value = response
            translator = OpenAICompatibleTranslator(LLMConfig(api_style=style), TranslationConfig(), "secret")
            with patch.object(translator, "_log_successful_response") as success:
                with self.assertRaises(RepetitionLoopError):
                    translator.request({"messages": []})
                success.assert_not_called()
            self.assertTrue(response.closed)


if __name__ == "__main__":
    unittest.main()
