import io
import json
import unittest

from subtitle_pipeline.llm_stream import read_chat_stream
from subtitle_pipeline.repetition import RepetitionLoopError, StreamingRepetitionDetector


def event(value):
    return ('data: ' + json.dumps(value, ensure_ascii=False) + '\n\n').encode()


class StreamTests(unittest.TestCase):
    def test_detection_independent_of_chunks_and_ignores_spaces(self):
        for size in (1, 7, 500):
            detector = StreamingRepetitionDetector()
            text = '不知道 ' * 60
            match = None
            for i in range(0, len(text), size):
                match = detector.feed(text[i:i + size])
                if match:
                    break
            self.assertEqual(match.pattern, '不知道')
            self.assertEqual(match.end, 162)
        self.assertIsNone(StreamingRepetitionDetector().feed('はいはいはいはい、大丈夫です'))

    def test_filter_candidates_require_exact_match(self):
        self.assertIsNone(StreamingRepetitionDetector().feed(''.join(f'{i:04d}!' for i in range(200))))

    def test_stream_reassembles_unicode_finish_and_usage(self):
        raw = b''.join(event({'choices': [{'index': 0, 'delta': {'content': part}, 'finish_reason': None}]})
                       for part in ('{"text":"', '你好', '"}'))
        raw += event({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
        raw += event({'choices': [], 'usage': {'completion_tokens': 8}})
        raw += b'data: [DONE]\n\n'
        result = read_chat_stream(io.BytesIO(raw))
        self.assertEqual(result['choices'][0]['message']['content'], '{"text":"你好"}')
        self.assertEqual(result['usage']['completion_tokens'], 8)

    def test_loop_aborts_before_reading_remaining_stream_and_closes(self):
        raw = event({'choices': [{'delta': {'content': '前' * 160}}]}) + b'unread remainder\n'
        stream = io.BytesIO(raw)
        with self.assertRaises(RepetitionLoopError):
            with stream:
                read_chat_stream(stream)
        self.assertTrue(stream.closed)

    def test_truncated_stream_is_not_success(self):
        with self.assertRaises(RuntimeError):
            read_chat_stream(io.BytesIO(event({'choices': [{'delta': {'content': '{}'}, 'finish_reason': 'stop'}]})))


class ResponsesStreamTests(unittest.TestCase):
    def test_preserves_terminal_payload_and_usage(self):
        from subtitle_pipeline.llm_stream import read_responses_stream
        payload = {'status': 'completed', 'output': [], 'usage': {'output_tokens': 3}}
        raw = event({'type': 'response.output_text.delta', 'item_id': 'a', 'delta': 'hello'})
        raw += event({'type': 'response.completed', 'response': payload})
        self.assertEqual(read_responses_stream(io.BytesIO(raw)), payload)

    def test_checks_reasoning_and_tool_arguments_across_events(self):
        from subtitle_pipeline.llm_stream import read_responses_stream
        for kind in ('response.output_text.delta', 'response.reasoning_text.delta',
                     'response.reasoning_summary_text.delta', 'response.function_call_arguments.delta'):
            raw = b''.join(event({'type': kind, 'item_id': 'a', 'delta': '前' * 40}) for _ in range(4))
            with self.assertRaises(RepetitionLoopError):
                read_responses_stream(io.BytesIO(raw))

    def test_incomplete_is_preserved_but_failure_and_disconnect_raise(self):
        from subtitle_pipeline.llm_stream import read_responses_stream
        payload = {'status': 'incomplete', 'output': [], 'incomplete_details': {'reason': 'max_output_tokens'}}
        self.assertEqual(read_responses_stream(io.BytesIO(event({'type': 'response.incomplete', 'response': payload}))), payload)
        for raw in (b'', b'data: [DONE]\n\n', event({'type': 'response.failed', 'response': {'error': 'failed'}})):
            with self.assertRaises(RuntimeError):
                read_responses_stream(io.BytesIO(raw))

    def test_chat_tool_call_arguments_are_assembled(self):
        raw = event({'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call1', 'function': {'name': 'lookup', 'arguments': '{'}}]}}]})
        raw += event({'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '}'}}]}, 'finish_reason': 'tool_calls'}]})
        raw += b'data: [DONE]\n\n'
        call = read_chat_stream(io.BytesIO(raw))['choices'][0]['message']['tool_calls'][0]
        self.assertEqual(call['function'], {'name': 'lookup', 'arguments': '{}'})
