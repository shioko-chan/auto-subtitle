from __future__ import annotations

import json

from .repetition import RepetitionLoopError, StreamingRepetitionDetector


def read_chat_stream(response) -> dict[str, object]:
    """Consume SSE inside the caller's response context so errors close the stream."""
    payload = {}
    content = []
    reasoning = []
    detectors = {name: StreamingRepetitionDetector() for name in ('content', 'reasoning_content')}
    finish = None
    tool_calls = {}
    for event in _sse_data(response):
        if event == '[DONE]':
            if finish is None:
                raise RuntimeError('LLM stream ended without a finish reason')
            message = {'role': 'assistant', 'content': ''.join(content)}
            if reasoning:
                message['reasoning_content'] = ''.join(reasoning)
            if tool_calls:
                message['tool_calls'] = [tool_calls[index] for index in sorted(tool_calls)]
            payload['choices'] = [{'index': 0, 'message': message, 'finish_reason': finish}]
            return payload
        chunk = json.loads(event)
        if 'error' in chunk:
            raise RuntimeError(f"LLM stream error: {chunk['error']}")
        for key in ('id', 'model', 'created', 'usage', 'timings'):
            if chunk.get(key) is not None:
                payload[key] = chunk[key]
        for choice in chunk.get('choices', []):
            if choice.get('index', 0) != 0:
                raise ValueError('LLM streaming requires a single completion')
            delta = choice.get('delta', {})
            for name, parts in (('content', content), ('reasoning_content', reasoning)):
                text = delta.get(name) or ''
                match = detectors[name].feed(text)
                if match is not None:
                    raise RepetitionLoopError(match)
                parts.append(text)
            for call in delta.get('tool_calls', []):
                index = call['index']
                target = tool_calls.setdefault(index, {'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if call.get('id'):
                    target['id'] = call['id']
                for key in ('name', 'arguments'):
                    value = call.get('function', {}).get(key) or ''
                    if key == 'arguments':
                        detector = detectors.setdefault(('tool', index), StreamingRepetitionDetector())
                        match = detector.feed(value)
                        if match is not None:
                            raise RepetitionLoopError(match)
                    target['function'][key] += value
            if choice.get('finish_reason') is not None:
                finish = choice['finish_reason']
    raise RuntimeError('LLM stream disconnected before [DONE]')


def _sse_data(response):
    data = []
    for raw_line in response:
        line = raw_line.decode('utf-8').rstrip('\r\n')
        if line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
        elif not line and data:
            yield '\n'.join(data)
            data.clear()


def read_responses_stream(response) -> dict[str, object]:
    """Check deltas as they arrive; the terminal event carries the full response."""
    detectors = {}
    delta_types = {
        'response.output_text.delta',
        'response.reasoning_text.delta',
        'response.reasoning_summary_text.delta',
        'response.function_call_arguments.delta',
        'response.refusal.delta',
    }
    for data in _sse_data(response):
        if data == '[DONE]':
            break
        event = json.loads(data)
        kind = event.get('type')
        if kind in delta_types:
            key = (kind, event.get('output_index'), event.get('item_id'),
                   event.get('content_index'), event.get('summary_index'))
            detector = detectors.setdefault(key, StreamingRepetitionDetector())
            match = detector.feed(event['delta'])
            if match is not None:
                raise RepetitionLoopError(match)
        elif kind in ('error', 'response.failed'):
            raise RuntimeError(f"Responses stream failed: {event}")
        elif kind in ('response.completed', 'response.incomplete'):
            payload = event['response']
            expected = kind.removeprefix('response.')
            if payload.get('status') != expected:
                raise RuntimeError('Responses stream terminal status mismatch')
            return payload
    raise RuntimeError('Responses stream disconnected before terminal event')
