"""Direct OpenAI-compatible Chat Completions client (no agent framework)."""
import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import sys
import time

import httpx


def build_request(config, prompt, system=None, images=(), stream=True):
    model = config.get('model', '').strip()
    if not model:
        raise ValueError('Fill model with the exact API model ID.')
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    content = prompt
    if images:
        content = [{'type': 'text', 'text': prompt}]
        for filename in images:
            path = Path(filename)
            mime = mimetypes.guess_type(path.name)[0]
            if mime not in ('image/png', 'image/jpeg', 'image/webp'):
                raise ValueError('Images must be PNG, JPEG or WebP.')
            encoded = base64.b64encode(path.read_bytes()).decode('ascii')
            content.append({'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}})
    messages.append({'role': 'user', 'content': content})
    body = {'model': model, 'messages': messages, 'stream': stream}
    for key in ('max_tokens', 'temperature', 'reasoning_effort'):
        if config.get(key) is not None:
            body[key] = config[key]
    # Provider-specific options must be verified for the chosen model.
    extra = config.get('extra_body', {})
    if not isinstance(extra, dict) or set(extra) & {'model', 'messages', 'stream'}:
        raise ValueError('extra_body must be an object and cannot override model/messages/stream.')
    body.update(extra)
    return body


def sse_events(lines):
    data = []
    for line in lines:
        if not line:
            if data:
                yield '\n'.join(data)
                data = []
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
    if data:
        yield '\n'.join(data)


def consume(chunks, show_reasoning=False):
    answer, reasoning, usage, finish = [], [], None, None
    first_answer = None
    started = time.monotonic()
    for chunk in chunks:
        if chunk.get('error'):
            raise ValueError('API returned an error in the response stream.')
        if chunk.get('usage') is not None:
            usage = chunk['usage']
        for choice in chunk.get('choices', []):
            if choice.get('index', 0) != 0:
                continue
            message = choice.get('delta', choice.get('message', {}))
            thought = message.get('reasoning_content') or ''
            text = message.get('content') or ''
            if thought:
                reasoning.append(thought)
                if show_reasoning:
                    print(thought, end='', file=sys.stderr, flush=True)
            if text:
                if first_answer is None:
                    first_answer = time.monotonic()-started
                answer.append(text)
                print(text, end='', flush=True)
            if choice.get('finish_reason') is not None:
                finish = choice['finish_reason']
    print()
    result = dict(answer=''.join(answer), usage=usage, finish_reason=finish,
                  reasoning_characters=sum(map(len, reasoning)),
                  first_answer_s=first_answer, elapsed_s=time.monotonic()-started)
    if show_reasoning:
        result['reasoning_content'] = ''.join(reasoning)
    print(json.dumps({k: v for k, v in result.items() if k not in ('answer', 'reasoning_content')},
                     ensure_ascii=False), file=sys.stderr)
    if finish == 'length':
        print('Response reached its token limit and may be incomplete.', file=sys.stderr)
    if finish is None:
        raise ValueError('Response ended without finish_reason; treat the output as incomplete.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/deepseek.local.json')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt')
    source.add_argument('--prompt-file', help='UTF-8 file; use - for stdin')
    parser.add_argument('--system-file', help='UTF-8 system prompt')
    parser.add_argument('--image', action='append', default=[])
    parser.add_argument('--no-stream', action='store_true')
    parser.add_argument('--show-reasoning', action='store_true')
    parser.add_argument('--output', help='Optional JSON result; existing files are not overwritten')
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        key = os.environ.get('DEEPSEEK_API_KEY') or config.get('api_key', '')
        if not key.strip():
            raise ValueError('Fill api_key in the local config or set DEEPSEEK_API_KEY.')
        base = config.get('base_url', '').rstrip('/')
        if not base.startswith('https://'):
            raise ValueError('Fill base_url with an HTTPS API root, excluding /chat/completions.')
        prompt = args.prompt
        if prompt is None:
            prompt = sys.stdin.read() if args.prompt_file == '-' else Path(args.prompt_file).read_text(encoding='utf-8-sig')
        system = Path(args.system_file).read_text(encoding='utf-8-sig') if args.system_file else None
        body = build_request(config, prompt, system, args.image, not args.no_stream)
        output = None
        if args.output:
            # Reserve before charging an API request; never overwrite an old result.
            output = Path(args.output).open('x', encoding='utf-8')
        try:
            with httpx.Client(timeout=httpx.Timeout(float(config.get('timeout_s', 120)), connect=15),
                              headers={'Authorization': 'Bearer '+key}) as client:
                with client.stream('POST', base+'/chat/completions', json=body) as response:
                    if response.status_code >= 400:
                        # Do not print request bodies, headers, or provider error bodies containing secrets.
                        raise ValueError(f'HTTP {response.status_code}; check endpoint/model and supported request options. No automatic retry.')
                    if args.no_stream:
                        response.read()
                        chunks = [response.json()]
                    else:
                        def chunks_iter():
                            for data in sse_events(response.iter_lines()):
                                if data == '[DONE]':
                                    break
                                yield json.loads(data)
                        chunks = chunks_iter()
                    result = consume(chunks, args.show_reasoning)
                    if output:
                        json.dump(result, output, ensure_ascii=False, indent=2)
        finally:
            if output:
                output.close()
    except (ValueError, OSError, httpx.HTTPError) as error:
        message = str(error) if not isinstance(error, httpx.HTTPError) else type(error).__name__
        print('Error: '+message, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
