#!/usr/bin/env python3
"""Exercise real MTP generation, prefix reuse, stop, cancellation and recovery.

Run via scripts/run_board.py; the production server does not inherit its lock.
vision_api.py and vision_tui.py separately cover image/M-RoPE inputs.
"""
import json
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

from vision_support import arguments, command, environment


def main():
    args = arguments(__doc__, photos_required=False)
    assert args.spec_type == 'draft-mtp' and args.spec_draft_n_max > 0, 'enable --spec-type draft-mtp'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    server_command = command(args) + ['serve', '--port', str(port)]
    report = {'command': server_command, 'cases': [], 'passed': False}
    log = (args.output/'server.log').open('w')
    server = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT,
                              env=environment(), close_fds=True)

    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')

    def body(messages, **options):
        return dict(model=model, messages=messages, enable_thinking=False,
                    temperature=0, max_tokens=32, **options)

    def open_request(payload):
        endpoint = '/v1/chat/completions' if 'messages' in payload else '/v1/completions'
        request = urllib.request.Request(base+endpoint,
                                         json.dumps(payload).encode(),
                                         {'Content-Type': 'application/json'})
        return opener.open(request, timeout=240)

    def request(name, payload, stream=False):
        payload = dict(payload, stream=stream)
        if stream:
            payload['stream_options'] = {'include_usage': True}
        start = time.monotonic()
        with open_request(payload) as response:
            if stream:
                text, usage, reason, done = '', None, None, False
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    raw = line[6:].strip()
                    if raw == b'[DONE]':
                        done = True
                        break
                    chunk = json.loads(raw)
                    assert 'error' not in chunk, chunk
                    for choice in chunk.get('choices', []):
                        text += choice.get('delta', {}).get('content', '') or ''
                        reason = choice.get('finish_reason') or reason
                    usage = chunk.get('usage') or usage
                assert done and usage is not None and reason is not None, name
            else:
                result = json.load(response)
                assert 'error' not in result, result
                choice = result['choices'][0]
                text = choice['message']['content'] if 'messages' in payload else choice['text']
                usage = result['usage']
                reason = result['choices'][0]['finish_reason']
        row = dict(name=name, text=text, usage=usage, finish_reason=reason,
                   elapsed_seconds=time.monotonic()-start)
        report['cases'].append(row)
        save()
        print(name, json.dumps(row, ensure_ascii=False), flush=True)
        return row

    try:
        deadline = time.monotonic()+300
        while True:
            if server.poll() is not None:
                raise RuntimeError(f'server exited: {server.returncode}')
            try:
                with opener.open(base+'/health', timeout=1) as response:
                    assert json.load(response)['status'] == 'ok'
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() > deadline:
                    raise TimeoutError('server startup')
                time.sleep(.2)
        with opener.open(base+'/v1/models', timeout=5) as response:
            model = json.load(response)['data'][0]['id']

        messages = [{'role': 'user', 'content': 'Explain merge sort in a few sentences.'}]
        baseline = request('greedy', body(messages))
        repeated = request('same-prefix-stream', body(messages), stream=True)
        assert baseline['text'] == repeated['text']
        assert baseline['usage']['completion_tokens'] == repeated['usage']['completion_tokens']

        history = messages + [{'role': 'assistant', 'content': baseline['text']},
                              {'role': 'user', 'content': 'What is its time complexity?'}]
        followup = request('multi-turn', body(history))
        assert followup['text'].strip()

        # The hybrid decoder cannot truncate to an older nonzero prefix. Use
        # a one-output seed, then extend its entirely retained input prefix.
        seed = [1, 42, 128, 5, 17]
        raw = dict(model=model, prompt=seed, temperature=0, max_tokens=1)
        request('reuse-seed', raw)
        raw.update(prompt=seed+[19], max_tokens=16)
        reused = request('reuse-extended-prefix', raw)
        fresh = request('reuse-cold-reference', raw)
        assert reused['usage']['prompt_tokens_details']['cached_tokens'] == len(seed)
        assert fresh['usage']['prompt_tokens_details']['cached_tokens'] == 0
        assert reused['text'] == fresh['text']

        zero = body(messages)
        zero['max_tokens'] = 0
        try:
            with open_request(zero):
                raise AssertionError('API accepted max_tokens=0')
        except urllib.error.HTTPError as error:
            result = json.load(error)
            error.close()
            assert error.code == 400 and result['error']['param'] == 'max_tokens', result
            report['cases'].append(dict(name='budget-0-rejected', status=error.code,
                                        error=result['error']))
            save()

        for count in (1, 2):
            payload = body(messages)
            payload['max_tokens'] = count
            row = request(f'budget-{count}', payload)
            assert row['usage']['completion_tokens'] == count, row

        # Obtain the stop marker from the actual tokenized output so that this
        # remains a state/streaming test rather than a model-quality assertion.
        marker = baseline['text'][:min(8, len(baseline['text']))]
        assert marker
        stopped = request('stop', body(messages, stop=[marker]), stream=True)
        assert stopped['finish_reason'] == 'stop' and marker not in stopped['text']

        random = body(messages)
        random.update(temperature=.8, top_k=40, top_p=.9, min_p=.05,
                      repeat_penalty=1.1, presence_penalty=.2,
                      frequency_penalty=.1, seed=1337)
        first = request('sampling', random)
        second = request('sampling-repeat', random, stream=True)
        assert first['text'] == second['text'] and first['text'].strip()

        # Close an active stream before its output budget is exhausted, then
        # queue another request. The worker must roll back/reset before reuse.
        interrupted = body([{'role': 'user', 'content': 'Count upward, one number per line.'}])
        interrupted.update(stream=True, max_tokens=256)
        with open_request(interrupted) as response:
            for line in response:
                if line.startswith(b'data: '):
                    chunk = json.loads(line[6:])
                    if any(c.get('delta', {}).get('content') for c in chunk.get('choices', [])):
                        break
            else:
                raise AssertionError('no streamed token before cancellation')
        recovered = request('after-disconnect', body(messages))
        assert recovered['text'] == baseline['text']
        report['passed'] = True
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.terminate()
                server.wait(timeout=10)
        log.close()
        report['server_exit_code'] = server.returncode
        save()
    assert server.returncode == 0


if __name__ == '__main__':
    main()
