#!/usr/bin/env python3
"""Real-photo OpenAI server tests: image history, replacement and SSE."""
import base64
import json
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

from vision_support import arguments, command, environment


def main():
    args = arguments(__doc__)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    server_command = command(args) + ['serve', '--host', '127.0.0.1', '--port', str(port)]
    report = {'command': server_command, 'cases': [], 'passed': False}
    log = (args.output/'server.log').open('w')
    server = subprocess.Popen(server_command, stdout=log, stderr=subprocess.STDOUT,
                              env=environment(), close_fds=True)

    def content(name, text):
        path = args.photos/name
        mime = 'image/png' if path.suffix == '.png' else 'image/jpeg'
        return {'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()}},
            {'type': 'text', 'text': text}]}

    def save():
        (args.output/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')

    def request(name, messages, checks=(), stream=False):
        body = {'model': model, 'messages': messages, 'temperature': 0,
                'max_tokens': 64, 'enable_thinking': False, 'stream': stream}
        if stream:
            body['stream_options'] = {'include_usage': True}
        req = urllib.request.Request(base+'/v1/chat/completions', json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        start = time.monotonic()
        with opener.open(req, timeout=240) as response:
            if stream:
                text, usage, done = '', None, False
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
                    if chunk.get('usage'):
                        usage = chunk['usage']
                assert done and usage is not None, name
            else:
                result = json.load(response)
                assert 'error' not in result, result
                text = result['choices'][0]['message']['content']
                usage = result['usage']
        row = {'name': name, 'text': text, 'usage': usage, 'stream': stream,
               'elapsed_seconds': time.monotonic()-start,
               'messages': len(messages), 'semantic_checks': []}
        report['cases'].append(row)
        assert text.strip() and usage['completion_tokens'] > 0, row
        for choices in checks:
            passed = any(word in text.lower() for word in choices)
            row['semantic_checks'].append({'any_of': choices, 'passed': passed})
        save()
        print(name, text, flush=True)
        # Preserve the transcript even when recognition quality is disappointing.
        return {'role': 'assistant', 'content': text}

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
        bus = content('bus.jpg', 'What is the large vehicle behind the people? Answer in one brief sentence.')
        history = [bus]
        history.append(request('bus', history, [('bus', 'coach')]))
        history.append({'role': 'user', 'content': 'What colors is that vehicle? Answer briefly.'})
        history.append(request('bus-followup', history, [('blue',)]))
        history.append(content('cats.png', 'Now look at this new photograph. What animals are there, and how many?'))
        history.append(request('new-photo-in-history', history, [('cat',), ('two', '2', 'pair')]))
        history.append({'role': 'user', 'content': 'What color is the surface those animals are lying on? Answer briefly.'})
        request('cats-followup-stream', history, [('pink', 'red')], stream=True)
        request('replace-photo-same-prompt', [content('cats.png', 'Describe the main subject of this photo in one sentence.')], [('cat',)])
        request('replace-photo-same-prompt-bus', [content('bus.jpg', 'Describe the main subject of this photo in one sentence.')], [('bus', 'coach')])
        first = content('bus.jpg', '')['content'][0]
        second = content('cats.png', '')['content'][0]
        request('two-photos-stream', [{'role': 'user', 'content': [first, second,
            {'type': 'text', 'text': 'Which photo shows animals and which photo shows public transport? One brief sentence.'}]}],
            [('cat', 'animal'), ('bus', 'transport')], stream=True)
        request('chinese-bus', [content('bus.jpg', '这张图里主要是什么交通工具？请用一句中文回答。')], [('公交', '巴士', '公共汽车', '大巴')])
        request('text-after-images', [{'role': 'user', 'content': 'Reply exactly OK.'}], [('ok',)])
        report['passed'] = all(check['passed'] for row in report['cases'] for check in row['semantic_checks'])
        save()
        assert report['passed'], 'review recognition answers in report.json'
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


if __name__ == '__main__':
    main()
