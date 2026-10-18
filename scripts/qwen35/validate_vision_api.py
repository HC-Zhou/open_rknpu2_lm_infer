#!/usr/bin/env python3
"""Board-only API regression for inline images, history, streaming and recovery.
Run under scripts/run_board.py. Child server inherits the runner's lock fd.
"""
import argparse
import base64
import http.client
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixtures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", default=21235, type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    command = [str(args.binary), "--model", str(args.model), "--ctx-size", "512",
               "--ubatch-size", "128", "--threads", "4", "--cpu-mask", "0xf0",
               "--npu-mask", "7", "--no-tune", "serve", "--port", str(args.port)]
    report = dict(command=command, cases={})
    descriptors = tuple(int(os.environ[name]) for name in ["LLM_INFER_LOCK_FD"] if name in os.environ)
    environment = dict(os.environ, RUST_LOG="info", NO_COLOR="1")
    log = (args.output / "server.log").open("w")
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, pass_fds=descriptors)

    def request(messages, **extra):
        data = dict(model=args.model.name, messages=messages, temperature=0,
                    max_tokens=16, enable_thinking=False)
        data.update(extra)
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(data).encode(), {"Content-Type": "application/json"})
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as response:
            if data.get("stream"):
                text, usage, done = "", None, False
                for line in response:
                    if not line.startswith(b"data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == b"[DONE]":
                        done = True
                        break
                    chunk = json.loads(payload)
                    assert "error" not in chunk, chunk
                    for choice in chunk["choices"]:
                        text += choice["delta"].get("content", "")
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                assert done and usage is not None
                result = dict(text=text, usage=usage)
            else:
                body = json.load(response)
                result = dict(text=body["choices"][0]["message"]["content"], usage=body["usage"])
        result["elapsed_seconds"] = time.monotonic() - start
        return result

    def expect_invalid(name, messages, **extra):
        try:
            request(messages, **extra)
        except urllib.error.HTTPError as error:
            assert error.code == 400
            body = json.load(error)
            report["cases"][name] = dict(status=error.code, error=body["error"]["message"])
        else:
            raise AssertionError(f"{name}: invalid request succeeded")

    try:
        deadline = time.monotonic() + 180
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited: {server.returncode}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=1) as response:
                    assert json.load(response)["status"] == "ok"
                break
            except (OSError, urllib.error.URLError):
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup")
                time.sleep(.2)
        messages = {name: json.loads((args.fixtures / f"messages-{name}.json").read_text()) for name in ["red", "blue", "two"]}
        for name in ["red", "blue"]:
            result = request(messages[name])
            assert name in result["text"].lower(), result
            report["cases"][name] = result
        # Same dimensions and identical text/token placeholders, different image.
        square = args.fixtures / "blue-square.png"
        blue_square = json.loads(json.dumps(messages["red"]))
        blue_square[0]["content"][1]["image_url"]["url"] = "data:image/png;base64," + base64.b64encode(square.read_bytes()).decode()
        for name, content in [("same-shape-red", messages["red"]), ("same-shape-blue", blue_square)]:
            result = request(content)
            assert ("red" if name.endswith("red") else "blue") in result["text"].lower(), result
            report["cases"][name] = result
        assert report["cases"]["same-shape-red"]["usage"]["prompt_tokens"] == report["cases"]["same-shape-blue"]["usage"]["prompt_tokens"]
        result = request(messages["two"], stream=True, stream_options={"include_usage": True})
        assert all(color in result["text"].lower() for color in ["red", "blue"]), result
        report["cases"]["two-stream"] = result
        history = messages["red"] + [{"role": "assistant", "content": "red"}, {"role": "user", "content": "What color was in the previous picture? Answer one word."}]
        result = request(history)
        assert "red" in result["text"].lower(), result
        report["cases"]["history"] = result
        expect_invalid("context-overflow", messages["red"], max_tokens=500)
        expect_invalid("remote-image", [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.org/image.png"}}]}])
        expect_invalid("bad-image", [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}])
        # Read only the immediate SSE role event and disconnect during prefill.
        connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=120)
        data = dict(model=args.model.name, messages=messages["two"], temperature=0, max_tokens=128, enable_thinking=False, stream=True)
        connection.request("POST", "/v1/chat/completions", json.dumps(data), {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        assert response.readline().startswith(b"data:")
        response.close()
        connection.close()
        result = request(messages["blue"])
        assert "blue" in result["text"].lower(), result
        report["cases"]["after-disconnect"] = result
        result = request([{"role": "user", "content": "Reply with OK."}])
        assert "ok" in result["text"].lower(), result
        report["cases"]["text-after-images"] = result
        report["passed"] = True
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.terminate()
                server.wait(timeout=10)
        log.close()
        report["server_exit_code"] = server.returncode
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
