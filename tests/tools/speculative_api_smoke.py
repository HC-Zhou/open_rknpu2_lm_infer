"""Exercise the production CLI and OpenAI API with a saved real-text request."""
import argparse
import http.client
import json
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--draft-model", type=Path, required=True)
parser.add_argument("--tune-cache", type=Path, required=True)
parser.add_argument("--inputs", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
item = json.loads(args.inputs.read_text())["prompts"][0]
prompt = item["tokens"]
prompt_file = args.output / "prompt.txt"
prompt_file.write_text(item["formatted"])
reference = {}
report = {"checks": [], "status": "RUNNING"}


def record(name, mode):
    report["checks"].append({"name": name, "mode": mode})
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS {mode} {name}", flush=True)


def command(mode):
    result = [str(args.binary), "-m", str(args.model), "--tune-cache", str(args.tune_cache)]
    if mode.startswith("dflash"):
        result += ["--spec-type", "draft-dflash", "-md", str(args.draft_model), "--spec-draft-n-max", mode[-1]]
    elif mode == "mtp1":
        result += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]
    return result


def compare(name, mode, response):
    value = (response["choices"][0], response["usage"]["completion_tokens"])
    if mode == "ar":
        reference[name] = value
    else:
        assert value == reference[name], (mode, name, value, reference[name])
    record(name, mode)


for mode in ("ar", "dflash3", "dflash7", "mtp1"):
    cli = subprocess.run(command(mode) + ["run", "-f", str(prompt_file), "--temp", "0", "-n", "16", "--stats"], capture_output=True, text=True, check=True)
    (args.output / (mode + "-cli.log")).write_text(cli.stderr)
    start = cli.stderr.rfind("\n{\n")
    assert start >= 0, cli.stderr[-2000:]
    stats, _ = json.JSONDecoder().raw_decode(cli.stderr[start+1:])
    expected_type = "draft-dflash" if mode.startswith("dflash") else "draft-mtp" if mode == "mtp1" else "none"
    assert stats["stats"]["spec_type"] == expected_type
    assert stats["completion_tokens"] == 16
    if mode == "ar":
        reference["cli"] = cli.stdout
    else:
        assert cli.stdout == reference["cli"], (mode, cli.stdout, reference["cli"])
        assert stats["timings"]["spec_rounds"] > 0
    record("cli", mode)
    if mode == "mtp1":
        continue
    with socket.socket() as selection:
        selection.bind(("127.0.0.1", 0))
        port = selection.getsockname()[1]
    log = (args.output / (mode + "-server.log")).open("w")
    process = subprocess.Popen(command(mode) + ["serve", "--host", "127.0.0.1", "--port", str(port)], stdout=log, stderr=log)
    try:
        deadline = time.monotonic() + 180
        while True:
            assert process.poll() is None, (mode, "server exited")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                    model_id = json.load(response)["model"]
                break
            except OSError:
                assert time.monotonic() < deadline, (mode, "server readiness timeout")
                time.sleep(.2)

        def request(tokens=prompt, maximum=16, **extra):
            body = {"model":model_id, "prompt":tokens, "temperature":0, "seed":42, "max_tokens":maximum, **extra}
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=180) as response:
                return json.load(response)

        normal = request()
        compare("completion", mode, normal)
        # Consume an actual prompt prefix, then extend it by its final real token.
        request(prompt[:-1], 1)
        reused = request()
        assert reused["usage"]["prompt_tokens_details"]["cached_tokens"] == len(prompt)-1
        compare("prefix-reuse", mode, reused)
        stop = reference["completion"][0]["text"][16:24]
        assert stop
        compare("stop", mode, request(stop=stop))

        def stream(cancel=False):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
            body = {"model":model_id,"prompt":prompt,"temperature":0,"seed":42,"max_tokens":128 if cancel else 16,"stream":True,"stream_options":{"include_usage":True}}
            connection.request("POST", "/v1/completions", json.dumps(body), {"Content-Type":"application/json"})
            response = connection.getresponse()
            assert response.status == 200, response.read()
            text, usage = "", None
            while True:
                line = response.readline()
                if not line:
                    break
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                chunk = json.loads(data)
                for choice in chunk["choices"]:
                    text += choice.get("text", "")
                usage = chunk.get("usage") or usage
                if cancel and text:
                    break
            response.close()
            connection.close()
            return text, usage

        streamed, usage = stream()
        assert streamed == normal["choices"][0]["text"]
        assert usage["completion_tokens"] == 16
        record("stream", mode)
        stream(cancel=True)
        compare("cancel-recovery", mode, request(maximum=8))
    finally:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=10)
        log.close()
report["status"] = "PASS"
(args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
