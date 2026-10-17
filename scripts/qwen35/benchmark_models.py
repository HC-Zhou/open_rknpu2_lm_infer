#!/usr/bin/env python3
"""Compare two models with cached strategies and continuous prefill/decode."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--qwen35", type=Path, required=True)
    parser.add_argument("--qwen3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--paired", action="store_true", help="one process per model, in Qwen3.5/Qwen3 order")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve().parents[1] / "run_board.py"
    models = {"qwen35": args.qwen35.resolve(), "qwen3": args.qwen3.resolve()}
    caches = {name: args.output / (name + "-cache") for name in models}
    for cache in caches.values():
        cache.mkdir(exist_ok=True)
    common = ["--ctx-size", "4224", "--ubatch-size", "128", "--threads", "4",
              "--cpu-mask", "0xf0", "--npu-mask", "7", "--attention-tile", "512"]
    protocol = dict(order=["qwen35", "qwen3"] if args.paired else ["qwen35", "qwen3", "qwen3", "qwen35"],
        warmup_per_case_per_process=1, repetitions_per_case=args.repetitions,
        prompt_generate=[[512, 128], [4096, 128]], context=4224, ubatch=128,
        cpu_mask="0xf0", threads=4, npu_mask=7, cpu_khz=2256000, npu_hz=1000000000,
        dmc_controlled=False, tuning="explicit tune; measured calls only load cached strategies",
        prefill_without_lm_head=True,
        timing="head-free prefill; decode includes output head; excludes load, prepare, tokenization, sampling and stats snapshots",
        token_source="per-model MT19937(42) modulo its vocabulary; fixed IDs per model",
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        models={name: dict(path=str(path), config_sha256=hashlib.sha256(
            (path / "config.json").read_bytes()).hexdigest()) for name, path in models.items()})
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    runs = [] if args.reuse_cache else [("tune-" + name, name, True) for name in models]
    runs += [(f"{i + 1}-{name}", name, False) for i, name in enumerate(protocol["order"])]
    completed = []
    for label, name, tuning in runs:
        command = [str(args.binary.resolve()), "--model", str(models[name]), *common,
                   "--tune-cache", str(caches[name])]
        report = args.output / (label + ".json")
        if tuning:
            command += ["tune", "--batches", "1,128", "--report", str(report)]
        else:
            command += ["bench", "-p", "0", "-n", "0", "-pg", "512,128",
                        "-pg", "4096,128", "-r", str(args.repetitions), "--report", str(report)]
        clock = args.output / (label + "-clocks.json")
        wrapped = [sys.executable, str(runner), "--wait-lock", "--cpu-khz", "2256000",
                   "--skip-dmc", "--report", str(clock), "--", *command]
        tick = time.monotonic()
        print(json.dumps(dict(started=label, command=wrapped)), flush=True)
        with (args.output / (label + ".log")).open("w") as log:
            status = subprocess.run(wrapped, stdout=log, stderr=subprocess.STDOUT).returncode
        measured = json.loads(clock.read_text())
        if measured["restore_errors"] or status:
            raise SystemExit(f"{label} failed: status={status}, restore={measured['restore_errors']}")
        for node, before in measured["before"].items():
            if node != "thermal":
                for field, value in before.items():
                    if field.endswith(("min_freq", "max_freq", "governor")):
                        assert measured["restored"][node][field] == value, (label, node, field)
        result = json.loads(report.read_text())
        if not tuning:
            assert result["prefill_without_lm_head"] and result["head_free_check"]["continuation_logits_exact"]
            stats = result["stats"]
            if any(stats[key] for key in ["tune_calls", "weight_reload_calls", "strategy_fallback_calls"]):
                raise SystemExit(f"{label}: uncached strategy or weight reload in measured process")
        completed.append(dict(label=label, seconds=time.monotonic() - tick,
                              report=str(report), clocks=str(clock)))
        (args.output / "progress.json").write_text(json.dumps(completed, indent=2) + "\n")
        print(json.dumps(completed[-1]), flush=True)
    (args.output / "completed.json").write_text(json.dumps(dict(completed=True, runs=completed), indent=2) + "\n")


if __name__ == "__main__":
    main()
