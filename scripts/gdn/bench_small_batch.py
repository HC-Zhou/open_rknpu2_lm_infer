#!/usr/bin/env python3
"""Paired C ABI replay using exported FP32 fixtures; no model or NPU calls."""
import argparse
from array import array
import ctypes as ct
import json
from pathlib import Path
import statistics
import time


class Parameters(ct.Structure):
    _fields_ = [(name, ct.c_size_t) for name in
                ("rows", "key_heads", "value_heads", "key_dim", "value_dim", "threads")] + [
                    (name, ct.POINTER(ct.c_float)) for name in
                    ("initial_state", "state", "query", "key", "value", "log_decay", "beta", "output")]


def buffer(values):
    return (ct.c_float * len(values))(*values)


def packed(values):
    return ct.string_at(values, ct.sizeof(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True, help="exported manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1, help="canonical states with one shared working state")
    parser.add_argument("--rows", help="optional comma-separated window sizes")
    parser.add_argument("--modes", default="copy_and_output,in_place_output,in_place_replay")
    args = parser.parse_args()
    libraries = [ct.CDLL(str(path.resolve())) for path in (args.baseline, args.candidate)]
    functions = [lib.rknpu2_gdn_run for lib in libraries]
    for function in functions:
        function.argtypes = [ct.POINTER(Parameters)]
        function.restype = None
    cases = json.loads(args.cases.read_text())
    results = []
    for case in cases:
        rows = case["rows"]
        if args.rows and rows not in [int(v) for v in args.rows.split(",")]:
            continue
        raw = array("f")
        raw.frombytes((args.cases.parent / case["input"]).read_bytes())
        sizes = [rows * 16 * 128, rows * 16 * 128, rows * 32 * 128,
                 rows * 32, rows * 32, 32 * 128 * 128]
        at, inputs = 0, []
        for size in sizes:
            inputs.append(buffer(raw[at:at + size]))
            at += size
        assert at == len(raw)
        q, k, v, g, beta, initial = inputs
        state, output = buffer(initial), (ct.c_float * len(v))()
        expected = (args.cases.parent / case["expected"]).read_bytes()
        params = Parameters(rows, 16, 32, 128, 128, args.threads,
                            initial, state, q, k, v, g, beta, output)
        for mode in args.modes.split(","):
            assert mode in ("copy_and_output", "in_place_output", "in_place_replay")
            params.initial_state = initial if mode == "copy_and_output" else state
            params.query = q if mode != "in_place_replay" else None
            params.output = output if mode != "in_place_replay" else None
            actual = []
            for function in functions:
                ct.memmove(state, initial, ct.sizeof(initial))
                function(ct.byref(params))
                result = (packed(output) if params.output else b"") + packed(state)
                oracle = expected if params.output else expected[len(v) * 4:]
                assert result == oracle, (case["name"], mode, "differs from pinned ggml fixture")
                actual.append(result)
            assert actual[0] == actual[1]
            if not case.get("timed", True):
                results.append(dict(case=case["name"], rows=rows, mode=mode, bit_exact=True))
                continue
            assert args.layers >= 1
            initial_bytes = packed(initial)
            bank = []
            canonical = [type(initial).from_buffer_copy(initial_bytes) for _ in range(args.layers)]
            scratch = type(state).from_buffer_copy(initial_bytes)
            for original in canonical:
                bank.append(Parameters(rows, 16, 32, 128, 128, args.threads,
                                       original, scratch if mode == "copy_and_output" else original,
                                       q if params.query else None, k, v, g, beta,
                                       output if params.output else None))

            def run_bank(function):
                nonlocal scratch
                for layer, entry in enumerate(bank):
                    entry.initial_state = canonical[layer]
                    entry.state = scratch if mode == "copy_and_output" else canonical[layer]
                    function(ct.byref(entry))
                    if mode == "copy_and_output":
                        canonical[layer], scratch = scratch, canonical[layer]

            loops = min(100, max(1, 128 // rows // args.layers))
            samples = [[], []]
            for function in functions:
                for _ in range(2):
                    run_bank(function)
            # Seven ABBA blocks, reseeding before each sample (outside timing).
            # In-place windows advance state within a sample, as in production.
            for _ in range(7):
                for side in (0, 1, 1, 0):
                    for original in canonical:
                        ct.memmove(original, initial, ct.sizeof(initial))
                    ct.memmove(scratch, initial, ct.sizeof(initial))
                    start = time.perf_counter_ns()
                    for _ in range(loops):
                        run_bank(functions[side])
                    samples[side].append((time.perf_counter_ns() - start) / loops / 1e6)
            means = [statistics.mean(side) for side in samples]
            results.append(dict(case=case["name"], rows=rows, mode=mode,
                                loops_per_sample=loops, samples_ms=samples, mean_ms=means,
                                mean_ms_per_layer=[m / args.layers for m in means],
                                latency_reduction_percent=100 * (1 - means[1] / means[0]),
                                bit_exact=True))
            print(case["name"], mode, means, flush=True)
    args.output.write_text(json.dumps(dict(threads=args.threads, layers=args.layers, scope="C ABI incl. per-head initial copy; excludes Rust validation and Python/ctypes adds call overhead", results=results), indent=2) + "\n")


if __name__ == "__main__":
    main()
