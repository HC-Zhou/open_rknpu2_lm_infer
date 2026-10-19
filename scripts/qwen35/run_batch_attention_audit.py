#!/usr/bin/env python3
"""Compare a native Attention carrier fix with the unchanged AR implementation.

Run this script through scripts/run_board.py on RK3588. The probe executable
accepts TRACE_ROOT OUTPUT_DIR START and emits compact FP32 output files.
"""
import argparse
from array import array
import json
import math
from pathlib import Path
import subprocess


def difference(reference, actual):
    a_bytes, b_bytes = reference.read_bytes(), actual.read_bytes()
    if len(a_bytes) != len(b_bytes):
        raise AssertionError(f"output sizes differ: {reference}, {actual}")
    if a_bytes == b_bytes:
        return {"elements": len(a_bytes) // 4, "different": 0, "nrmse": 0, "max_abs": 0}
    a, b = array("f"), array("f")
    a.frombytes(a_bytes)
    b.frombytes(b_bytes)
    delta = [float(x) - float(y) for x, y in zip(a, b)]
    return {"elements": len(a), "different": sum(x != y for x, y in zip(a, b)),
            "nrmse": math.sqrt(sum(x * x for x in delta) / max(sum(x * x for x in a), 1e-30)),
            "max_abs": max(map(abs, delta))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, nargs="+",
                        default=[5, 27, 31, 253, 255, 509, 511, 1021, 4093, 32759])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for start in args.starts:
        case = args.output / f"p{start}"
        for label, executable in [("before", args.before), ("after", args.after)]:
            directory = case / label
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "probe.log").open("w") as log:
                subprocess.run([str(executable), str(args.trace), str(directory), str(start)],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        before, after = case / "before", case / "after"
        checks = {}
        for lut in [0, 1]:
            prefix = f"lut{lut}"
            checks[f"{prefix}_old_ar_vs_batch"] = difference(
                before / f"{prefix}-serial-all.f32", after / f"{prefix}-batch-all.f32")
            checks[f"{prefix}_old_ar_vs_new_ar"] = difference(
                before / f"{prefix}-serial-all.f32", after / f"{prefix}-serial-all.f32")
            for mode, patterns in [(1, [0]), (2, [0, 1, 2])]:
                for pattern in patterns:
                    name = f"{prefix}-mode{mode}-pattern{pattern}.f32"
                    checks[name] = difference(before / name, after / name)
        failures = [name for name, result in checks.items() if result["different"]]
        row = {"start": start, "real_trace": start == 5, "checks": checks, "pass": not failures,
               "before_lut1_ar_vs_batch": difference(
                   before / "lut1-serial-all.f32", before / "lut1-batch-all.f32"),
               "before_lut0_ar_vs_batch": difference(
                   before / "lut0-serial-all.f32", before / "lut0-batch-all.f32")}
        report.append(row)
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"start": start, "pass": not failures, "failures": failures,
                          "old_ar_vs_new_batch": checks["lut1_old_ar_vs_batch"]}), flush=True)
        if failures:
            raise AssertionError(f"start {start}: {failures}")


if __name__ == "__main__":
    main()
