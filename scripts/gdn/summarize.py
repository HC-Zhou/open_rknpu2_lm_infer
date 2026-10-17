#!/usr/bin/env python3
"""Summarize paired FP32 GDN module calls against the real ggml CPU op."""
import argparse
import json
from pathlib import Path
import statistics


def samples(values):
    assert values and all(value > 0 for value in values)
    return dict(mean=statistics.mean(values), stddev=statistics.stdev(values), samples=values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    clocks = []
    for threads in [1, 4]:
        report = json.loads((args.directory / f"final-t{threads}/comparison.json").read_text())
        clock = json.loads((args.directory / f"final-t{threads}-clocks.json").read_text())
        assert clock["exit_code"] == 0 and not clock["restore_errors"]
        assert report["threads"] == threads and report["state_dtype"] == "FP32"
        for node, before in clock["before"].items():
            if node != "thermal":
                for field, value in before.items():
                    if field.endswith(("min_freq", "max_freq", "governor")):
                        assert clock["restored"][node][field] == value
        clocks.append(dict(threads=threads, restored=True))
        for case in report["cases"]:
            for mode in ["unfused", "fused"]:
                assert case[mode]["output"]["passed"] and case[mode]["state"]["passed"]
            own = samples(case["cpp_fused_ms"])
            ref = samples(case["ggml_ms"])
            records.append(dict(threads=threads, tokens=case["rows"],
                cpp_fused_ms=own, ggml_ms=ref,
                cpp_unfused_ms=samples(case["cpp_unfused_ms"]),
                latency_change_percent=100 * (own["mean"] / ref["mean"] - 1),
                throughput_speedup=ref["mean"] / own["mean"],
                output_max_abs=case["fused"]["output"]["max_abs"],
                state_max_abs=case["fused"]["state"]["max_abs"]))
    result = dict(scope="FP32 GDN recurrence module, not whole mixer/model",
        wrapper="real Rust safe interface and C ABI; includes validation, worker state copy and token-major output",
        reference="actual pinned ggml CPU graph_compute; Q/K expanded to preserve HF head mapping",
        key_heads=16, value_heads=32, key_dim=128, value_dim=128,
        samples_per_case=7, cpu_khz=2256000, cpu_mask="0xf0", dmc_controlled=False,
        clock_restoration=clocks,
        all_outputs_and_states_bit_exact=all(row["output_max_abs"] == row["state_max_abs"] == 0 for row in records),
        records=records)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("threads tokens | C++ ms | ggml ms | latency change")
    for row in records:
        print(f"{row['threads']:>7} {row['tokens']:>6} | {row['cpp_fused_ms']['mean']:.6f} | "
              f"{row['ggml_ms']['mean']:.6f} | {row['latency_change_percent']:+.2f}%")


if __name__ == "__main__":
    main()
