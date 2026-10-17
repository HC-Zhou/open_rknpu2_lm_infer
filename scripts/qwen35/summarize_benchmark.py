#!/usr/bin/env python3
"""Validate and summarize the same-board Qwen3.5/Qwen3 ABBA measurements."""
import argparse
import json
import math
from pathlib import Path
import statistics


def distribution(values):
    assert values and all(math.isfinite(value) and value > 0 for value in values)
    return dict(samples=values, mean=statistics.mean(values),
                stddev=statistics.stdev(values) if len(values) > 1 else 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    protocol = json.loads((root / "protocol.json").read_text())
    assert json.loads((root / "completed.json").read_text())["completed"]
    order = protocol["order"]
    processes = {name: order.count(name) for name in set(order)}
    assert processes["qwen35"] == processes["qwen3"]
    expected_samples = processes["qwen35"] * protocol["repetitions_per_case"]
    groups = {name: {} for name in order}
    identities = {}
    sessions = []
    clocks = []
    memory = {name: [] for name in order}
    for index, name in enumerate(order):
        label = f"{index + 1}-{name}"
        report = json.loads((root / (label + ".json")).read_text())
        assert report["prefill_without_lm_head"] and report["head_free_check"]["continuation_logits_exact"]
        clock = json.loads((root / (label + "-clocks.json")).read_text())
        assert clock["exit_code"] == 0 and not clock["restore_errors"]
        assert clock["clock_policy"] == dict(cpu_khz=protocol["cpu_khz"], dmc_controlled=False)
        for node, before in clock["before"].items():
            if node != "thermal":
                for field, value in before.items():
                    if field.endswith(("min_freq", "max_freq", "governor")):
                        assert clock["restored"][node][field] == value, (label, node, field)
        clocks.append(dict(run=label, restored=True,
            thermal_after=clock["after_command"]["thermal"]))
        stats = report["stats"]
        assert all(stats[key] == 0 for key in
                   ["tune_calls", "weight_reload_calls", "strategy_fallback_calls"])
        ids = report["synthetic_token_ids"]
        assert name not in identities or identities[name] == ids
        identities[name] = ids
        assert report["warmup"] == 1 and report["repetitions"] == protocol["repetitions_per_case"]
        memory[name].append(dict(peak_rss_kib=stats["peak_rss_kib"],
            weight_payload_bytes=stats["weight_payload_bytes"],
            workspace_payload_bytes=stats["workspace_payload_bytes"]))
        session = dict(run=label, cases=[])
        for case in report["cases"]:
            prompt = case["prompt"]
            assert [prompt, case["generate"]] in protocol["prompt_generate"] and case["depth"] == 0
            case_rows = dict(prompt=prompt)
            for phase in ["prefill", "decode", "prompt_gen"]:
                key = (prompt, phase)
                values = case[phase]["samples_tokens_per_second"]
                times = case[phase]["samples_ms"]
                assert len(values) == len(times) == protocol["repetitions_per_case"]
                group = groups[name].setdefault(key, dict(speeds=[], times=[], stages=[]))
                group["speeds"].extend(values)
                group["times"].extend(times)
                if phase != "prompt_gen":
                    stages = [sample[phase] for sample in case["stage_samples"]]
                    if phase == "prefill":
                        assert all(sample["head_ms"] == 0 for sample in stages)
                    group["stages"].extend(stages)
                case_rows[phase] = dict(mean_tokens_per_second=statistics.mean(values),
                                        mean_ms=statistics.mean(times))
            session["cases"].append(case_rows)
        sessions.append(session)
    rows = []
    for prompt, _ in protocol["prompt_generate"]:
        for phase in ["prefill", "decode", "prompt_gen"]:
            key = (prompt, phase)
            row = dict(prompt_tokens=prompt, decode_tokens=128, phase=phase)
            for name in ["qwen35", "qwen3"]:
                group = groups[name][key]
                assert len(group["speeds"]) == expected_samples
                row[name] = dict(tokens_per_second=distribution(group["speeds"]),
                                 milliseconds=distribution(group["times"]))
                if group["stages"]:
                    keys = set.intersection(*(set(sample) for sample in group["stages"]))
                    row[name]["stages_ms_mean"] = {key: statistics.mean(
                        sample[key] for sample in group["stages"]) for key in sorted(keys)}
            ratio = row["qwen35"]["tokens_per_second"]["mean"] / row["qwen3"]["tokens_per_second"]["mean"]
            row["qwen35_speed_ratio"] = ratio
            row["qwen35_throughput_change_percent"] = (ratio - 1) * 100
            rows.append(row)
    result = dict(protocol=protocol, samples_per_case_per_model=expected_samples,
        sample_structure=f"{processes['qwen35']} processes per model, {protocol['repetitions_per_case']} repetitions per process; descriptive sample standard deviation",
        results=rows, sessions=sessions, memory=memory,
        measured_tune_calls=0, measured_weight_reload_calls=0,
        measured_strategy_fallback_calls=0, clocks=clocks,
        fixed_ids_within_each_model=True, identical_ids_across_models=False)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("case | Qwen3.5 t/s | Qwen3 t/s | Qwen3.5 change")
    for row in rows:
        phase = row["phase"]
        label = f"pp{row['prompt_tokens']}" if phase == "prefill" else (
            f"tg128@{row['prompt_tokens']}" if phase == "decode" else f"pg{row['prompt_tokens']},128")
        a = row["qwen35"]["tokens_per_second"]
        b = row["qwen3"]["tokens_per_second"]
        print(f"{label} | {a['mean']:.3f} +/- {a['stddev']:.3f} | "
              f"{b['mean']:.3f} +/- {b['stddev']:.3f} | {row['qwen35_throughput_change_percent']:+.2f}%")


if __name__ == "__main__":
    main()
