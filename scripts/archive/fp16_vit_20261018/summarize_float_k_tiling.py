#!/usr/bin/env python3
"""Summarize FP16 K tiling and paired complete ViT measurements."""
import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from compare_float_autotune import compare  # noqa: E402
from summarize_float_util import flops  # noqa: E402


def spread(values):
    return {"min": min(values), "median": statistics.median(values),
            "geomean": statistics.geometric_mean(values), "max": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    failures = list(summary["failures"])
    rows = summary["microbench"]
    assert len(rows) == len(summary["cache"]["profiles"]) == 21
    report = {
        "peak_tflops": 3, "fma_flops": 2,
        "speedup_vs_m128": spread([r["speedup"] for r in rows]),
        "speedup_vs_revision2": spread([r["speedup_vs_previous"] for r in rows]),
        "k_tiled_shapes": sum(int(r["winner"]["Ktile"]) > 0 for r in rows),
        "npu_reduce_shapes": sum(int(r["winner"]["npu_reduce"]) > 0 for r in rows),
        "vision": {},
    }
    for name in ["red", "blue", "red512"]:
        runs, features = [], []
        for index, variant in enumerate(["previous", "tuned", "tuned", "previous"]):
            directory = args.root / f"profile-{name}-{index}-{variant}"
            row = json.loads((directory / "profile.json").read_text())
            counters = json.loads((directory / "ioctl.json").read_text())
            assert row["features_bitwise_equal"] and row["feature_cache_hits"] == 0
            assert counters["errors"] == 0 and row["encode_calls"] == 5
            row.update(variant=variant, index=index, ioctl=counters)
            runs.append(row)
            features.append(np.fromfile(directory / "features.f32", dtype="<f4"))
        assert np.array_equal(features[0].view(np.uint32), features[3].view(np.uint32))
        assert np.array_equal(features[1].view(np.uint32), features[2].view(np.uint32))
        difference = compare(features[0], features[1])
        if difference["relative_l2"] >= .02 or difference["cosine"] <= .9999:
            failures.append(f"vision {name} versus revision2")
        m = runs[0]["grid"][1] * runs[0]["grid"][2]
        row = {"grid": runs[0]["grid"], "flops": flops(m),
               "features_vs_revision2": difference, "runs": runs}
        for variant in ["previous", "tuned"]:
            selected = [r for r in runs if r["variant"] == variant]
            samples = [s for r in selected for s in r["samples"]]
            assert len(samples) == 6
            total_ms = statistics.median(s["total_ms"] for s in samples)
            row[variant] = {
                "warm_median_ms": total_ms,
                "cold_samples_ms": [r["cold_ms"] for r in selected],
                "warm_samples_ms": [s["total_ms"] for s in samples],
                "mfu": flops(m) / (total_ms / 1000) / 3e12,
                "stages_median_ms": {
                    k: statistics.median(s["stages_ms"][k] for s in samples)
                    for k in samples[0]["stages_ms"]},
                "hardware_mean_ms": sum(r["ioctl"]["hardware_us"] for r in selected)
                                    / sum(r["encode_calls"] for r in selected) / 1000,
            }
        row["speedup"] = row["previous"]["warm_median_ms"] / row["tuned"]["warm_median_ms"]
        report["vision"][name] = row
    report["failures"] = failures
    report["passed"] = not failures
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "vision"}, indent=2))
    print(json.dumps({n: {"previous_ms": r["previous"]["warm_median_ms"],
                          "tuned_ms": r["tuned"]["warm_median_ms"],
                          "speedup": r["speedup"], "mfu": r["tuned"]["mfu"]}
                      for n, r in report["vision"].items()}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
