#!/usr/bin/env python3
"""Summarize C7 ViT stage timings and resident-device FP16 probes."""
import argparse
import json
from pathlib import Path
import statistics


def flops(m):
    return (2*m*1536*1024 + 24*(24*m*1024**2 + 4*m*m*1024)
            + 2*(m//4)*(4096**2 + 4096*2560))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peak-tflops", type=float, default=3)
    args = parser.parse_args()
    report = {"peak_tflops": args.peak_tflops, "fma_flops": 2, "mm": [], "vision": {}}
    for path in sorted(args.root.glob("mm-*.log")):
        row = json.loads(next(line[6:] for line in path.read_text().splitlines() if line.startswith("PROBE ")))
        counters = json.loads(path.with_name(path.stem + "-ioctl.json").read_text())
        assert counters["errors"] == 0 and counters["hardware_us"] > 0
        row["device_median_us"] = statistics.median(row["device_us"])
        row["host_median_us"] = statistics.median(row["host_us"])
        row["device_tflops"] = row["flops"] / row["device_median_us"] / 1e6
        row["host_tflops"] = row["flops"] / row["host_median_us"] / 1e6
        row["hardware_mean_us"] = counters["hardware_us"] / counters["submits"]
        row["hardware_tflops"] = row["flops"] / row["hardware_mean_us"] / 1e6
        row["ioctl"] = counters
        report["mm"].append(row)
    for name in ["red", "blue", "red512"]:
        row = json.loads((args.root / name / "profile.json").read_text())
        counters = json.loads((args.root / name / "ioctl.json").read_text())
        assert counters["errors"] == 0 and row["features_bitwise_equal"] and row["feature_cache_hits"] == 0
        m = row["grid"][1]*row["grid"][2]
        row["flops"] = flops(m)
        row["warm_median_ms"] = statistics.median(s["total_ms"] for s in row["samples"])
        row["stages_median_ms"] = {k: statistics.median(s["stages_ms"][k] for s in row["samples"])
                                   for k in row["samples"][0]["stages_ms"]}
        row["mfu"] = row["flops"] / (row["warm_median_ms"]/1000) / (args.peak_tflops*1e12)
        row["hardware_mean_ms"] = counters["hardware_us"] / row["encode_calls"] / 1000
        row["hardware_model_tflops"] = row["flops"] / (row["hardware_mean_ms"]/1000) / 1e12
        row["ioctl"] = counters
        report["vision"][name] = row
    report["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "mm_cases": len(report["mm"]),
                      "vision": {n: {k: row[k] for k in ["grid", "warm_median_ms", "mfu"]}
                                 for n, row in report["vision"].items()}}, indent=2))


if __name__ == "__main__":
    main()
