#!/usr/bin/env python3
"""Summarize floating-MM ABBA calls, cache reuse, ViT features and logits."""
import argparse
import json
from pathlib import Path
import re
import statistics

import numpy as np


def fields(line):
    return dict(re.findall(r"(\w+)=([^ ]+)", line))


def compare(a, b):
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    identical = bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))
    a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return {
        "values": a.size,
        "relative_l2": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
        "cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
        "max_abs": float(np.abs(a - b).max()),
        "bitwise_equal": identical,
    }


def logp(a):
    a = a.astype(np.float64) - a.max()
    return a - np.log(np.exp(a).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--vision-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, ref = args.root, args.vision_reference
    report = {"microbench": [], "vision": {}, "model": {}, "failures": []}
    for path in sorted(root.glob("*16-*.log")):
        if not re.fullmatch(r"(?:fp16|bf16)-\d+-\d+-\d+\.log", path.name):
            continue
        lines = path.read_text().splitlines()
        winner = fields(next(line for line in lines if line.startswith("winner ")))
        samples = [fields(line) for line in lines if line.startswith("ABBA ")]
        base = [float(s["us"]) for s in samples if s["path"] == "baseline"]
        tuned = [float(s["us"]) for s in samples if s["path"] == "tuned"]
        assert len(base) == len(tuned) == 6
        result = {"case": path.stem, "winner": winner, "samples": samples,
                  "baseline_us": statistics.median(base), "tuned_us": statistics.median(tuned)}
        result["speedup"] = result["baseline_us"] / result["tuned_us"]
        prior = [fields(line) for line in lines if line.startswith("PRIOR_ABBA ")]
        if prior:
            assert len(prior) == 12
            old = [float(s["us"]) for s in prior if s["path"] == "previous"]
            new = [float(s["us"]) for s in prior if s["path"] == "tuned"]
            result.update(previous_us=statistics.median(old), refined_us=statistics.median(new),
                          speedup_vs_previous=statistics.median(old)/statistics.median(new),
                          previous_samples=prior)
        result["paired_speedups"] = [
            statistics.mean(float(s["us"]) for s in samples if s["block"] == str(i) and s["path"] == "baseline") /
            statistics.mean(float(s["us"]) for s in samples if s["block"] == str(i) and s["path"] == "tuned")
            for i in range(3)
        ]
        delta = next(line for line in lines if line.startswith("full_output_baseline_delta="))
        assert delta.endswith(" PASS")
        result["full_output_max_abs_delta"] = float(fields(delta)["full_output_baseline_delta"])
        report["microbench"].append(result)
    cache = json.loads((root / "load-only.json").read_text())
    report["cache"] = cache
    if cache["stats"]["tune_calls"] != 0 or cache["stats"]["tune_cache_hits"] != len(cache["profiles"]) or any(p["total_us"] <= 0 for p in cache["profiles"]):
        report["failures"].append("load-only cache reuse")
    for name in ["red", "blue"]:
        load = lambda directory: np.fromfile(directory / "features.f32", dtype="<f4")
        baseline, tuned = load(root / f"baseline-{name}"), load(root / f"tuned-{name}")
        reference = load(ref / f"reference-{name}")
        before = load(ref / f"board-{name}")
        result = {
            "pre_autotune_vs_baseline": compare(before, baseline),
            "baseline_vs_tuned": compare(baseline, tuned),
            "reference_vs_tuned": compare(reference, tuned),
            "baseline_encoder": json.loads((root / f"baseline-{name}" / "encoder.json").read_text()),
            "tuned_encoder": json.loads((root / f"tuned-{name}" / "encoder.json").read_text()),
        }
        if result["reference_vs_tuned"]["cosine"] <= .9999 or result["reference_vs_tuned"]["relative_l2"] >= .02:
            report["failures"].append(f"vision {name}")
        report["vision"][name] = result
    for name in ["red", "blue", "two"]:
        load = lambda directory: np.fromfile(directory / name / "logits.f32", dtype="<f4").reshape(-1, 248320)
        baseline, tuned = load(root / "baseline-model"), load(root / "tuned-model")
        result = compare(baseline, tuned)
        result["rows"] = []
        for a, b in zip(baseline, tuned):
            la, lb = logp(a), logp(b)
            row = {"baseline_top1": int(a.argmax()), "tuned_top1": int(b.argmax()),
                   "kl": float((np.exp(la) * (la - lb)).sum())}
            result["rows"].append(row)
            if row["baseline_top1"] != row["tuned_top1"] or row["kl"] >= .01:
                report["failures"].append(f"logits {name}")
        if result["relative_l2"] >= .02:
            report["failures"].append(f"logits L2 {name}")
        report["model"][name] = result
    report["passed"] = not report["failures"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "failures": report["failures"],
                      "microbench": [{k: row[k] for k in ["case", "baseline_us", "tuned_us", "speedup"]} for row in report["microbench"]],
                      "vision": report["vision"], "model": report["model"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
