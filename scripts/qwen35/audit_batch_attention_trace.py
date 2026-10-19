#!/usr/bin/env python3
"""Inspect the first divergent Attention row and its dynamic A8 propagation."""
import argparse
import json
from pathlib import Path
import numpy as np


def metric(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    difference = a - b
    return {"elements": a.size, "different": int(np.count_nonzero(difference)),
            "nrmse": float(np.linalg.norm(difference) / max(np.linalg.norm(a), 1e-30)),
            "max_abs": float(np.max(np.abs(difference)))}


def quantize(a):
    scale = np.float32(np.max(np.abs(a)) / np.float32(127))
    inverse = np.float32(1) / scale
    scaled = np.float32(a * inverse)
    # W4A8 quant16 uses FCVTA: ties away from zero, matching std::round.
    codes = np.copysign(np.floor(np.abs(scaled) + np.float32(.5)), scaled)
    return np.clip(codes, -127, 127).astype(np.int16), float(scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def read(kind, layer, stage, position=5):
        return np.fromfile(args.trace / kind / f"p{position}-l{layer}-{stage}.f32", dtype="<f4")

    report = {"early_gdn": [], "layer3": {}}
    for layer in range(3):
        for stage in ["input", "gdn_qkv", "attn", "o", "post", "middle", "hidden"]:
            a = read("serial", layer, stage)
            b = read("verify", layer, stage)[:a.size]
            report["early_gdn"].append({"layer": layer, "stage": stage, **metric(a, b)})
    for stage in ["input", "query", "key", "value", "attn", "o", "post", "middle", "hidden"]:
        a = read("serial", 3, stage)
        b = read("verify", 3, stage)[:a.size]
        row = metric(a, b)
        if stage in ["input", "attn", "post", "middle"]:
            qa, sa = quantize(a)
            qb, sb = quantize(b)
            row.update(a8_scale_serial=sa, a8_scale_batch=sb,
                       a8_code_changes=int(np.count_nonzero(qa - qb)))
        report["layer3"][stage] = row
    for stage in ["query", "key", "value"]:
        a, b = read("serial", 3, stage, 0), read("verify", 3, stage, 0)
        report["layer3"][f"prompt_{stage}"] = metric(a, b)
    raw_ar = np.fromfile(args.probe / "lut1-one6.f32", dtype="<f4")
    raw_batch = np.fromfile(args.probe / "lut1-one13.f32", dtype="<f4")
    ar = read("serial", 3, "attn")
    batch = read("verify", 3, "attn")[:ar.size]
    inferred_gate = np.float32(ar / raw_ar)
    report["raw_attention"] = metric(raw_ar, raw_batch)
    report["gate_consistency"] = {
        "inferred_gate_min": float(inferred_gate.min()), "inferred_gate_max": float(inferred_gate.max()),
        "batch_prediction": metric(batch, np.float32(raw_batch * inferred_gate)),
        "note": "Gate inferred from AR output, not a separately captured gate tensor."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["layer3"], indent=2))


if __name__ == "__main__":
    main()
