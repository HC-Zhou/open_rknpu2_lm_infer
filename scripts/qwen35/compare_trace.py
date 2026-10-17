#!/usr/bin/env python3
"""Compare equal-position Qwen3.5 decoder traces from the board and oracle."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = []
    for layer in range(32):
        for stage in ["input", "gdn_qkv", "attn", "o", "post", "middle", "hidden"]:
            name = f"p{args.position}-l{layer}-{stage}.f32"
            ref, actual = args.reference / name, args.actual / name
            if not (ref.exists() and actual.exists()):
                continue
            a, b = np.fromfile(ref, dtype="<f4"), np.fromfile(actual, dtype="<f4")
            if a.shape != b.shape:
                report.append(dict(layer=layer, stage=stage, shape_match=False,
                                   reference_elements=len(a), actual_elements=len(b)))
                continue
            difference = a.astype(np.float64) - b
            nrmse = float(np.sqrt(np.square(difference).sum() / max(np.square(a.astype(np.float64)).sum(), 1e-30)))
            row = dict(layer=layer, stage=stage, elements=len(a), nrmse=nrmse,
                       max_abs=float(np.abs(difference).max()))
            report.append(row)
            print(json.dumps(row), flush=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
