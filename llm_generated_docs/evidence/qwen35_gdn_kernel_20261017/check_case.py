#!/usr/bin/env python3
"""Check a kernel's packed FP32 output, or run the standalone C++ reference."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np


def check(actual, expected):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    difference = actual.astype(np.float64) - expected.astype(np.float64)
    maximum = float(np.max(np.abs(difference)))
    relative_l2 = float(np.linalg.norm(difference.ravel()) /
                        max(np.linalg.norm(expected.astype(np.float64).ravel()), 1e-30))
    limit = 1e-5 + 1e-4 * float(np.max(np.abs(expected)))
    return dict(max_abs=maximum, relative_l2=relative_l2,
                bit_exact=actual.tobytes() == expected.tobytes(),
                passed=maximum <= limit and relative_l2 <= 1e-4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--actual", type=Path, help="output then final state, flat little-endian FP32")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    assert bool(args.runner) != bool(args.actual), "provide --runner or --actual"
    data = np.load(args.case)
    with tempfile.TemporaryDirectory(prefix="qwen35-gdn-") as temporary:
        if args.runner:
            path = Path(temporary)
            with (path / "input.f32").open("wb") as packed:
                for key in ["q", "k", "v", "g", "beta", "state_initial"]:
                    packed.write(data[key].astype("<f4").tobytes())
            args.actual = path / "output.f32"
            subprocess.run([str(args.runner), str(path / "input.f32"), str(args.actual),
                            str(data["q"].shape[0]), str(args.threads)], check=True)
        actual = np.fromfile(args.actual, dtype="<f4")
        split = data["output"].size
        result = dict(case=args.case.name,
            output=check(actual[:split].reshape(data["output"].shape), data["output"]),
            state=check(actual[split:].reshape(data["state_final"].shape), data["state_final"]))
    print(json.dumps(result))
    if not result["output"]["passed"] or not result["state"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
