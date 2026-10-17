#!/usr/bin/env python3
"""Compare native end-to-end PPL with the paired exported C7 FP32 reference."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.native.read_text())
    reference = json.loads(args.reference.read_text())
    assert native["completed"]
    assert hashlib.sha256(args.tokens.read_bytes()).hexdigest() == reference["tokens_sha256"]
    for field in ["windows", "window_size", "scored_target_positions"]:
        assert native["protocol"][field] == reference[field], field
    assert native["scored_tokens"] == reference["scored_tokens"]
    assert len(native["chunks"]) == reference["windows"]
    paired = []
    for index, (row, expected) in enumerate(zip(native["chunks"], reference["token_nll"])):
        actual = row["token_nll"]
        assert row["chunk"] == index + 1 and len(actual) == len(expected)
        assert all(math.isfinite(value) for value in actual)
        difference = [a - b for a, b in zip(actual, expected)]
        paired.append(dict(window=index + 1, native_nll=row["nll"],
            reference_nll=sum(expected) / len(expected),
            mean_nll_difference=sum(difference) / len(difference),
            mean_absolute_token_nll_difference=sum(map(abs, difference)) / len(difference),
            max_absolute_token_nll_difference=max(map(abs, difference))))
    relative = native["ppl"] / reference["ppl"] - 1
    limit = reference["investigation_relative_ppl"]
    result = dict(windows=reference["windows"], scored_tokens=native["scored_tokens"],
        native_ppl=native["ppl"], reference_ppl=reference["ppl"],
        native_mean_nll=native["mean_nll"], reference_mean_nll=reference["mean_nll"],
        relative_ppl_difference=relative, investigation_relative_ppl=limit,
        within_investigation_line=abs(relative) <= limit, windows_comparison=paired,
        native_report=str(args.native), reference_report=str(args.reference),
        tokens_sha256=reference["tokens_sha256"],
        reference_arithmetic=reference["reference_arithmetic"],
        native_arithmetic=reference["native_arithmetic"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "windows_comparison"}, indent=2))
    if not result["within_investigation_line"]:
        raise SystemExit("PPL difference exceeds the investigation line")


if __name__ == "__main__":
    main()
