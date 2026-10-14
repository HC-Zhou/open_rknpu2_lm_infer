#!/usr/bin/env python3
"""Compare exported FP32 logits from identical input token sequences."""
import argparse
import array
import json
import math
from pathlib import Path


def read(path):
    values = array.array("f")
    values.frombytes(path.read_bytes())
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--vocab", type=int, required=True)
    parser.add_argument("--max-nrmse", type=float, default=1e-4)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    reference, actual = read(args.reference), read(args.actual)
    if len(reference) != len(actual) or not reference or len(reference) % args.vocab:
        raise SystemExit("incompatible logits shape")
    if not all(math.isfinite(x) for x in reference) or not all(math.isfinite(x) for x in actual):
        raise SystemExit("nonfinite logits")
    error = sum((a-b)**2 for a,b in zip(reference,actual))
    energy = sum(a*a for a in reference)
    nrmse = math.sqrt(error/max(energy,1e-30))
    rows = len(reference)//args.vocab
    top1 = 0
    for row in range(rows):
        begin,end = row*args.vocab,(row+1)*args.vocab
        expected = max(range(begin,end),key=reference.__getitem__)
        obtained = max(range(begin,end),key=actual.__getitem__)
        top1 += expected == obtained
    report = {
        "reference": str(args.reference), "actual": str(args.actual),
        "rows":rows,"vocab":args.vocab,"nrmse":nrmse,
        "max_absolute":max(abs(a-b) for a,b in zip(reference,actual)),
        "top1_matches":top1,"equal_elements":sum(a==b for a,b in zip(reference,actual)),
        "elements":len(reference),"max_nrmse":args.max_nrmse,
        "passed":nrmse<=args.max_nrmse and top1==rows,
    }
    args.report.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
