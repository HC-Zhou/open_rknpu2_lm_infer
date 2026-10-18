#!/usr/bin/env python3
"""Summarize all ViT stages and teacher-forced multimodal logits."""
import argparse
import json
from pathlib import Path
import numpy as np


def metrics(reference, actual):
    assert reference.shape == actual.shape and np.isfinite(actual).all()
    a, b = reference.astype(np.float64), actual.astype(np.float64)
    return dict(values=a.size, relative_l2=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                cosine=float(np.dot(a.ravel(), b.ravel())/(np.linalg.norm(a)*np.linalg.norm(b))),
                max_abs=float(np.abs(a-b).max()))


def log_softmax(x):
    shifted = x.astype(np.float64) - x.max()
    return shifted - np.log(np.exp(shifted).sum())


def distribution(reference, actual):
    result=metrics(reference,actual)
    ref_logp,logp=log_softmax(reference),log_softmax(actual)
    result.update(reference_top1=int(reference.argmax()),actual_top1=int(actual.argmax()),
                  kl=float(np.sum(np.exp(ref_logp)*(ref_logp-logp))))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()
    root = args.root
    report = dict(vision={}, model={}, attribution={}, failures=[])
    for name in ["red", "blue"]:
        stages = ["p0-l-1-vision_patch_position"] + [f"p0-l{i}-vision_hidden" for i in range(24)] + ["features"]
        result = {}
        for stage in stages:
            a = np.fromfile(root/f"reference-{name}"/f"{stage}.f32", dtype="<f4")
            b = np.fromfile(root/f"board-{name}"/f"{stage}.f32", dtype="<f4")
            result[stage] = metrics(a, b)
        assert result["features"]["cosine"] > .9999
        assert result["features"]["relative_l2"] < .02
        report["vision"][name] = result
    if not args.features_only:
        for case in json.loads((root/"model-cases.json").read_text()):
            name = case["name"]
            a = np.fromfile(root/f"model-reference-{name}"/"logits.f32", dtype="<f4").reshape(-1,248320)
            b = np.fromfile(root/"board-model"/name/"logits.f32", dtype="<f4").reshape(-1,248320)
            result = metrics(a,b)
            rows = []
            for i, (ref, value) in enumerate(zip(a, b)):
                ref_logp, logp = log_softmax(ref), log_softmax(value)
                row = distribution(ref,value)
                if i < len(case.get("continuation", [])):
                    token = case["continuation"][i]
                    row.update(target=token, reference_nll=float(-ref_logp[token]), actual_nll=float(-logp[token]))
                rows.append(row)
            result["rows"] = rows
            if not all(row["cosine"]>.98 and row["kl"]<.1 for row in rows):
                report["failures"].append(f"{name}: logits distribution threshold")
            if rows[0]["actual_top1"] != rows[0]["reference_top1"]:
                report["failures"].append(f"{name}: first top1 differs")
            report["model"][name] = result
            same_features=args.root/f"model-board-{name}"/"logits.f32"
            if same_features.exists():
                middle=np.fromfile(same_features,dtype="<f4").reshape(-1,248320)
                report["attribution"][name]={
                    "vision_change_with_same_text_reference":[distribution(x,y) for x,y in zip(a,middle,strict=True)],
                    "same_board_features_text_execution":[distribution(x,y) for x,y in zip(middle,b,strict=True)],
                }
    report["passed"] = not report["failures"]
    (root/"comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    if report["attribution"]:
        (root/"attribution.json").write_text(json.dumps(report["attribution"],indent=2)+"\n")
    print(json.dumps({"passed":report["passed"],"failures":report["failures"],"vision":{k:v["features"] for k,v in report["vision"].items()},
                      "model":{k:{"relative_l2":v["relative_l2"],"top1":[r["actual_top1"] for r in v["rows"]],"kl":[r["kl"] for r in v["rows"]]} for k,v in report["model"].items()}},indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
