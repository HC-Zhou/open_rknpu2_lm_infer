#!/usr/bin/env python3
"""Export fixed WikiText windows and their paired, existing C7 FP32 scores."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-project", type=Path, required=True)
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.quant_project.resolve()
    source = root / "data/wiki_full.npy"
    report_path = root / "results/C7/ppl.json"
    scores_path = root / "results/C7/ppl.token_nll.npy"
    reference = json.loads(report_path.read_text())
    tokens = np.load(source, allow_pickle=False)
    scores = np.load(scores_path, allow_pickle=False)
    assert reference["completed"] and not reference["source"]
    assert reference["sequence_length"] == 512
    assert reference["scored_target_positions"] == [257, 511]
    assert tokens.shape == (len(reference["chunks"]), 512)
    assert scores.shape == (len(tokens), 255) and np.isfinite(scores).all()
    assert 0 < args.windows <= len(tokens)
    np.testing.assert_allclose(scores.mean(axis=1),
        [row["nll"] for row in reference["chunks"]], rtol=0, atol=1e-12)
    args.output.mkdir(parents=True, exist_ok=True)
    token_path = args.output / "tokens.json"
    token_path.write_text(json.dumps(tokens[:args.windows].tolist()) + "\n")
    values = scores[:args.windows]
    manifest = dict(
        checkpoint="C7", windows=args.windows, window_selection="first N, original order",
        window_size=512, scored_target_positions=[257, 511],
        scored_tokens=int(values.size), mean_nll=float(values.mean()),
        ppl=math.exp(float(values.mean())), token_nll=values.tolist(),
        tokens_sha256=digest(token_path),
        provenance={str(path.relative_to(root)): digest(path) for path in [source,
            report_path, scores_path, root / "models/C7/config.json",
            root / "models/C7/model.safetensors.index.json"]},
        reference_arithmetic="C7 GPU exact integer GEMM, FP32 model; original torch A8 rounding",
        native_arithmetic="C7 NPU INT32 GEMM, round-away A8; CPU FP32 GDN; native FP16 attention",
        investigation_relative_ppl=0.01,
    )
    (args.output / "reference.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in
        ["checkpoint", "windows", "scored_tokens", "mean_nll", "ppl", "tokens_sha256"]}))


if __name__ == "__main__":
    main()
