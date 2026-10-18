#!/usr/bin/env python3
"""Teacher-forced C7 image/text oracle using the quantization project's integer
text reference and the pinned ViT features from vision_reference.py.

Does not modify the quantization project, checkpoint or Python environment.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--feature-source", choices=["reference","board"], default="reference")
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime_root))
    from common import load_model, configure
    from transformers import AutoTokenizer
    import transformers
    configure()
    model = load_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    report = dict(model=str(args.model), text_reference=str(args.runtime_root / "common.py"),
                  transformers=transformers.__version__, torch=torch.__version__, feature_source=args.feature_source, cases={})
    with torch.inference_mode():
        for case in json.loads(args.cases.read_text()):
            name = case["name"]
            prepared = json.loads((args.root / f"host-prompt-{name}" / "input.json").read_text())
            tokens = torch.tensor([prepared["tokens"]], device="cuda")
            embeddings = model.model.embed_tokens(tokens)
            for span, image in zip(prepared["images"], case["images"], strict=True):
                features = np.fromfile(args.root / f"{args.feature_source}-{image}" / "features.f32", dtype="<f4").reshape(-1, embeddings.shape[-1])
                assert len(features) == span["tokens"]
                embeddings[0, span["start"]:span["start"] + len(features)] = torch.from_numpy(features).to("cuda")
            coordinates = torch.tensor(prepared["positions"], device="cuda").T[:, None, :]
            rows = []
            # Recompute full teacher-forced sequences to avoid any implicit
            # multimodal decode-position handling in framework generation.
            continuation = case.get("continuation", [])
            start = time.monotonic()
            for index in range(len(continuation) + 1):
                positions = torch.cat([torch.arange(embeddings.shape[1], device="cuda")[None, None, :], coordinates], dim=0)
                logits = model(inputs_embeds=embeddings, position_ids=positions, use_cache=False, logits_to_keep=1).logits[0, -1].float()
                rows.append(logits.cpu().numpy())
                if index < len(continuation):
                    token = continuation[index]
                    embeddings = torch.cat([embeddings, model.model.embed_tokens(torch.tensor([[token]], device="cuda"))], dim=1)
                    coordinates = torch.cat([coordinates, torch.full((3, 1, 1), prepared["next_position"] + index, device="cuda")], dim=-1)
            out = args.root / f"model-{args.feature_source}-{name}"
            out.mkdir(exist_ok=True)
            logits = np.stack(rows)
            logits.tofile(out / "logits.f32")
            ids = logits.argmax(-1).tolist()
            report["cases"][name] = dict(prompt_tokens=len(prepared["tokens"]), rows=len(rows),
                                         top1_ids=ids, top1_text=[tokenizer.decode([i]) for i in ids],
                                         elapsed_seconds=time.monotonic() - start)
            print(json.dumps(report["cases"][name]), flush=True)
    (args.root / f"model-{args.feature_source}-manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
