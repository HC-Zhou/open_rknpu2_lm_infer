#!/usr/bin/env python3
"""Compare prepared three-axis coordinates with pinned Qwen3.5 get_rope_index."""
import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace, MethodType
import torch
from vision_reference import extract


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    args=parser.parse_args()
    namespace={"torch":torch,"itertools":itertools}
    extract(args.root/"upstream/modeling_qwen3_5.py",set(),namespace,
            methods={"get_rope_index","get_vision_position_ids"})
    context=SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)))
    context.get_vision_position_ids=MethodType(namespace["get_vision_position_ids"],context)
    report={}
    for case in json.loads((args.root/"model-cases.json").read_text()):
        name=case["name"]
        prepared=json.loads((args.root/f"host-prompt-{name}"/"input.json").read_text())
        ids=torch.tensor([prepared["tokens"]]); modalities=torch.zeros_like(ids)
        for span in prepared["images"]:
            modalities[:,span["start"]:span["start"]+span["tokens"]]=1
        grid=torch.tensor([span["grid"] for span in prepared["images"]])
        positions,delta=namespace["get_rope_index"](context,ids,modalities,image_grid_thw=grid)
        assert torch.equal(positions[:,0].T,torch.tensor(prepared["positions"]))
        assert int(delta.item())+ids.shape[1]==prepared["next_position"]
        report[name]=dict(tokens=ids.shape[1],rope_delta=int(delta.item()),
                          positions_exact=True,next_position=prepared["next_position"])
    (args.root/"positions-comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
