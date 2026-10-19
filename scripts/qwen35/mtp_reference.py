#!/usr/bin/env python3
"""Export unchanged BF16 MTP weights evaluated by the HF FP32 decoder.

Uses the quantization handoff's MultiTokenPredictor (the official Qwen3.5
decoder layer); no calibration, quantization or model mutation is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quant-project', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.quant_project.resolve()))
    from evaluate_mtp import MultiTokenPredictor
    from common import configure, Qwen3_5TextConfig
    from prepare_rotation import rotation_matrices
    from transformers.cache_utils import DynamicCache
    import torch
    configure()
    config = Qwen3_5TextConfig(**json.loads((args.model/'config.json').read_text())['text_config'])
    model = MultiTokenPredictor(config, args.model).float()
    h, rows = config.hidden_size, 8
    index = torch.arange(rows*h, device='cuda', dtype=torch.float32).reshape(1, rows, h)
    hidden = torch.sin(index*0.0031) * 1.3
    embedding = torch.cos(index*0.0027) * 0.2
    rotation = rotation_matrices()[0].float().cuda()
    adapter = json.loads((args.model/'mtp_coordinate_adapter.json').read_text())
    gamma = torch.tensor(adapter['original_final_norm_gain'], device='cuda')
    original_h = (hidden@rotation.T)*gamma
    original_e = embedding@rotation.T
    with torch.inference_mode():
        output = model(original_h, original_e)
        cache = DynamicCache(config=model.config)
        cached = torch.cat([model(original_h[:,i:i+1], original_e[:,i:i+1], i, cache)
                            for i in range(rows)], dim=1)
        torch.testing.assert_close(cached, output, rtol=3e-4, atol=2e-3)
        head = (output/gamma)@rotation
    args.output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, tensor in dict(target_hidden=hidden, embedding=embedding,
                             original_hidden=original_h, original_embedding=original_e,
                             output=output, head=head).items():
        path = args.output/(name+'.f32')
        tensor.detach().float().contiguous().cpu().numpy().tofile(path)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    signs = torch.randint(0, 2, (2560,), generator=torch.Generator().manual_seed(42)).tolist()
    packed = ''.join(format(sum(signs[i+j]<<j for j in range(4)), 'x') for i in range(0,2560,4))
    report = dict(rows=rows, width=h, torch=torch.__version__, weight_dtype='BF16 widened to FP32',
                  model=str(args.model), target_hidden='after final RMSNorm',
                  shift='hidden[t] + embedding(token[t+1]) -> logits(token[t+2])',
                  mtp_source_sha256=hashlib.sha256((args.quant_project/'evaluate_mtp.py').read_bytes()).hexdigest(),
                  checkpoint_sha256=hashlib.sha256((args.model/'model-mtp.safetensors').read_bytes()).hexdigest(),
                  cached_vs_full_max_abs=float((cached-output).abs().max()), signs_packed=packed,
                  sha256=hashes)
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
