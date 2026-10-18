#!/usr/bin/env python3
"""FP32 ViT oracle and CUDA FP16 control, using pinned upstream definitions.

No text decoder or integer reference is involved. Checkpoint tensors are decoded
to FP32 before model loading; TF32 and autocast are disabled. A separate FP16
control uses FP32 GEMM accumulation and math SDPA. Original-checkpoint outputs
are additionally rotated into C7 coordinates outside the model computation.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from safetensors import safe_open
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import vision_reference as reference


def visual_state(path):
    index = json.loads((path/'model.safetensors.index.json').read_text())['weight_map']
    state = {}
    for shard in sorted({v for k, v in index.items() if k.startswith('model.visual.')}):
        with safe_open(path/shard, framework='pt', device='cpu') as reader:
            for key in reader.keys():
                if key.startswith('model.visual.'):
                    state[key.removeprefix('model.visual.')] = reader.get_tensor(key).float()
    return state


def rotation():
    # Same Paley20 x Sylvester128 / signed rows / seed42 as C7 prepare_rotation.
    prime = 19
    residues = {i*i % prime for i in range(1, prime)}
    core = torch.zeros(prime, prime, dtype=torch.float64)
    for i in range(prime):
        for j in range(prime):
            if i != j:
                core[i, j] = 1 if (i-j) % prime in residues else -1
    h20 = torch.ones(20, 20, dtype=torch.float64)
    h20[1:, 1:] = core-torch.eye(prime, dtype=torch.float64)
    h128 = torch.ones(1, 1, dtype=torch.float64)
    while len(h128) < 128:
        h128 = torch.cat((torch.cat((h128, h128), 1), torch.cat((h128, -h128), 1)), 0)
    signs = 2*torch.randint(0, 2, (2560,), generator=torch.Generator().manual_seed(42))-1
    return signs[:, None]*torch.kron(h20, h128)/(2560**0.5)


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, SimpleNamespace):
        yield from tensors(vars(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--original-model', required=True, type=Path)
    parser.add_argument('--reference-root', required=True, type=Path)
    parser.add_argument('--artifact-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    definitions = reference.definitions(args.reference_root/'upstream')
    processor = json.loads((args.model/'preprocessor_config.json').read_text())
    fixtures = {}
    for image in ('red', 'blue', 'red512'):
        path = args.reference_root/f'{image}.png' if image != 'red512' else args.artifact_root.parent/'floating-mm-util-20261018/red512.png'
        patches, grid = reference.prepare(Image.open(path), processor, definitions)
        if image != 'red512':
            existing = np.fromfile(args.reference_root/f'reference-{image}/patches.f32', dtype='<f4')
            assert np.array_equal(existing, patches.ravel())
        reference.dump(args.output/f'{image}-patches.f32', patches)
        fixtures[image] = (patches, grid)
    config = json.loads((args.model/'config.json').read_text())['vision_config']
    config.update(num_attention_heads=config['num_heads'],
                  rope_parameters={'rope_type':'axial','rope_theta':10000}, _attn_implementation='sdpa')
    model = definitions['Qwen3_5VisionModel'](SimpleNamespace(**config)).eval().float().cuda()
    state = visual_state(args.model)
    model.load_state_dict(state, strict=True)
    del state
    # RoPE frequencies are explicitly constructed in FP32 by upstream. Preserve
    # their values when casting parameters for the FP16 control.
    fp32_buffers = {name: value.clone() for name, value in model.named_buffers()
                    if value.is_floating_point() and value.dtype == torch.float32}
    report = dict(torch=torch.__version__, device=torch.cuda.get_device_name(),
                  model=str(args.model), original_model=str(args.original_model),
                  definitions_revision=reference.REVISION, text_decoder=False,
                  weight_provenance='BF16 checkpoint values widened to FP32; no original FP32 master weights available',
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  matmul_precision=torch.get_float32_matmul_precision(), autocast=False,
                  sdpa='MATH', fp16_gemm_accumulation=False,
                  fp16_gemm_reduced_precision_reduction=False,
                  preserved_fp32_buffers=list(fp32_buffers), runs={})
    rotated = rotation().cuda()
    for variant in ('c7-fp32', 'cuda-fp16', 'original-fp32'):
        if variant == 'original-fp32':
            model.float()
            state = visual_state(args.original_model)
            # FP16 control changes model storage, so reload all original tensors.
            model.load_state_dict(state, strict=True)
            del state
        elif variant == 'cuda-fp16':
            model.half()
        for name, value in fp32_buffers.items():
            parent, _, leaf = name.rpartition('.')
            model.get_submodule(parent)._buffers[leaf] = value.clone()
        expected_dtype = torch.float16 if variant == 'cuda-fp16' else torch.float32
        assert all(p.dtype == expected_dtype for p in model.parameters())
        audit = Counter()
        def check_dtype(module, inputs, output):
            for value in tensors(output):
                if value.is_floating_point():
                    audit[str(value.dtype)] += 1
                    if expected_dtype == torch.float32:
                        assert value.dtype == torch.float32, (type(module).__name__, value.dtype)
        handles = [module.register_forward_hook(check_dtype) for module in model.modules()]
        for image, (patches, grid) in fixtures.items():
            directory = args.output/f'{variant}-{image}'
            directory.mkdir()
            trace_handles = [block.register_forward_hook(lambda _, __, value, i=i:
                reference.dump(directory/f'p0-l{i}-vision_hidden.f32', value)) for i, block in enumerate(model.blocks)]
            pixels = torch.from_numpy(patches).to(device='cuda', dtype=expected_dtype)
            grid_tensor = torch.tensor([grid], device='cuda')
            with torch.inference_mode(), torch.autocast('cuda', enabled=False), sdpa_kernel(SDPBackend.MATH):
                output = model(pixels, grid_tensor).pooler_output
                reference.dump(directory/'features.f32', output)
                if variant == 'original-fp32':
                    # Coordinate transform only, after the original FP32 ViT.
                    reference.dump(directory/'features-c7-coordinates.f32', (output.double() @ rotated).float())
            for handle in trace_handles:
                handle.remove()
            report['runs'][f'{variant}-{image}'] = dict(grid=grid, features=list(output.shape), dtype=str(output.dtype))
            print(variant, image, output.shape, output.dtype, flush=True)
        for handle in handles:
            handle.remove()
        report['runs'][variant+'-dtype-audit'] = dict(audit)
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
