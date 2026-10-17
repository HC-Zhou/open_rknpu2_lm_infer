#!/usr/bin/env python3
"""Produce small FP32 fixtures by executing pinned upstream reference functions.

Requires PyTorch; does not load, quantize or download model weights.
"""
import argparse
import ast
import __future__
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import torch
from torch import nn
from torch.nn import functional as F


TRANSFORMERS_REVISION = "09281b8d56b08199ce9f397c694f38853cc45278"
VLLM_REVISION = "ae8f41238cc7316781ef4359890940749735b209"
LLAMA_CPP_REVISION = "a17e94d229fc7fed1567a799baebd11b771fe5af"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SOURCE_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/"
    + TRANSFORMERS_REVISION
    + "/src/transformers/models/qwen3_5/modeling_qwen3_5.py"
)


def reference_functions(source):
    """Load the actual reference definitions without importing Transformers/FLA."""
    names = {
        "l2norm", "torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule",
        "causal_conv1d_fn", "Qwen3_5RMSNorm", "Qwen3_5RMSNormGated",
    }
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node.decorator_list = []
            selected.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5TextRotaryEmbedding":
            selected.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "recomposition_frequencies")
    namespace = {"torch": torch, "nn": nn, "F": F, "ACT2FN": {"silu": F.silu}, "is_torchdynamo_exporting": lambda: False}
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, SOURCE_URL, "exec", flags=__future__.annotations.compiler_flag), namespace)
    assert names.issubset(namespace)
    return namespace


def values(size, offset=0, scale=1.0):
    return ((torch.arange(size, dtype=torch.float32) + offset) * 0.173).sin() * scale


def flat(tensor):
    return tensor.detach().reshape(-1).tolist()


def gdn_fixture(ref):
    rows, kh, vh, kd, vd = 7, 2, 4, 5, 3
    q = values(rows * kh * kd, 1).reshape(1, rows, kh, kd)
    k = values(q.numel(), 71).reshape_as(q)
    v = values(rows * vh * vd, 29).reshape(1, rows, vh, vd)
    g = -values(rows * vh, 5).abs().reshape(1, rows, vh) * 0.1
    beta = values(rows * vh, 3).sigmoid().reshape_as(g)
    initial = values(vh * kd * vd, 8, 0.2).reshape(1, vh, kd, vd)
    qn, kn = ref["l2norm"](q), ref["l2norm"](k)
    qe, ke = q.repeat_interleave(vh // kh, 2), k.repeat_interleave(vh // kh, 2)
    out, state = ref["torch_recurrent_gated_delta_rule"](qe, ke, v, g, beta, initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=True)
    chunk_out, chunk_state = ref["torch_chunk_gated_delta_rule"](qe, ke, v, g, beta, initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=True, chunk_size=4)
    torch.testing.assert_close(out, chunk_out, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(state, chunk_state, atol=1e-5, rtol=1e-4)
    return dict(rows=rows, key_heads=kh, value_heads=vh, key_dim=kd, value_dim=vd,
                query=flat(qn), key=flat(kn), value=flat(v), log_decay=flat(g), beta=flat(beta),
                initial_state=flat(initial.transpose(-1, -2)), output=flat(out), state=flat(state.transpose(-1, -2)))


def mixer_fixture(ref):
    rows, kh, vh, kd, vd, kernel = 7, 2, 4, 5, 3, 4
    kw, vw = kh * kd, vh * vd
    channels = 2 * kw + vw
    qkv = values(rows * channels, 14).reshape(rows, channels)
    z = values(rows * vw, 43).reshape(1, rows, vh, vd)
    a = values(rows * vh, 19, 0.3).reshape(1, rows, vh)
    b = values(rows * vh, 11).reshape_as(a)
    conv = values(channels * kernel, 61, 0.2).reshape(channels, kernel)
    a_log = values(vh, 3, 0.4)
    dt_bias = values(vh, 9, 0.2)
    norm_weight = 1 + values(vd, 7, 0.1)
    convolved = ref["causal_conv1d_fn"](qkv.T.unsqueeze(0), conv, activation="silu").transpose(1, 2)
    q, k, v = convolved.split([kw, kw, vw], -1)
    q = q.reshape(1, rows, kh, kd).repeat_interleave(vh // kh, 2)
    k = k.reshape(1, rows, kh, kd).repeat_interleave(vh // kh, 2)
    v = v.reshape(1, rows, vh, vd)
    g = -a_log.exp() * F.softplus(a + dt_bias)
    core, state = ref["torch_recurrent_gated_delta_rule"](q, k, v, g, b.sigmoid(), output_final_state=True, use_qk_l2norm_in_kernel=True)
    norm = ref["Qwen3_5RMSNormGated"](vd)
    norm.weight.data.copy_(norm_weight)
    output = norm(core, z)
    return dict(rows=rows, key_heads=kh, value_heads=vh, key_dim=kd, value_dim=vd, kernel=kernel,
                qkv=flat(qkv), z=flat(z), a=flat(a), b=flat(b), conv_weight=flat(conv),
                a_log=flat(a_log), dt_bias=flat(dt_bias), norm_weight=flat(norm_weight), output=flat(output),
                state=flat(state.transpose(-1, -2)), history=flat(qkv[-(kernel-1):].T))


def ops_fixture(ref):
    x = values(24, 13).reshape(3, 8)
    weight = values(8, 7, 0.1)
    norm = ref["Qwen3_5RMSNorm"](8)
    norm.weight.data.copy_(weight)
    head_dim, rotary_dim, heads = 24, 12, 2
    positions = torch.tensor([[71, 71, 71], [1, 19, 37], [4, 5, 6]])
    rope_input = values(3 * heads * head_dim, 23).reshape(3, heads, head_dim)
    inv = 1 / (10000.0 ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    freq = positions.T[:, None, :, None].float() * inv
    ctx = SimpleNamespace(mrope_section=[2, 2, 2])
    cos = ref["recomposition_frequencies"](ctx, freq.cos()).squeeze(0).unsqueeze(1)
    sin = ref["recomposition_frequencies"](ctx, freq.sin()).squeeze(0).unsqueeze(1)
    rot, tail = rope_input[..., :rotary_dim], rope_input[..., rotary_dim:]
    rotated_half = torch.cat([-rot[..., rotary_dim//2:], rot[..., :rotary_dim//2]], -1)
    rope_out = torch.cat([rot * cos + rotated_half * sin, tail], -1)
    return dict(norm_input=flat(x), norm_weight=flat(weight), norm_output=flat(norm(x)),
                rope_input=flat(rope_input), rope_output=flat(rope_out), positions=positions.tolist(), heads=heads, head_dim=head_dim, rotary_dim=rotary_dim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="local copy of the pinned modeling_qwen3_5.py")
    parser.add_argument("--output", type=Path, default=Path("llm_generated_docs/evidence/qwen35_mtp/cpu_fixtures.json"))
    args = parser.parse_args()
    source = args.source.read_text() if args.source else urllib.request.urlopen(SOURCE_URL).read().decode()
    torch.set_num_threads(1)
    with torch.no_grad():
        ref = reference_functions(source)
        result = dict(provenance=dict(transformers_revision=TRANSFORMERS_REVISION, vllm_revision=VLLM_REVISION,
                                      llama_cpp_revision=LLAMA_CPP_REVISION, model_revision=MODEL_REVISION,
                                      torch_version=torch.__version__, reference_source=SOURCE_URL),
                      gdn=gdn_fixture(ref), mixer=mixer_fixture(ref), ops=ops_fixture(ref))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
