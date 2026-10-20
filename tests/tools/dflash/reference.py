"""Independent FP32 reference for the board's real C4 DFlash fixtures.

Equations follow z-lab/dflash 07ebd93db9f472af339b644bb70221ad8428328a,
with ordinary Torch Linear/RMSNorm/softmax and the released draft weights.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("fixtures", type=Path)
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()
torch.set_num_threads(args.threads)
config = json.loads((args.model / "config.json").read_text())
h = config["hidden_size"]
heads = config["num_attention_heads"]
kv_heads = config["num_key_value_heads"]
d = config["head_dim"]
eps = config["rms_norm_eps"]
width = h * len(config["dflash_config"]["target_layer_ids"])
weights = safe_open(str(args.model / "model.safetensors"), framework="pt", device="cpu")


def weight(name):
    return weights.get_tensor(name + ".weight").float()


def linear(x, name):
    return F.linear(x, weight(name))


def norm(x, name):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps) * weight(name)


def rope(x, positions):
    inv = config["rope_parameters"]["rope_theta"] ** (-torch.arange(0, d, 2).float() / d)
    angle = positions.float()[:, None] * inv[None, :]
    angle = torch.cat((angle, angle), dim=-1)[:, None, :]
    rotated = torch.cat((-x[..., d // 2:], x[..., :d // 2]), dim=-1)
    return x * angle.cos() + rotated * angle.sin()


results = []
with torch.inference_mode():
    for name in ["dflash4", "dflash8"]:
        fixture = json.loads((args.fixtures / f"{name}.json").read_text())
        position = fixture["position"]
        prefix = torch.from_file(str(args.fixtures / f"{name}-prefix.f32"), shared=False, size=position * width, dtype=torch.float32).reshape(position, width)
        context = norm(linear(prefix, "fc"), "hidden_norm")
        hidden = torch.tensor(fixture["embeddings"], dtype=torch.float32).reshape(-1, h)
        rows = hidden.shape[0]
        assert not fixture["new_features"]
        q_positions = torch.arange(position, position + rows)
        k_positions = torch.arange(position + rows)
        for index in range(config["num_hidden_layers"]):
            p = f"layers.{index}"
            residual = hidden
            x = norm(hidden, p + ".input_layernorm")
            q = norm(linear(x, p + ".self_attn.q_proj").reshape(rows, heads, d), p + ".self_attn.q_norm")
            k = torch.cat((linear(context, p + ".self_attn.k_proj"), linear(x, p + ".self_attn.k_proj")), dim=0).reshape(-1, kv_heads, d)
            v = torch.cat((linear(context, p + ".self_attn.v_proj"), linear(x, p + ".self_attn.v_proj")), dim=0).reshape(-1, kv_heads, d)
            k = norm(k, p + ".self_attn.k_norm")
            q = rope(q, q_positions).transpose(0, 1)
            k = rope(k, k_positions).transpose(0, 1).repeat_interleave(heads // kv_heads, dim=0)
            v = v.transpose(0, 1).repeat_interleave(heads // kv_heads, dim=0)
            scores = (q @ k.transpose(-1, -2)) / d**0.5
            if config["layer_types"][index] == "sliding_attention":
                visible = (k_positions[None, :] <= q_positions[:, None]) & (q_positions[:, None] - k_positions[None, :] < config["sliding_window"])
                scores.masked_fill_(~visible[None], -torch.inf)
            attended = (scores.softmax(dim=-1) @ v).transpose(0, 1).reshape(rows, heads * d)
            hidden = residual + linear(attended, p + ".self_attn.o_proj")
            residual = hidden
            x = norm(hidden, p + ".post_attention_layernorm")
            hidden = residual + linear(F.silu(linear(x, p + ".mlp.gate_proj")) * linear(x, p + ".mlp.up_proj"), p + ".mlp.down_proj")
        expected = norm(hidden, "norm")
        actual = torch.tensor(fixture["draft_hidden"], dtype=torch.float32).reshape_as(expected)
        difference = actual - expected
        relative_rmse = (difference.square().mean().sqrt() / expected.square().mean().sqrt()).item()
        cosine = F.cosine_similarity(actual, expected).min().item()
        result = {"mode": name, "relative_rmse": relative_rmse, "max_abs": difference.abs().max().item(), "min_row_cosine": cosine}
        print(json.dumps(result), flush=True)
        assert relative_rmse < 0.02 and cosine > 0.999, result
        results.append(result)
(args.fixtures / "reference.json").write_text(json.dumps({"status": "PASS", "draft_dtype":"fp16", "source_commit": "07ebd93db9f472af339b644bb70221ad8428328a", "results": results}, indent=2) + "\n")
