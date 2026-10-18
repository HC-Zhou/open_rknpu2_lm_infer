#!/usr/bin/env python3
"""Run pinned Transformers ViT definitions on unchanged C7 BF16 weights.

Uses AST extraction (no Transformers/torchvision installation required). The
extracted definitions retain upstream math; only framework/config wrappers and
the standard torch SDPA dispatch are supplied here. CPU patches use the pinned
PIL processor's smart_resize and patchify, Pillow bicubic, and float32 normalize.
Upstream source is Apache-2.0, Copyright Qwen team / Hugging Face.
"""
import __future__
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

REVISION = "09281b8d56b08199ce9f397c694f38853cc45278"
BASE = f"https://raw.githubusercontent.com/huggingface/transformers/{REVISION}/src/transformers/"
FILES = {
    "modeling_qwen3_5.py": "models/qwen3_5/modeling_qwen3_5.py",
    "vision_utils.py": "vision_utils.py",
    "image_processing_pil_qwen2_vl.py": "models/qwen2_vl/image_processing_pil_qwen2_vl.py",
}


def extract(path, names, namespace, methods=()):
    selected = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.ClassDef):
            selected.extend(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods)
    for node in selected:
        for child in ast.walk(node):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                child.decorator_list = [d for d in child.decorator_list if isinstance(d, ast.Name) and d.id in {"staticmethod", "classmethod"}]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)


class BaseModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.config = config

    def post_init(self):
        pass


def sdpa(module, q, k, v, attention_mask=None, scaling=None, **kwargs):
    return F.scaled_dot_product_attention(q, k, v, scale=scaling).transpose(1, 2), None


def definitions(source):
    namespace = dict(torch=torch, nn=nn, F=F, np=np, math=math,
                     GradientCheckpointingLayer=nn.Module,
                     Qwen3_5PreTrainedModel=BaseModel,
                     ACT2FN={"gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh")},
                     ALL_ATTENTION_FUNCTIONS=SimpleNamespace(get_interface=lambda *args: sdpa),
                     eager_attention_forward=sdpa,
                     is_flash_attention_requested=lambda _: False,
                     get_max_seqlen=lambda *args, **kwargs: None,
                     BaseModelOutputWithPooling=lambda **kwargs: SimpleNamespace(**kwargs))
    extract(source / "vision_utils.py", {
        "get_vision_cu_seqlens", "get_vision_attention_seqlens", "get_vision_position_ids",
        "_interpolation_axis_taps_weights", "get_vision_interpolation_indices_and_weights",
    }, namespace)
    extract(source / "modeling_qwen3_5.py", {
        "Qwen3_5VisionRotaryEmbedding", "Qwen3_5VisionMLP", "Qwen3_5VisionPatchEmbed",
        "Qwen3_5VisionPatchMerger", "apply_rotary_pos_emb_vision", "rotate_half",
        "Qwen3_5VisionAttention", "Qwen3_5VisionBlock", "Qwen3_5VisionModel",
    }, namespace)
    extract(source / "image_processing_pil_qwen2_vl.py", {"smart_resize"}, namespace, methods={"patchify"})
    return namespace


def prepare(image, processor, ref):
    image = ImageOps.exif_transpose(image)
    h, w = ref["smart_resize"](image.height, image.width, factor=32,
                              min_pixels=processor["size"]["shortest_edge"], max_pixels=262144)
    rgb = np.asarray(image.convert("RGB").resize((w, h), Image.Resampling.BICUBIC))
    normalized = rgb.astype(np.float32).transpose(2, 0, 1) * np.float32(1 / 255)
    normalized = (normalized - np.array(processor["image_mean"], dtype=np.float32)[:, None, None]) / np.array(processor["image_std"], dtype=np.float32)[:, None, None]
    patches, gh, gw = ref["patchify"](None, normalized, 16, 2, 2)
    return patches, [1, gh, gw]


def dump(path, tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().float().cpu().numpy()
    np.asarray(tensor, dtype="<f4").tofile(path)


def create_images(output):
    for name, color, size in [("red", "red", (256, 256)), ("blue", "blue", (384, 256)), ("blue-square", "blue", (256, 256))]:
        im = Image.new("RGB", size, "white")
        ImageDraw.Draw(im).rectangle((32, 32, size[0] - 33, size[1] - 33), fill=color)
        im.save(output / f"{name}.png")
    oriented = Image.new("RGB", (192, 96), "red")
    ImageDraw.Draw(oriented).rectangle((96, 0, 191, 95), fill="blue")
    exif = Image.Exif()
    exif[274] = 6
    oriented.save(output / "jpeg-orientation.jpg", quality=92, subsampling=0, exif=exif)
    rng = np.random.default_rng(17)
    for i, shape in enumerate([(57, 91), (701, 1093), (256, 256)]):
        Image.fromarray(rng.integers(0, 256, (*shape, 3), dtype=np.uint8)).save(output / f"resize-{i}.png")


def cpu_fixture(ref):
    x = (torch.arange(36).float() * .173).sin().reshape(3, 12)
    weight = 1 + torch.arange(12).float().sin() * .1
    bias = torch.arange(12).float().cos() * .1
    gelu_input = torch.linspace(-6, 6, 25)
    grid = torch.tensor([[1, 2, 4]])
    positions = ref["get_vision_position_ids"](grid, 2)
    indices, weights = ref["get_vision_interpolation_indices_and_weights"](grid, 3, align_corners=True, spatial_merge_size=2)
    table = torch.arange(36).float().sin().reshape(9, 4)
    q = (torch.arange(8 * 2 * 8).float() * .173).sin().reshape(8, 2, 8)
    rotary = ref["Qwen3_5VisionRotaryEmbedding"](SimpleNamespace(rope_parameters={"rope_type": "axial", "rope_theta": 10000}, head_dim=8))
    cos, sin = rotary(q, positions)
    rotated, _ = ref["apply_rotary_pos_emb_vision"](q, q, cos, sin)
    flat = lambda t: t.reshape(-1).tolist()
    return dict(norm_input=flat(x), norm_weight=flat(weight), norm_bias=flat(bias),
                norm_output=flat(F.layer_norm(x, (12,), weight, bias, 1e-6)),
                gelu_input=flat(gelu_input), gelu_exact=flat(F.gelu(gelu_input)),
                gelu_tanh=flat(F.gelu(gelu_input, approximate="tanh")),
                positions=positions.tolist(), table=flat(table),
                interpolated=flat((table[indices] * weights[:, :, None]).sum(1)),
                rope_input=flat(q), rope_output=flat(rotated))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--processor-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.source or args.output / "upstream"
    source.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for file, relative in FILES.items():
        path = source / file
        if not path.exists():
            path.write_bytes(urllib.request.urlopen(BASE + relative).read())
        provenance[file] = dict(url=BASE + relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    ref = definitions(source)
    (args.output / "cpu-fixture.json").write_text(json.dumps(cpu_fixture(ref), indent=2) + "\n")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = json.loads((args.model / "preprocessor_config.json").read_text())
    create_images(args.output)
    manifest = dict(transformers_revision=REVISION, sources=provenance, torch=torch.__version__,
                    numpy=np.__version__, processor_backend="PIL", max_pixels=262144,
                    model=str(args.model), precision="BF16 checkpoint widened to FP32; TF32 disabled", cases={})
    model = None
    if not args.processor_only:
        config = json.loads((args.model / "config.json").read_text())["vision_config"]
        config.update(num_attention_heads=config["num_heads"], rope_parameters={"rope_type": "axial", "rope_theta": 10000}, _attn_implementation="sdpa")
        model = ref["Qwen3_5VisionModel"](SimpleNamespace(**config)).eval().cuda()
        index = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
        state = {}
        for shard in sorted({file for key, file in index.items() if key.startswith("model.visual.")}):
            with safe_open(args.model / shard, framework="pt", device="cpu") as weights:
                for key in weights.keys():
                    if key.startswith("model.visual."):
                        state[key.removeprefix("model.visual.")] = weights.get_tensor(key).float()
        model.load_state_dict(state, strict=True)
        del state
    torch.set_num_threads(4)
    for image_path in sorted(args.output.glob("*.png")):
        name = image_path.stem
        directory = args.output / f"reference-{name}"
        directory.mkdir(exist_ok=True)
        patches, grid = prepare(Image.open(image_path), processor, ref)
        dump(directory / "patches.f32", patches)
        entry = dict(grid=grid, image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest())
        if model is not None and name in {"red", "blue"}:
            with torch.inference_mode():
                handles = []
                for i, block in enumerate(model.blocks):
                    handles.append(block.register_forward_hook(lambda _, __, value, i=i: dump(directory / f"p0-l{i}-vision_hidden.f32", value)))
                pixels = torch.from_numpy(patches).cuda()
                grid_tensor = torch.tensor([grid], device="cuda")
                indices, weights = ref["get_vision_interpolation_indices_and_weights"](grid_tensor, 48, align_corners=True, spatial_merge_size=2)
                hidden = model.patch_embed(pixels) + (model.pos_embed(indices) * weights[:, :, None]).sum(1)
                dump(directory / "p0-l-1-vision_patch_position.f32", hidden)
                output = model(pixels, grid_tensor)
                dump(directory / "features.f32", output.pooler_output)
                for handle in handles:
                    handle.remove()
                entry["features_shape"] = list(output.pooler_output.shape)
        manifest["cases"][name] = entry
    (args.output / "reference-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
