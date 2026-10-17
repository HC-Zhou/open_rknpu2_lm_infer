#!/usr/bin/env python3
"""Export same-checkpoint FP32 logits with exact integer GEMMs for board checks."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-project", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.quant_project.resolve()))
    from common import configure, load_model
    import torch
    import gpu_runtime
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as reference

    configure()
    # The runtime ABI rounds halfway values away from zero and multiplies by
    # reciprocal scale. Keep weight codes/scales exactly as supplied by C7.
    def native_a8(value):
        value = value.float()
        maximum = value.abs().amax(-1, keepdim=True)
        scale = torch.where(maximum == 0, torch.ones_like(maximum),
                            (maximum / 127).clamp_min(torch.finfo(torch.float32).tiny))
        scaled = value * scale.reciprocal()
        codes = (scaled.abs() + 0.5).floor() * scaled.sign()
        return codes.clamp(-127, 127).to(torch.int8), scale
    gpu_runtime.quantize_token = native_a8
    model = load_model(args.model)
    # Same recurrence as the CPU path; avoids the chunk kernel's different FP32
    # reduction order and the old HF cache's multi-row continuation limitation.
    def recurrent(query, key, value, g, beta, initial_state=None,
                  output_final_state=False, use_qk_l2norm_in_kernel=False, **_kwargs):
        return reference.torch_recurrent_gated_delta_rule(
            query, key, value, g, beta, initial_state, output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
    for layer in model.model.layers:
        if layer.layer_type == "linear_attention":
            layer.linear_attn.chunk_gated_delta_rule = recurrent
            layer.linear_attn.recurrent_gated_delta_rule = recurrent
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for name, text in [("arithmetic", "请计算 12+7，只输出结果。"),
                       ("chinese", "用一句话说明什么是机器学习。")]:
        tokens = tokenizer.apply_chat_template([{"role": "user", "content": text}],
            tokenize=True, return_tensors="pt", return_dict=False,
            add_generation_prompt=True, enable_thinking=False).cuda()
        prompt = tokens[0].tolist()
        trace_position = [0]
        handles = []
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        if args.trace:
            traced = directory / "trace"
            traced.mkdir(exist_ok=True)
            def save(layer, stage, tensor):
                tensor.detach().float().contiguous().cpu().numpy().tofile(
                    traced / f"p{trace_position[0]}-l{layer}-{stage}.f32")
            def hook(layer, stage):
                return lambda _m, _args, out: save(layer, stage, out[0] if isinstance(out, tuple) else out)
            def prehook(layer, stage):
                return lambda _m, inputs: save(layer, stage, inputs[0])
            for i, layer in enumerate(model.model.layers):
                handles.append(layer.input_layernorm.register_forward_hook(hook(i, "input")))
                handles.append(layer.post_attention_layernorm.register_forward_hook(hook(i, "post")))
                handles.append(layer.mlp.down_proj.register_forward_pre_hook(prehook(i, "middle")))
                handles.append(layer.register_forward_hook(hook(i, "hidden")))
                if layer.layer_type == "linear_attention":
                    handles.append(layer.linear_attn.in_proj_qkv.register_forward_hook(hook(i, "gdn_qkv")))
                    handles.append(layer.linear_attn.norm.register_forward_hook(hook(i, "attn")))
                    handles.append(layer.linear_attn.out_proj.register_forward_hook(hook(i, "o")))
                else:
                    handles.append(layer.self_attn.o_proj.register_forward_pre_hook(prehook(i, "attn")))
                    handles.append(layer.self_attn.o_proj.register_forward_hook(hook(i, "o")))
        with torch.inference_mode():
            output = model(input_ids=tokens, use_cache=True, logits_to_keep=1)
            logits = [output.logits[0, -1].float().cpu()]
            continuation = []
            cache = output.past_key_values
            for _ in range(args.steps):
                trace_position[0] = len(prompt) + len(continuation)
                token = int(logits[-1].argmax())
                continuation.append(token)
                output = model(input_ids=torch.tensor([[token]], device="cuda"),
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits.append(output.logits[0, -1].float().cpu())
                cache = output.past_key_values
        for handle in handles:
            handle.remove()
        (directory / "prompt.json").write_text(json.dumps(prompt) + "\n")
        (directory / "continuation.json").write_text(json.dumps(continuation) + "\n")
        matrix = torch.stack(logits).contiguous().numpy()
        matrix.tofile(directory / "logits.f32")
        case = dict(name=name, prompt=prompt, continuation=continuation,
                    logits=f"{name}/logits.f32", top1=matrix.argmax(-1).tolist(),
                    response=tokenizer.decode(continuation, skip_special_tokens=True))
        cases.append(case)
        print(json.dumps({"name": name, "prompt_tokens": len(prompt),
                          "response": case["response"]}, ensure_ascii=False), flush=True)
    report = dict(model=str(args.model.resolve()), torch=torch.__version__,
        transformers=reference.__file__, activation_quantization="full-K maxabs/127, reciprocal multiply, round-away",
        gdn="official FP32 recurrent", attention="official SDPA FP32", cases=cases,
        gates=dict(reference_logit_nrmse=0.01, reference_top1="all rows",
                   chunk_logit_nrmse=0.001, chunk_top1="all rows"))
    (args.output / "cases.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
