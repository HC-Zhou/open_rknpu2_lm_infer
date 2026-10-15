<div align="center">

# open_rknpu2_lm_infer

**A fully open-source, high-performance LLM inference engine for the RK3588 NPU**

![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Rust](https://img.shields.io/badge/rust-1.95-orange)
![Platform](https://img.shields.io/badge/platform-RK3588-green)

[English](README.md) | [简体中文](README_zh.md)

</div>

## Highlights

- 🚀 **Faster than the vendor SDK**: Autotuning, heterogeneous pipelining, dataflow optimization, and zero-copy execution deliver over 50% higher average inference speed than the vendor's rknn-llm SDK, with larger gains at longer contexts: up to **2.33×** for prefill and **3.56×** for decode. With a 3,800-token history, Qwen3-4B generates 128 tokens in 15.6 seconds, down from 55.4 seconds.
- 🎯 **Better accuracy than the vendor SDK**: Advanced quantization algorithms reduce WikiText-2 perplexity by 2.64% for Qwen3-4B and 2.28% for MiniCPM5-2B compared with rknn-llm W8A8.
- 🔓 **Fully open source**: Built on our high-performance, open-source RKNPU2 operator library, [`rknpu2_matmul_open`](https://github.com/happyme531/rknpu2_matmul_open/). The entire inference pipeline runs without proprietary vendor libraries, so you can study it, optimize it, and add support for new models.
- 📦 **Mixed W4/W8 quantization**: A mix of 4-bit and 8-bit weights with dynamic 8-bit activations uses less memory than the vendor's W8A8 quantization. Weights are loaded on demand through mmap; Qwen3-4B reaches a peak RSS of approximately 3.4 GiB with a 4K context.
- 🧩 **Ready to use**: A single binary provides a CLI, a terminal chat TUI, and an OpenAI-compatible API, with support for streaming, thinking, function calling, and prefix caching.

## Performance: Compared with rknn-llm 1.3.1

### Qwen3-4B

| Test | rknn-llm 1.3.1 | open_rknpu2_lm_infer | Speedup |
|---|---:|---:|---:|
| pp512 | 105.037 ± 0.031 | 169.711 ± 0.127 | +61.6% |
| pp1024 | 94.619 ± 0.021 | 157.033 ± 0.160 | +66.0% |
| pp4096 | 50.133 ± 0.084 | 116.973 ± 0.050 | +133.3% |
| decode128 @512 | 5.772 ± 0.002 | 10.497 ± 0.003 | +81.8% |
| decode128 @1024 | 5.125 ± 0.004 | 10.090 ± 0.004 | +96.9% |
| decode128 @3800 | 2.310 ± 0.005 | 8.226 ± 0.003 | +256.0% |

### Granite-4.2-3B

| Test | rknn-llm 1.3.1 | open_rknpu2_lm_infer | Speedup |
|---|---:|---:|---:|
| pp512 | 120.459 ± 0.107 | 182.476 ± 0.114 | +51.5% |
| pp1024 | 106.277 ± 0.019 | 164.162 ± 0.240 | +54.5% |
| pp4096 | 54.404 ± 0.004 | 110.307 ± 0.066 | +102.8% |
| decode128 @512 | 6.806 ± 0.003 | 9.295 ± 0.012 | +36.6% |
| decode128 @1024 | 6.370 ± 0.006 | 9.054 ± 0.004 | +42.1% |
| decode128 @3800 | 3.932 ± 0.001 | 7.727 ± 0.003 | +96.5% |

### MiniCPM5-2B

| Test | rknn-llm 1.3.1 | open_rknpu2_lm_infer | Speedup |
|---|---:|---:|---:|
| pp512 | 211.052 ± 0.484 | 245.465 ± 0.279 | +16.3% |
| pp1024 | 188.643 ± 0.082 | 219.865 ± 0.220 | +16.6% |
| pp4096 | 98.874 ± 0.009 | 173.578 ± 0.034 | +75.6% |
| decode128 @512 | 9.431 ± 0.006 | 13.151 ± 0.029 | +39.4% |
| decode128 @1024 | 8.985 ± 0.003 | 12.954 ± 0.005 | +44.2% |
| decode128 @3800 | 7.117 ± 0.003 | 11.613 ± 0.012 | +63.2% |

## Perplexity

| Model | rknn-llm W8A8 | open_rknpu2_lm_infer mixed quantization | PPL change (lower is better) |
|---|---:|---:|---:|
| Qwen3-4B | 12.8709 | 12.5317 | -2.64% |
| Granite-4.2-3B | 15.7448 | 16.0519 | +1.95% |
| MiniCPM5-2B | 14.1293 | 13.8073 | -2.28% |

## Supported Models

| Architecture | Validated checkpoint |
|---|---|
| qwen3 | [Qwen3-4B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/Qwen3-4B-W4A8-Mixed-QQQ-Rot) |
| granite | [Granite-4.2-3B-W4A8-Mixed-QQQ-SQ](https://huggingface.co/happyme531/Granite-4.2-3B-W4A8-Mixed-QQQ-SQ) |
| llama | [MiniCPM5-2B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/MiniCPM5-2B-W4A8-Mixed-QQQ-Rot) |

The engine supports models in the compressed-tensors format with W4A8 or W8A8 linear layers, including mixed configurations, using symmetric per-channel weight and per-token activation quantization. Use the models above as examples to try other model sizes, mixed quantization configurations, and quantization algorithms.

## Quick Start

Copy and paste the following commands on an RK3588 Linux board to download the latest [release](https://github.com/happyme531/open_rknpu2_lm_infer/releases/latest) and the Qwen3-4B model, then start a terminal chat session. Supports Debian 11 / glibc 2.31 or newer, with no compilation required.

```bash
sudo apt install curl jq python3-pip libgomp1 libstdc++6

curl -fL "$(curl -fsSL https://api.github.com/repos/happyme531/open_rknpu2_lm_infer/releases/latest | jq -r '.assets[] | select(.name | endswith("-linux-aarch64.tar.gz")) | .browser_download_url')" | tar -xz --strip-components=1
pip3 install 'huggingface_hub[cli]>=0.34,<1'
hf download happyme531/Qwen3-4B-W4A8-Mixed-QQQ-Rot --local-dir models/Qwen3-4B-W4A8-Mixed-QQQ-Rot
./bin/open_rknpu2_lm_infer --model models/Qwen3-4B-W4A8-Mixed-QQQ-Rot chat
```

Replace `chat` in the last command with `serve --port 8080` to start an OpenAI-compatible LLM API server.

## Getting full tuned performance

Inference reads existing tuning results and uses built-in strategies for missing entries. It **does not tune automatically**, including on the first request. Run the explicit preparation below once per model and resource configuration to reach tuned throughput and cover short chat tails. The first run can take several minutes; progress and the best candidate time are shown, and valid cache entries are reused on later runs.

The release archive includes `scripts/run_board.py`. It holds the shared NPU lock, fixes CPU/NPU/DDR clocks for the command, and restores them on exit. Run on the RK3588 board with working `sudo` access:

```sh
export RKNPU_MODEL=models/Qwen3-4B-W4A8-Mixed-QQQ-Rot
export RKNPU_TUNE_CACHE="$HOME/.cache/open_rknpu2_lm_infer/qwen3-4b-ub64"
sudo -v
python3 scripts/run_board.py --report tune-clocks.json -- \
  ./bin/open_rknpu2_lm_infer --ctx-size 8192 --ubatch-size 64 \
  --tune-cache "$RKNPU_TUNE_CACHE" tune --batches 1,2,4,8,16,32,64 --report tune.json
python3 scripts/run_board.py --report bench-clocks.json -- \
  ./bin/open_rknpu2_lm_infer --ctx-size 8192 --ubatch-size 64 \
  --tune-cache "$RKNPU_TUNE_CACHE" bench --suite contexts --repetitions 3 --report bench.json
./bin/open_rknpu2_lm_infer --ctx-size 8192 --ubatch-size 64 \
  --tune-cache "$RKNPU_TUNE_CACHE" chat
```

For a source build, replace `./bin/open_rknpu2_lm_infer` with `target/release/open_rknpu2_lm_infer`. Use the same model, context, batch size, CPU/NPU settings and cache directory when starting `chat` or `serve`. Keep separate cache directories for different models. For throughput-only preparation at batch64, `tune --batches 1,64` is sufficient; the other batches cover irregular chat tails. Batch128 is also supported: set `--ubatch-size 128` consistently and add 128 to `--batches`.

In a fully prepared benchmark, `stats.tune_calls`, `stats.weight_reload_calls` and `stats.strategy_fallback_calls` should all be 0. `tune_cache_misses` can include unused W8 partition candidates, so it is not a substitute for these checks. Both clock reports should have `exit_code: 0` and an empty `restore_errors` list. Empty-cache inference remains usable, but its built-in strategies may not reach the tabled performance. See [usage](llm_generated_docs/USAGE.md) and [performance protocol](llm_generated_docs/PERFORMANCE.md).

## Architecture

```
   CLI  ·  TUI  ·  OpenAI API (axum)
                 │
   Engine ── Dedicated inference thread · Request queue · Sampling · Prefix KV reuse
                 │
   LanguageModel ── Qwen3 / Llama / Granite
        │                         │
   CPU (Rust)                NPU (rk_npu_matmul, C ABI)
   RMSNorm · RoPE · Residual  W4A8/W8A8 Linear · FP16 GQA Attention
                                  │
                         Open-source RKNPU2 kernel driver
```


## Development

### Build from Source

Build on an RK3588 Linux board (rustup automatically installs Rust 1.95 as specified in `rust-toolchain.toml`):

```sh
sudo apt install build-essential cmake libclang-dev git
git clone --depth 1 https://github.com/happyme531/open_rknpu2_lm_infer.git
cd open_rknpu2_lm_infer
git submodule update --init --depth 1
cargo build --release --locked
```

Set the model directory, then run:

```sh
export RKNPU_MODEL=/path/to/Qwen3-4B-W4A8-Mixed-QQQ-Rot

# Single streaming generation
target/release/open_rknpu2_lm_infer run 'Hello, please introduce yourself.' --conversation -n 128
# Terminal chat
target/release/open_rknpu2_lm_infer chat
# OpenAI-compatible server
target/release/open_rknpu2_lm_infer serve --port 8080
```

Once the server is running, call it with the standard OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
stream = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "Introduce the RK3588 in two sentences."}],
    stream=True,
    extra_body={"enable_thinking": False},
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## License

[AGPL-3.0-only](LICENSE). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the operator library's provenance and licensing boundaries.
