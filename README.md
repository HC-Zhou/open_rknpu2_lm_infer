<div align="center">

# open_rknpu2_lm_infer

**完全开源的高性能 RK3588 NPU 大模型推理引擎**

![License](https://img.shields.io/badge/license-AGPL--3.0-blue)
![Rust](https://img.shields.io/badge/rust-1.95-orange)
![Platform](https://img.shields.io/badge/platform-RK3588-green)

## 亮点

- 🚀 **比原厂更快**：采用自动调优，异构流水执行，数据流优化，零拷贝等先进策略，平均推理速度比原厂 rknn-llm SDK 高50%以上，上下文越长优势越大，prefill 最高 **2.33×**，decode 最高 **3.56×**。在 3800-token 历史上，Qwen3-4B 生成 128 个 token 的时间从 55.4 秒降到 15.6 秒。
- 🎯 **比原厂更准**：采用先进的量化算法，Qwen3-4B 和 MiniCPM5-2B 的 WikiText-2 PPL 分别比 rknn-llm W8A8 低 2.64% 和 2.28%。
- 🔓 **完全开源**：基于自研高性能开源 RKNPU2 算子库 [`rknpu2_matmul_open`](https://github.com/happyme531/rknpu2_matmul_open/)，整个推理流程无任何原厂闭源库依赖，可随意研究/优化/添加新模型支持。
- 📦 **混合 W4/W8 量化**：4-bit/8-bit 权重配合 8-bit 动态激活的混合量化，相比原厂W8A8量化占用内存更低。权重 mmap 按需加载，Qwen3-4B 在 4K 上下文下峰值 RSS 约 3.4 GiB。
- 🧩 **开箱即用**：一个二进制同时提供 CLI、终端聊天 TUI 和 OpenAI 兼容 API，支持流式输出、thinking、function calling 和前缀缓存。

## 性能：对比 rknn-llm 1.3.1

### Qwen3-4B

| 测试 | rknn-llm 1.3.1 | open_rknpu2_lm_infer | 性能提升 |
|---|---:|---:|---:|
| pp512 | 105.037 ± 0.031 | 169.711 ± 0.127 | +61.6% |
| pp1024 | 94.619 ± 0.021 | 157.033 ± 0.160 | +66.0% |
| pp4096 | 50.133 ± 0.084 | 116.973 ± 0.050 | +133.3% |
| decode128 @512 | 5.772 ± 0.002 | 10.497 ± 0.003 | +81.8% |
| decode128 @1024 | 5.125 ± 0.004 | 10.090 ± 0.004 | +96.9% |
| decode128 @3800 | 2.310 ± 0.005 | 8.226 ± 0.003 | +256.0% |

### Granite-4.2-3B

| 测试 | rknn-llm 1.3.1 | open_rknpu2_lm_infer | 性能提升 |
|---|---:|---:|---:|
| pp512 | 120.459 ± 0.107 | 182.476 ± 0.114 | +51.5% |
| pp1024 | 106.277 ± 0.019 | 164.162 ± 0.240 | +54.5% |
| pp4096 | 54.404 ± 0.004 | 110.307 ± 0.066 | +102.8% |
| decode128 @512 | 6.806 ± 0.003 | 9.295 ± 0.012 | +36.6% |
| decode128 @1024 | 6.370 ± 0.006 | 9.054 ± 0.004 | +42.1% |
| decode128 @3800 | 3.932 ± 0.001 | 7.727 ± 0.003 | +96.5% |

### MiniCPM5-2B

| 测试 | rknn-llm 1.3.1 | open_rknpu2_lm_infer | 性能提升 |
|---|---:|---:|---:|
| pp512 | 211.052 ± 0.484 | 245.465 ± 0.279 | +16.3% |
| pp1024 | 188.643 ± 0.082 | 219.865 ± 0.220 | +16.6% |
| pp4096 | 98.874 ± 0.009 | 173.578 ± 0.034 | +75.6% |
| decode128 @512 | 9.431 ± 0.006 | 13.151 ± 0.029 | +39.4% |
| decode128 @1024 | 8.985 ± 0.003 | 12.954 ± 0.005 | +44.2% |
| decode128 @3800 | 7.117 ± 0.003 | 11.613 ± 0.012 | +63.2% |

## Perplexity

| 模型 | rknn-llm W8A8 | open_rknpu2_lm_infer 混合量化 | PPL 变化（越低越好） |
|---|---:|---:|---:|
| Qwen3-4B | 12.8709 | 12.5317 | -2.64% |
| Granite-4.2-3B | 15.7448 | 16.0519 | +1.95% |
| MiniCPM5-2B | 14.1293 | 13.8073 | -2.28% |

## 支持的模型

| 架构 | 已验证 checkpoint |
|---|---|
| qwen3 | [Qwen3-4B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/Qwen3-4B-W4A8-Mixed-QQQ-Rot) |
| granite | [Granite-4.2-3B-W4A8-Mixed-QQQ-SQ](https://huggingface.co/happyme531/Granite-4.2-3B-W4A8-Mixed-QQQ-SQ) |
| llama | [MiniCPM5-2B-W4A8-Mixed-QQQ-Rot](https://huggingface.co/happyme531/MiniCPM5-2B-W4A8-Mixed-QQQ-Rot) |

程序支持线性层使用w4a8或w8a8(可以混合使用)per-channel per-token对称量化的 compressed-tensors 格式的模型。因此你也可以根据以上示例模型，尝试加载其他大小的模型，试验不同的混合量化配置和量化算法。

## 快速开始

预编译 ARM64 版本见 [Releases](https://github.com/happyme531/open_rknpu2_lm_infer/releases/latest)。适用于 Debian 11 / glibc 2.31 或更新的系统；解压后运行 `bin/open_rknpu2_lm_infer`，模型独立提供。

在 RK3588 Linux 板上构建（rustup 会按 `rust-toolchain.toml` 自动安装 Rust 1.95）：

```sh
sudo apt install build-essential cmake libclang-dev git
git submodule update --init --depth 1
cargo build --release --locked
```

指定模型目录后即可使用：

```sh
export RKNPU_MODEL=/path/to/Qwen3-4B-W4A8-Mixed

# 单次流式生成
target/release/open_rknpu2_lm_infer run '你好，请介绍一下自己。' --chat-template -n 128
# 终端聊天
target/release/open_rknpu2_lm_infer chat
# OpenAI 兼容服务
target/release/open_rknpu2_lm_infer serve --port 8080
```

首次加载会自动调优 NPU 策略并缓存到 `~/.cache/open_rknpu2_lm_infer`，后续启动直接复用。

服务启动后，可以直接用标准 OpenAI SDK 调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
stream = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[{"role": "user", "content": "用两句话介绍 RK3588。"}],
    stream=True,
    extra_body={"enable_thinking": False},
)
for chunk in stream:
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

完整命令、参数和测速方法见 [USAGE.md](llm_generated_docs/USAGE.md)；API 支持范围见 [API.md](llm_generated_docs/API.md)。

程序默认使用 `OMP_WAIT_POLICY=PASSIVE`、`GOMP_SPINCOUNT=0`，与性能测试的 OpenMP 等待配置一致；用户显式设置的值优先。



## 架构

```
   CLI  ·  TUI  ·  OpenAI API (axum)
                 │
   Engine ── 专用推理线程 · 请求队列 · 采样 · 前缀 KV 复用
                 │
   LanguageModel ── Qwen3 / Llama / Granite
        │                         │
   CPU (Rust)                NPU (rk_npu_matmul, C ABI)
   RMSNorm · RoPE · 残差      W4A8/W8A8 Linear · FP16 GQA Attention
                                  │
                         开源 RKNPU2 内核驱动
```

| 路径 | 内容 |
|---|---|
| `src/engine.rs` | 生成引擎、请求调度、取消与 KV 复用 |
| `src/model/` | checkpoint 加载、dense 前向、CPU 算子 |
| `src/npu.rs` | NPU 后端：Linear、Attention、调优与工作区 |
| `src/cli.rs` `src/tui.rs` `src/api.rs` | 三个入口，共用同一引擎 |
| `src/bench.rs` `src/ppl.rs` | 测速与 PPL 评测 |
| `native/rk_npu_matmul/` | 固定提交的 [NPU 算子库 Git submodule](https://github.com/happyme531/rknpu2_matmul_open) |

设计细节见 [ARCHITECTURE.md](llm_generated_docs/ARCHITECTURE.md)。

## 开发

新增模型架构时，实现 `LanguageModel` trait 并在 `model::load` 中注册。CLI、TUI 和 API 无需改动。

提交前在板上运行：

```sh
cargo fmt --check
cargo clippy --all-targets --release --locked -- -D warnings
cargo test --release --locked
cargo test --release --locked --lib npu::tests -- --ignored --test-threads=1   # NPU 板端测试
```

涉及 NPU、KV 或前向的修改需要额外比较 logits 和测速，见 [CONTRIBUTING.md](CONTRIBUTING.md) 与 [VALIDATION.md](llm_generated_docs/VALIDATION.md)。

## 许可证

[AGPL-3.0-only](LICENSE)。算子库的来源与许可边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
