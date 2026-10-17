# Qwen3.5 / MTP 实现进度

日期：2026-10-17，Asia/Shanghai（UTC+08:00）。

已完成本阶段的 Qwen3.5 混合文本 AR 主干、CPU GDN 优化和备用1验证。最新 C++ GDN / 批量 Attention 版本在固定前 32 个 WikiText 窗口、8,160 个评分 token 上的 PPL 为 **8.578578**；同权重 FP32 参考为 **8.601783**，相对差异 **−0.269768%**。首轮 `ubatch=32/128` 的 PPL 均为 8.594177，历史结果单独保留。详细交接、范围和复现命令见 [QWEN35_C7_BOARD_VALIDATION.md](QWEN35_C7_BOARD_VALIDATION.md)。本阶段完成不代表整个 MTP 实现计划完成。

本轮按用户确认，以端到端 PPL 验证整模型精度；小幅浮点漂移不阻塞文本主干。严格 logits 诊断保留原门槛和未通过记录，不据此宣称逐值等价。C7 后续补入15个未量化 BF16 MTP tensor，source 原值未变，target coordinate alignment 待完成；MTP driver 和视觉路径尚未实现。

GDN 算子已完成与实际 llama.cpp 的直接对照：C++ 模块经 Rust 接口调用，4线程下各窗口耗时低约30%–33%，输出与最终状态逐值一致。用户随后授权恢复端到端测试；当前已完成C++版本并修复prefill逐query Attention的遗留路径。最新同板pp512/pp4096为130.911/124.487 tok/s，Qwen3为180.740/120.497 tok/s；32窗PPL8.578578，批量Attention前后8160项NLL逐值相等。整模型数据见 [QWEN35_VS_QWEN3_PERFORMANCE.md](QWEN35_VS_QWEN3_PERFORMANCE.md)。算子对照见 [QWEN35_GDN_LLAMA_ALIGNMENT.md](QWEN35_GDN_LLAMA_ALIGNMENT.md)。

后续已按公共算子、GDN 层和模型适配重组代码，并让 Dense/Qwen3.5 共用输出投影与 trace。重构前后两模型各5行 logits 位级一致，各2窗/510项 NLL逐项相等；release54项测试和额外4个NPU测试通过。上述完整32窗PPL和性能数字保留重构前测量身份，重构回归见 [RUNTIME_MODULE_LAYOUT.md](RUNTIME_MODULE_LAYOUT.md)。

## 模块边界

| 模块 | 责任 |
|---|---|
| `src/model/qwen35/config.rs`、`weights.rs` | Qwen 配置、层类型、权重名称/形状和量化声明来源 |
| `src/model/qwen35/rope.rs`、`attention.rs` | Qwen RoPE JSON 默认值/限制、interleaved Q/gate 布局 |
| `src/ops/activation.rs`、`norm.rs`、`rope.rs` | 通用激活、归一化及数值 M-RoPE；不解析模型配置 |
| `src/ops/causal_conv.rs`、`causal_conv/` | depthwise causal conv + SiLU、历史追加和 NEON |
| `src/ops/gated_delta.rs`、`gated_delta/`、`native/cpu_kernels/` | GDN recurrence、Rust 安全 ABI、标量 oracle 和 replay |
| `src/layers/gated_delta/` | 投影后的完整 GDN mixer、每层 committed state/compact log、共享 scratch；不加载模型权重 |
| `src/model/output_head.rs`、`trace.rs` | Dense/Qwen3.5 共用输出投影与 trace；归一化、权重选择和缩放留给模型 |
| `src/model/qwen35/load.rs`、`head.rs` | C7 量化加载、IOMMU 分域、模型特有的输出头选择/最终 norm |
| `src/model/qwen35/decoder.rs`、`decoder/mixers.rs` | hybrid layer 编排、共享缓冲、残差/MLP |
| `src/model/qwen35/runtime.rs`、`stats.rs`、`timings.rs` | LanguageModel、取消恢复、模型统计和计时 |
| `src/model/projection.rs`、`metrics.rs`、`metadata.rs` | 共享量化投影、统计和模型元数据 |
| `src/npu/attention.rs`、`policy.rs` | 原生 Attention、窄 W8 投影核数选择 |
| `src/speculative.rs` | pending-anchor 预算和 greedy 接受账本 |
| `tests/cpu_layers/`、`examples/gdn_bench.rs`、`scripts/gdn/` | 公共 CPU 层测试、ggml 对照与独立 kernel 复现 |

`model::load` 注册 `qwen3_5` / `qwen3_5_text`。原 `dense.rs` 的投影与统计抽为公共模块，原 `npu.rs` 的 Attention 实现移入子模块；不把 hybrid decoder 堆进已有长文件。CLI 元数据解析同样移入共享模块。通用算子/层不反向依赖 `model::qwen35`；完整职责边界见 [RUNTIME_MODULE_LAYOUT.md](RUNTIME_MODULE_LAYOUT.md)。

文本请求的中途失败/取消会重置全部 GDN 与 KV；较短的非零前缀截断返回 false，由 engine 重算，不冒充已经具备 GDN 任意回滚。普通请求仅准备/读取策略，不进行在线搜索或重载权重。

## 固定的参考与语义

- 官方 Qwen3.5-4B 配置/tensor 索引：revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。
- Transformers：revision `09281b8d56b08199ce9f397c694f38853cc45278`，执行其原始 Python AST 中的 recurrent/chunk GDN、causal conv 和 norm 函数生成 fixture。
- llama.cpp：上游 revision `a17e94d229fc7fed1567a799baebd11b771fe5af`，本地 checkout 为 `de3ebb93bd8828496faefb44ed45059ec7ed3ea5`；所参考的 GDN kernel 在两者之间无改动。复用其 transposed state 与按 head 连续处理 token 窗口的循环组织，许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
- vLLM MTP：revision `ae8f41238cc7316781ef4359890940749735b209`，目前仅做网络清单参考；target seed、prompt shift 和 draft 修复尚待可执行 fixture。

GDN canonical state 为 FP32 `[value_head, value_channel, key_channel]`，与 HF `[value_head, key_channel, value_channel]` 互为转置。Q/K 使用 `h / (Hv/Hk)` 连续重复映射，与 safetensors 投影行顺序一致。卷积历史为 `[channel, oldest..newest]` 的 K-1 项，保留提交边界之前的真实投影输入。

GDN Q/K L2 epsilon 固定为 `1e-6`，query scaling 使用 `1/sqrt(key_dim)`；支持矩形 K/V 维度以便检出矩阵转置错误。GDN output norm 使用原始 weight 和 SiLU(z)；decoder/QK norm 使用 `1+weight`；full-attention output gate 使用 sigmoid。

`LayerState::verify` 不修改 committed state/history/position。`finish_round(n)` 按真实消费行数提交，只重放 normalized key、post-convolution value、log_decay、beta，并追加原始卷积输入；不保留 query，不重做投影。`n=1+j`，j=0 时仍提交 anchor；取消使用 `abort_round()`。此模块只保证单层事务；整模型 KV、hidden、draft 的原子提交仍待接入。

`RoundPlan` 的输出预算不包含已发送的 pending anchor。验证逻辑行数同时受 context 和剩余输出预算限制；NPU padding 不进入状态接口。接受账本不直接发送 token，最终 stop/EOS 和服务提交边界由后续 engine driver 处理。

量化声明位于外层或 `text_config` 均可解析；两处同时存在而不一致时显式报错。`tie_word_embeddings` 不覆盖显式存在的输出头。inventory 的 `--lm-head` 用于表示独立输出头。

## 验证与复现

小型参考数据位于 [evidence/qwen35_mtp/](evidence/qwen35_mtp/)。生成器不下载权重，不执行量化：

```sh
python scripts/qwen35/generate_fixtures.py
cargo run --release --example qwen35_inventory -- \
  llm_generated_docs/evidence/qwen35_mtp/official_config.json --verify-rows 8
cargo fmt --check
cargo clippy --all-targets --release --locked -- -D warnings
python scripts/run_board.py --wait-lock --report /tmp/qwen35-checks-clocks.json -- \
  cargo test --release --locked
```

FP32 数值门槛保持计划中的 `max_abs <= 1e-5 + 1e-4*max_abs(reference)` 和 relative L2 <= `1e-4`。同路径 chunk/rollback/replay 的 committed state 与卷积历史要求逐值相等。

覆盖非零初始状态、矩形维度、Hk=Hv 与 Hk<Hv、128 维 NEON、标量 tail、长度 1/2/3/4/5/7/8/31/32/33/127/128/129、强衰减、beta 0/1、不同 chunk 划分、K=1/2/3/4/7 的所有接受边界、跨层共享 scratch、重复取消和 reset 后重算。

不依赖 checkpoint 的首轮验证（备用1，`firefly-rk3588-node01`，AArch64 / Linux 5.10.198 / Rust 1.95.0）：

- `cargo fmt --check`、全 target release Clippy（`-D warnings`）通过。
- Rust 默认测试 48 passed、0 failed、6 ignored；其中新增 Qwen3.5/协议测试 15 个。6 个原有硬件/真实模型用例保持 ignored，未计入通过数量。
- Python 板端 runner 测试 2 passed，覆盖主板默认参数、备用板参数与子命令失败后的频率恢复。
- `qwen35_inventory` 示例执行通过，输出 426 个文本 tensor、15 个 MTP tensor；4B/8 行窗口的 GDN canonical state 为 48 MiB、卷积历史 2.25 MiB、compact log 为 10.55 MiB。此为 GDN 持久状态与日志预算，不含共享 scratch、KV、hidden、模型权重和输出头。结果见 [inventory_summary.json](evidence/qwen35_mtp/inventory_summary.json)。
- Rust 测试持有共享板锁，CPU mask `0xf0`，大核固定 2256000 kHz、NPU 固定 1000000000 Hz。备用1没有 DMC devfreq，本次使用 `--cpu-khz 2256000 --skip-dmc`；DDR 未受控，不提供性能结论。
- `restore_errors=[]`，测试前后全部受控频率上下限与 governor 相同。板载 UTC 时钟为 2024 年，报告日期以主机为准，未修改板端系统时钟。

首轮结构化记录：[validation.json](evidence/qwen35_mtp/validation.json)。原始测试输出：[qwen35-tests.log](evidence/qwen35_mtp/qwen35-tests.log)，Clippy：[qwen35-clippy.log](evidence/qwen35_mtp/qwen35-clippy.log)，时钟：[qwen35-checks-clocks.json](evidence/qwen35_mtp/qwen35-checks-clocks.json)。真实 C7 的结果和后续回归单列在 [evidence/qwen35_mtp/c7/](evidence/qwen35_mtp/c7/)，避免覆盖首轮证据。

C7 首次接入后的默认测试为 51 passed、7 ignored；当时额外显式执行 4 个 NPU 用例全部通过。format、全 target release Clippy、2 个 Python runner 测试通过。默认 ubatch 的 CLI 算术示例输出 `19` 并正常 EOS；5 次 runner 的时钟恢复记录均通过核对。该轮结构化记录：[c7/validation.json](evidence/qwen35_mtp/c7/validation.json)。

最新 C++ GDN / 批量 Attention 版本的 release 默认测试为 **54 passed、0 failed、7 ignored**；额外显式执行 `npu::tests::generic_attention_geometries_match_cpu` 通过，覆盖 C7 的 16/4 heads、D256 几何。format 和全 target release Clippy（`-D warnings`）通过。最新性能、PPL、head-free prefill 连续性及4次时钟恢复结果见 [validation.json](evidence/qwen35_batch_attn_20261017/validation.json)，测试原始输出见 [qwen35-batch-attn-tests.log](evidence/qwen35_batch_attn_20261017/qwen35-batch-attn-tests.log)。

独立GDN kernel/NPU实验交接见 [QWEN35_GDN_KERNEL_REPRO.md](QWEN35_GDN_KERNEL_REPRO.md)，数据包含11个NPZ及可独立编译的CPU源码快照。

本轮备用板构建副本为 `/tmp/qwen35-runtime`。临时 Rust 工具链为 `/tmp/qwen35-dev-tools/toolchain/bin`；构建时设置 `LIBCLANG_PATH=/tmp/qwen35-dev-tools/clang/usr/lib/aarch64-linux-gnu` 和 `BINDGEN_EXTRA_CLANG_ARGS=-resource-dir=/tmp/qwen35-dev-tools/clang/usr/lib/llvm-15/lib/clang/15.0.6`。Cargo 依赖使用该副本 `.cargo/config.toml` 中的离线 vendor；这些开发环境文件不进入主仓库。

## 阶段状态与后续工作

| 阶段 | implementation_status | correctness_status | board_status | performance_status |
|---|---|---|---|---|
| P0 | PARTIAL：revision、清单、CPU fixtures 和 C7 文本交接已落地 | 文本清单与 CPU fixture PASS；C7 MTP 为未对齐的 source BF16，seed/shift fixture 待完成 | TEXT_CHECKPOINT_LOADED | NOT_MEASURED |
| P1 | 文本 hybrid decoder、NPU 投影/批量Attention、普通 decode/eval/PPL 已接入 | CPU oracle PASS；真实 C7 32 窗 PPL PASS；严格整模型 logits 诊断保留 | TEXT_PPL_PASS | SAME_BOARD_AR_MEASURED：pp512/4096=130.911/124.487 tok/s |
| P2 | PARTIAL：NEON、单层 compact replay/abort 与共享 scratch 已实现；整模型事务待实现 | 单层接受边界、下一次输出、跨层 scratch 与取消 PASS | COMPONENTS_PASS | NOT_MEASURED |
| P3 | NOT_IMPLEMENTED：MTP 网络、prompt prefill、seed 与 draft KV 修复待实现 | NOT_RUN | NOT_RUN | NOT_MEASURED |
| P4 | PARTIAL：pending-anchor 预算、greedy 接受账本、多行输出头评分；target hidden 接口和 MTP driver 待接入 | 小词表协议 oracle PASS；真实文本 PPL PASS；真实 MTP 未验证 | TEXT_PPL_PASS | NOT_MEASURED |
| P5–P8 | NOT_IMPLEMENTED | NOT_RUN | NOT_RUN | NOT_MEASURED |
| P9 | NOT_IMPLEMENTED | NOT_RUN | NOT_RUN | NOT_MEASURED |

后续增加 target hidden/MTP prefill fixture、完整模型事务和 MTP driver。真实 draft 验证需要与 C7 residual rotation 一致的 MTP 权重，不能直接拼入原始 source 的未旋转 MTP。视觉、随机拒绝采样、服务集成及性能 gate 独立推进；现阶段不作真实 MTP 接受率或速度结论。
