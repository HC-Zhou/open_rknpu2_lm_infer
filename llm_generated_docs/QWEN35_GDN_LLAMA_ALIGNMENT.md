# Qwen3.5 GDN 与 llama.cpp 对齐

日期：2026-10-17，Asia/Shanghai（UTC+08:00）。板端为备用1。

当前实现已改为独立 `native/cpu_kernels/` C++ 模块，由 Rust 安全接口调用。GDN 算子优化与实测已完成，用户随后授权的端到端验证也已完成，并修复了prefill逐query Attention。当前pp512/pp4096=130.911/124.487 tok/s，32窗PPL8.578578；详见 [QWEN35_VS_QWEN3_PERFORMANCE.md](QWEN35_VS_QWEN3_PERFORMANCE.md)。供NPU kernel实验使用的独立源码/数据快照见 [QWEN35_GDN_KERNEL_REPRO.md](QWEN35_GDN_KERNEL_REPRO.md)。

## 初版构建与发现

优化前测量二进制 SHA256 为 `e38e8f58293da177a4ff371d5083a0511350c15a66b3fc011b763bfad105197c`，是 Cargo release：`opt-level=3`、thin LTO、codegen-units=1。没有使用 debug build。CPU GDN mixer 的初版 pp512 耗时为 5608.81 ms，包含状态复制、卷积、激活、norm 和 recurrence；prefill 的 final norm / lm_head 已完全跳过，`head_ms=0`。

在独立 pp512 probe 上以 `perf record -e cpu-clock:u -F199` 采样，5,214 项、0 lost。活跃用户态 CPU 样本中，`gdn::run_head` 占 33.41%，`ConvState::run` 10.97%，`expf` 10.91%，`expf@plt` 2.17%。这些比例不是 wall time 或 NPU 等待时间比例。原始采样文件保留在板端 `/tmp/qwen35-cpu-profile-20261017.data`；文本与状态记录见基线证据目录。被观察进程正常退出时 perf 返回 143，数据正常刷出且 `perf report` 成功；该 probe wrapper 的退出码 1 如实保留，不计入正式计时样本。

## 算法与激活对齐

对照 `../lowlevel/integrations/llama.cpp` 的本地 revision `de3ebb93bd8828496faefb44ed45059ec7ed3ea5`，上游基点 `a17e94d229fc7fed1567a799baebd11b771fe5af`。所用 GDN CPU kernel 与上游基点一致。

| 部分 | 对照位置 | 算法与布局 |
|---|---|---|
| recurrent state | `ggml-cpu/ops.cpp` 的 `ggml_compute_forward_gated_delta_net_one_chunk` | 按 head 并行，窗口内逐 token；FP32 transposed state，decay → prediction → delta → outer update → output |
| NEON dot | `ggml-cpu/vec.cpp`、`simd-mappings.h` | 从单路累加改为四路独立累加，每次16个 float，同一 pairwise reduction；保留 decay/dot 与 update/dot 的融合 |
| SiLU | `ggml-cpu/vec.h` 的 `ggml_v_expf` / `ggml_v_silu` | 使用相同 FP32 多项式系数；极值回退稳定标量实现 |
| causal conv | `ggml_ssm_conv` / 独立 SiLU 的并行思路 | 独立 `conv/kernel.rs`；K=4 NEON，token 并行，直接读取因果窗口，最后一次性更新历史 |
| output gated norm | per-head norm、SiLU gate | 按 head/token 分组并行，门控激活向量化；norm 保持原 FP32 reduction 顺序 |

Q/K 的 safetensors head 顺序仍为 HF 的连续 repeat-interleave，即 `kh = h / (Hv/Hk)`。ggml op 直接广播采用 modulo；对照程序显式展开 Q/K 为 Hv 个 head，避免把模型参数顺序差异当作算法差异。矩形状态、标量 tail、compact replay 和事务提交保持独立模块。许可与上游来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 历史 Rust 实现与实际 ggml 算子的对照

对照程序直接链接上述 checkout 的 ggml CPU 库，调用 `ggml_gated_delta_net` 和实际 CPU graph compute，没有重写 C++ recurrence。CMake Release `-O3 -DNDEBUG -march=armv8.2-a+fp16+dotprod`；Rust 同样使用 release。两侧均 CPU4/`0xf0`，2256 MHz，GGML worker `poll=0`；由共享板锁串行执行。

使用 C7 的 Hk16/Hv32/Dk128/Dv128、归一化 Q/K、FP32 非零初始 state。每次从相同初态重新运行，包含状态复制、计算和输出布局；Rust 另含其公开接口输入检查。预热2次，7个样本；单 token 每样本100次，其余每样本5次。计时不含文件 I/O、输入生成和 ggml graph/plan/threadpool 建立。

| tokens | Rust 平均 ms/call | 实际 ggml 平均 ms/call | Rust 耗时变化 | output / final state |
|---:|---:|---:|---:|---|
| 1 | 0.285643 | 0.197741 | +44.45% | 均逐值相等 |
| 128 | 10.578046 | 12.091407 | −12.52% | 均逐值相等 |
| 512 | 43.934062 | 48.537896 | −9.49% | 均逐值相等 |

该结果只覆盖 recurrence 完整调用，不包含卷积、投影、Attention 或 lm_head。单 token 调用仍慢于 ggml，不能宣称所有形状都超过 llama.cpp；Rust 的初态复制、校验、调度和输出整理也计入其中。

## C++ 模块与优化过程

`native/cpu_kernels/gdn.cpp` 从 ggml 的按 head/window 循环改写为小型 C ABI；`vec.h` 保留四路 NEON reduction；`block.h` 负责 Dk128 的按 value row 分块。Rust 保留长度/数值验证、safetensors head 顺序、矩形维度和状态事务，删除原 Rust NEON recurrence。直接由 worker 复制各自初始 head，输出写入 token-major，去掉主线程整层复制和 head-major 输出转置。replay 复用相同状态更新，不计算 query/output。

同时保留原样四遍扫描 `qwen35_gdn_run_unfused` 和两遍融合用于对照。四遍算法不改变精度，但独立移植的调度与向量化并不自动等于 ggml 的耗时；正式结果使用实际编译的 ggml 作基线。

进一步比较了两行/四行共享 Q/K 加载，以及通用/`-mtune=cortex-a76` 调度。独立 C++ 调用按 unfused → ggml → fused → fused → ggml → unfused 顺序，固定4线程；每侧均有14个描述性样本。以下为融合调用平均 ms，输入、FP32 state、输出均与实际 ggml 逐值一致：

| 候选 | token1 | token128 | token512 |
|---|---:|---:|---:|
| 单行融合，通用调度 | 0.14128 | 9.49862 | 38.12081 |
| 四行共享，通用调度 | 0.13673 | 8.37726 | 33.38674 |
| 两行共享，通用调度 | 0.13143 | 7.67304 | 30.71467 |
| 四行共享，A76 调度 | 0.13191 | 7.55425 | 30.19388 |
| 两行共享，A76 调度 | 0.13335 | 7.78922 | 31.51190 |

最终保留四行共享和 A76 调度。`-mtune` 不改变 ISA 要求，不启用 fast-math，不降低 state 精度。矩形/短 value row 走通用 fallback。上述为优化候选的独立 C++ 调用；最终结果还通过实际 Rust 接口测量，包含检查和 ABI 开销。

## 最终 C++ 模块：实际 Rust 接口完整调用

以下经实际 Rust 安全接口调用 C++，包含输入检查、FFI、各 worker 的初态复制和 token-major 输出。固定备用1 CPU2256 MHz、mask0xf0；分别测1/4线程。每档预热2次、7个样本；计时不含输入生成、文件 I/O、ggml graph/plan/threadpool 建立。实际 ggml CPU op 直接调用作为基线，ggml worker poll=0。

| 线程 | tokens | 本实现 ms/call | ggml ms/call | 耗时变化 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0.373914 | 0.523282 | -28.54% |
| 1 | 2 | 0.627123 | 0.882624 | -28.95% |
| 1 | 4 | 1.092719 | 1.604294 | -31.89% |
| 1 | 7 | 1.765398 | 2.692236 | -34.43% |
| 1 | 8 | 1.993424 | 3.060572 | -34.87% |
| 1 | 32 | 7.480613 | 11.822273 | -36.72% |
| 1 | 128 | 29.480054 | 46.665526 | -36.83% |
| 1 | 512 | 121.499687 | 190.959262 | -36.37% |
| 4 | 1 | 0.133325 | 0.197835 | -32.61% |
| 4 | 2 | 0.198819 | 0.286707 | -30.65% |
| 4 | 4 | 0.322836 | 0.472995 | -31.75% |
| 4 | 7 | 0.520681 | 0.748391 | -30.43% |
| 4 | 8 | 0.579795 | 0.842768 | -31.20% |
| 4 | 32 | 2.117050 | 3.076924 | -31.20% |
| 4 | 128 | 8.448469 | 12.152299 | -30.48% |
| 4 | 512 | 33.955851 | 48.626169 | -30.17% |

16组形状/线程配置的输出和最终 FP32 state 均逐值相等。CPU/状态回归53项通过、7项原有硬件用例保持 ignored；release 全 target Clippy `-D warnings` 和 format 通过。矩形维度、128维的短 value row/tail、非零初态、极端 decay/beta、chunk/replay/事务边界均有覆盖。

两次正式微基准均正常退出，频率上下限和 governor 全部恢复。原始7个样本、候选对照、构建/测试日志、源码及二进制身份见 [evidence/qwen35_gdn_cpp_20261017/](evidence/qwen35_gdn_cpp_20261017/)，汇总入口为 [summary.json](evidence/qwen35_gdn_cpp_20261017/summary.json)。这些结果只涉及 FP32 GDN recurrence；不将其代替卷积/投影/Attention 或整模型性能。

模块重组后源码位于通用 `ops/`、`layers/` 和 `native/cpu_kernels/`；本报告原始计时和源码哈希保留其测量时身份，路径调整本身不代表重新测速。

## 复现

使用上述固定 revision 的 `llama.cpp/ggml` 源码，在备用板执行：

```sh
cmake -S scripts/gdn/ggml_reference -B /tmp/gdn-reference \
  -DGGML_SOURCE_DIR=/path/to/llama.cpp/ggml \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
  -DGGML_OPENMP=ON -DGGML_RKNPU2=OFF
cmake --build /tmp/gdn-reference -j4
cargo build --release --example gdn_bench
python3 scripts/run_board.py --wait-lock --cpu-khz 2256000 --skip-dmc \
  --report /tmp/gdn-clocks.json -- \
  taskset 0xf0 target/release/examples/gdn_bench \
  --reference-binary /tmp/gdn-reference/gdn_reference --output /tmp/gdn-comparison
```

程序对全部输出和最终 state 做比较并保存逐样本时间。`--threads 1` 可测单线程，默认4线程；覆盖 token1/2/4/7/8/32/128/512，可用 `--rows 1,128` 缩小范围。此前 Rust 版本的整模型历史记录见 [QWEN35_VS_QWEN3_PERFORMANCE.md](QWEN35_VS_QWEN3_PERFORMANCE.md)，不用于宣称当前 C++ 工作树的端到端结果。
