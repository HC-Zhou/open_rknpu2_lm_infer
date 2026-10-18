# GDN 前置 Conv 的 NPU offload 简测

2026-10-17，备用1；运行时代码基点 `ff70cf9`，本轮未修改模型实现。

建议：**独立 Conv offload 优先级较低；若能与 NPU GDN 的预处理衔接、减少中间回传，则值得一起做。** 本轮只测 CPU 成本，尚无 NPU Conv 的实测收益。

每个 GDN 层为 **8192 通道、K=4 的 depthwise causal Conv + SiLU**，共24层。当前生产路径已经使用 AArch64 NEON。

| 窗口 | 24层 CPU 总耗时，两次运行均值范围 | 单层耗时 | 对照现有整模型耗时的估算占比 |
|---|---:|---:|---:|
| decode，1 token | 2.95–3.06 ms | 123–127 μs | 2.64–2.74% |
| prefill，128 tokens | 29.87–31.29 ms | 1.245–1.304 ms | 3.01–3.15% |

测量口径是历史复制 + 生产 `ConvState::run_parallel`（Conv、SiLU、历史更新），不含投影、Q/K 拆分归一化、GDN recurrence 或输出门控。24层逐层执行，复用输出 scratch。4线程池、大核 `0xf0`、CPU固定2256 MHz；Release，Rust 1.95，`opt-level=3, lto=thin, codegen-units=1`，默认 AArch64 target，与生产 Rust 构建一致。NPU未参与运算；DMC不可控。共享锁、频率恢复均正常。

输入来自 C7 真实权重和已保存的24行运行时 QKV trace；前3行初始化非零历史，剩余21行在长窗口中循环。因此128行是组件重放，并非完整128-token模型输入。每档预热3次、自适应循环、7个计时样本，完整流程重复两次；原始数据另含8/32 tokens、单线程及独立历史维护诊断。

与标量参考逐层对照，所有窗口/线程配置通过：最大绝对误差 `1.43e-6`，relative L2 不超过 `6.95e-8`，最终历史完全一致。

占比使用[已有 CPU GDN 整模型基线](evidence/qwen35_vs_rkllm_official_20261017/matched-pg128-64.json)：pp128均值991.94 ms，decode每token111.75 ms。它们是跨运行估算；即使这段成本完全消失，旧基线吞吐的理想提升也只有约2.7–3.3%。Conv约占旧 GDN mixer 的8%（prefill）/25%（decode）；**NPU GDN 接入后应重新计算占比**。

单独 offload 时，每层包含布局转换、传输、提交、同步和历史维护的完整调用，需低于 CPU 的约0.12 ms（decode）/1.25 ms（128行），才有收益。当前 Conv 后仍有 CPU Q/K 归一化；只搬 Conv 会增加设备边界。更合适的下一步是评估 `Conv + SiLU + Q/K norm → NPU GDN` 的连续路径，连同常驻历史一起测量。本轮没有测这条链路，也未重跑端到端/PPL。

原始计时、正确性、频率恢复和占比计算保存在 [evidence/qwen35_conv_20261017/](evidence/qwen35_conv_20261017/)，主结果为 `results-release{,-repeat}.json`，汇总为 `summary.json`。独立诊断计时受缓存和调度影响，不能相减当作精确阶段拆分。

复现入口：[权重/trace导出](../scripts/qwen35/export_conv_bench.py)、[CPU重放程序](../tests/tools/causal_conv_bench.rs)。原始权重、trace和实测二进制保留在本机共享目录 `/home/zt/mnt/3090x4/tmp/qwen35-dev/conv-probe-20261017/`，板端对应 `/home/firefly/mnt/3090x4/tmp/qwen35-dev/conv-probe-20261017/`，未拷入仓库。导出来源逐层记录于 `input-metadata.json`。

```sh
python3 scripts/qwen35/export_conv_bench.py --model /path/to/C7 \
  --trace /path/to/trace32 --output /path/to/conv-inputs
cargo build --release --example causal_conv_bench
python3 scripts/run_board.py --wait-lock --cpu-khz 2256000 --skip-dmc --report clocks.json -- \
  taskset 0xf0 target/release/examples/causal_conv_bench \
  /path/to/conv-inputs/weights.f32 /path/to/conv-inputs/trace.f32 results.json
```

实测为避免重编无关 NPU 库，直接用板端 `rustc` 将上述 example 的库导入替换为生产 `src/ops/mod.rs` 的路径导入，复用已有 anyhow/rayon/serde_json rlib，优化参数如上；算子源码不变。
