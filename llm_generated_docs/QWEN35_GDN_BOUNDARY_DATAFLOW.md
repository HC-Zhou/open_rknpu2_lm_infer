# Qwen3.5 GDN 边界数据流优化

2026-10-18，RK3588 C7，普通 TF32，FP32 state。

pp512 配对测试的 GDN 完整阶段 **842.583 → 760.984 ms，减少 9.68%**；完整模型 prefill **3222.814 → 3118.115 ms，减少 3.25%**，吞吐 **158.867 → 164.202 token/s，提升 3.36%**。算子库外部开销 **163.792 → 80.060 ms，减少 51.12%**。

## pp512 分解

各项分别取六样本中位数，单位 ms。差值项先逐样本相减再取中位数，因此表内中位数不能直接求和。

| 操作 | 修改前 | 修改后 |
|---|---:|---:|
| 投影拆分复制 | 68.010 | 0.019 |
| 输出复制 | 16.016 | 0.000 |
| gate 生成 | 18.846 | 17.151 |
| gated RMSNorm | 54.560 | 56.574 |
| 层 prepare | 1.441 | 1.322 |
| 缓冲区 resize | 0.027 | 0.017 |
| 外部未分项余量 | 5.337 | 4.971 |
| native 完整调用 | 679.145 | 681.046 |
| GDN 完整阶段 | 842.583 | 760.984 |
| 算子库外部开销 | 163.792 | 80.060 |

NPU hardware 为 **178.469 → 178.576 ms**，native 完整调用基本持平。主要收益来自消除 84 ms 左右的两处复制。gated RMSNorm 为 54.560 → 56.574 ms，本组数据未显示收益。CPU Conv/SiLU/QK L2 producer 为 168.915 → 177.875 ms；带步长输入下这项变慢，但 native 总计增加 1.901 ms，GDN 整体仍减少 81.599 ms。

| 完整模型阶段 | 修改前 | 修改后 |
|---|---:|---:|
| Linear | 1848.733 | 1833.567 |
| GDN | 842.583 | 760.984 |
| Full Attention | 156.889 | 151.800 |
| 其余 CPU | 375.015 | 370.949 |
| 完整 pp512 | 3222.814 | 3118.115 |

全部正式样本，按各版本的执行顺序列出：

| 版本 | pp512 样本（ms） | 进程内中位数（ms） |
|---|---|---|
| 修改前 | 3394.495, 3257.040, 3261.770, 3188.588, 3187.323, 3187.433 | 3261.770 / 3187.433 |
| 修改后 | 3105.648, 3105.990, 3108.041, 3128.190, 3131.056, 3139.275 | 3105.990 / 3131.056 |

### native 内部

| 分项 | 修改前（ms） | 修改后（ms） |
|---|---:|---:|
| CPU Conv/SiLU/QK L2 producer | 168.915 | 177.875 |
| 衰减/系数构造 | 51.005 | 50.799 |
| 三角求解 | 87.967 | 85.369 |
| 输出收集 | 42.914 | 41.900 |
| 状态布局/同步/其余余量（差值） | 150.168 | 146.281 |
| NPU hardware | 178.469 | 178.576 |
| native 完整调用 | 679.145 | 681.046 |

## 实现

正常 prefill 和 CPU decode 直接读取合并投影的 token-major QKV/Z/a/b 视图。对 C7，每行步长为 12352 个 FP32 元素，QKV 占前 8192 个元素。新增的 `rk_npu_gdn_run_projected_strided` 只读取 QKV 前缀，原有紧凑输入 C ABI 继续可用。

Rust `Projected::from_merged` 保存各投影的视图和行步长；`LayerState::forward_into` 让 gated RMSNorm 直接写入 decoder 的 `attended` 缓冲区。正常推理省去每个 pp512 的 579 MiB 投影拆分和 192 MiB 输出复制，合计 771 MiB payload、1542 MiB 读写流量。QKV 紧凑缓冲区只在 trace 开启时临时分配，C7 ubatch128 的常驻 CPU scratch 少 4 MiB。

输出 gated RMSNorm 将向量 SiLU、FP32 缩放和权重乘法合并到最终写入，保持原有 FP32 归约和乘法次序；静态 `exp(a_log)` 在权重加载时计算。canonical state 继续为 FP32 `[value_head,value_channel,key_channel]`，普通 TF32、CPU/NPU 路由、chunk64 与卷积历史格式均沿用原配置。

## 测试与复现

板端 release 常规测试 72 项通过；Clippy 全目标 `-D warnings` 通过。CPU 测试覆盖不同维度、K3/K4、短长窗口、verify/replay；显式 NPU 测试覆盖多层 workspace 复用、M12/13/32/64/128 和 NPU→CPU decode 接续。native 测试覆盖普通/补偿 TF32、多核掩码、2048-token 窗口、in-place state/history、BUSY、非零 IOMMU domain 和 C 头文件。合并视图与紧凑输入的输出、state 和卷积历史逐 bit 一致，padding 用 NaN 验证不会被当成输入读取。

固定真实文本 128-token prompt 加 16 个 continuation 的 **17×248320，共 4,221,440 个 logits 全部有限，修改前后逐 bit 一致**。数值结论针对同一 NPU 配置修改前后的等价性；CPU FP32 对照与 PPL 结果见 [原 GDN 集成报告](QWEN35_NPU_GDN_PREFILL.md)。

trace 按需分配路径使用固定 13-token prompt 加 2 个 continuation 对照：744,960 个 logits 及 726 份 trace 张量（含 72 份 QKV）逐 bit 一致，常驻 CPU scratch 减少 4 MiB。

固定板端 CPU2.4GHz、NPU1GHz、DDR2.112GHz，CPU4/0xf0、NPU7。基线是本轮修改前冻结的 NPU 集成二进制，外层 HEAD 为 `6143940`、native HEAD 为 `a9a2e06`，包含已有 GDN 集成工作区改动。C7 权重、量化策略缓存、ctx640、ubatch128、内部 chunk64 相同。顺序为 before→after→after→before，每进程预热一次、正式三次，合计每版本六样本。bench 为固定种子的合成 token IDs，prefill 不跑 lm_head，不计加载/prepare/采样。所有 case 要求 0 cache miss、0 在线搜索和 0 回退。

板端常规检查使用 `cargo fmt --check`、`cargo test --release --locked` 和 `cargo clippy --all-targets --release --locked -- -D warnings`；显式运行 `npu_prefill_threshold_state_handoff_and_layer_reuse` ignored 测试，以及 native `test_gdn_static --device` 和 `test_gdn_header_c`。

整模型配对脚本为 [compare.py](../scripts/archive/gdn_boundary_20261018/compare.py)。运行目录为项目根目录，并通过 `scripts/run_board.py --wait-lock` 持锁固定频率。对照脚本通过 `--baseline` 和 `--candidate` 指定已构建的二进制，`--reference-root` 指向含固定 token 输入的原实验目录。

完整样本、各调用统计和恢复记录位于忽略的 `logs/gdn-boundary-dataflow/`；紧凑副本保存在忽略目录 `llm_generated_docs/evidence/qwen35_gdn_boundary_20261018/`。所有正式运行均为 0 cache miss、0 在线搜索、0 回退。测试和配对 runner 均返回 0，频率上下限与 governor 已恢复，`restore_errors=[]`。

## 适用范围

这些数字对应 C7 pp512、ubatch128、普通 TF32。两个进程的基线中位数相差约 2.3%，完整模型收益受 Linear 等阶段波动影响；GDN 分项的变化更直接。每版本只有两个进程，进程内三次采样不是六个独立进程。

当前 native 调用占 GDN 约 89.5%。状态/同步/其余未分项余量为 146.281 ms，仍是逐样本差值。

TODO：分别测量 state 导入、导出与同步成本，再评估每层保留 native state、在 CPU decode 前导出一次 canonical state 的收益。Conv/SiLU/QK L2 producer 的数据流优化也留待后续。
